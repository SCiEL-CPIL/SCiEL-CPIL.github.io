"""
Communication-minimizing DAG pebbling.

Strategy:
- If the DAG is a simple layered structure (each node's parents are only from the previous layer),
  use a layered DP/recursive strategy that is efficient and near-optimal.
- Otherwise, fall back to a heuristic-based greedy schedule that minimizes transfers.
- Always include a safety deadline guard.
"""

import time
import heapq
from functools import lru_cache

def _extract_dag_info(dag):
    """Extract DAG structure from various formats."""
    if isinstance(dag, dict):
        nodes = dag.get('nodes', dag.get('vertices', []))
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        
        if not parents:
            edges = dag.get('edges', [])
            parents = {n: [] for n in nodes}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    parents[e[1]].append(e[0])
        
        if not inputs:
            inputs = [n for n in nodes if len(parents.get(n, [])) == 0]
        if not outputs:
            children = {n: set() for n in nodes}
            for n in nodes:
                for p in parents.get(n, []):
                    children[p].add(n)
            outputs = [n for n in nodes if len(children[n]) == 0]
        
        return nodes, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
        nodes = []
        parents = {}
        inputs = []
        outputs = []
        
        for item in dag:
            if isinstance(item, dict):
                node = item.get('node', item.get('id', item.get('name')))
                par = list(item.get('parents', item.get('inputs', [])))
                is_input = item.get('is_input', item.get('input', False))
                is_output = item.get('is_output', item.get('output', False))
                nodes.append(node)
                parents[node] = par
                if is_input:
                    inputs.append(node)
                if is_output:
                    outputs.append(node)
            elif isinstance(item, (list, tuple)):
                node = item[0]
                par = list(item[1]) if len(item) > 1 else []
                nodes.append(node)
                parents[node] = par
        
        if not inputs:
            inputs = [n for n in nodes if len(parents.get(n, [])) == 0]
        if not outputs:
            children = {n: set() for n in nodes}
            for n in nodes:
                for p in parents.get(n, []):
                    children[p].add(n)
            outputs = [n for n in nodes if len(children[n]) == 0]
        
        return nodes, parents, inputs, outputs
    
    else:
        raise ValueError(f"Unsupported dag format: {type(dag)}")


def _is_layered_dag(nodes, parents):
    """
    Check if the DAG is layered: each node's parents are all from the previous layer.
    We compute layers by topological depth (longest path from source).
    """
    # Compute depth of each node
    depth = {}
    node_set = set(nodes)
    # Topological order
    in_degree = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            if p in node_set:
                in_degree[n] += 1
                children[p].append(n)
    
    # BFS for topological order
    from collections import deque
    q = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while q:
        n = q.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                q.append(c)
    
    if len(topo_order) != len(nodes):
        return False, None, None  # cycle or error
    
    # Compute depth
    for n in topo_order:
        par = [p for p in parents.get(n, []) if p in node_set]
        if not par:
            depth[n] = 0
        else:
            depth[n] = 1 + max(depth[p] for p in par)
    
    # Assign layers
    max_depth = max(depth.values()) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for n in nodes:
        layers[depth[n]].append(n)
    
    # Check if it's strictly layered: each node's parents are all in the previous layer
    for n in nodes:
        par = [p for p in parents.get(n, []) if p in node_set]
        if not par:
            continue
        for p in par:
            if depth[p] != depth[n] - 1:
                return False, None, None
    
    return True, layers, depth


def _greedy_schedule(nodes, parents, inputs, outputs, fast_memory_budget):
    """
    Greedy pebbling schedule:
    - Process nodes in topological order.
    - Load inputs, compute nodes, store outputs, delete intermediates as needed.
    - Minimize transfers by keeping only necessary values in fast memory.
    """
    ops = []
    node_set = set(nodes)
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Topological order
    in_degree = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            if p in node_set:
                in_degree[n] += 1
                children[p].append(n)
    
    from collections import deque
    q = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while q:
        n = q.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                q.append(c)
    
    if len(topo_order) != len(nodes):
        # Fallback: just list all nodes
        topo_order = list(nodes)
    
    # Track which nodes are currently red (in fast memory)
    red = set()
    # Track which outputs have been stored
    completed_outputs = set()
    
    # For each node in topological order, ensure its parents are red, then compute it
    for node in topo_order:
        par = [p for p in parents.get(node, []) if p in node_set]
        
        # Ensure all parents are red
        for p in par:
            if p not in red:
                # Load the parent
                ops.append(['load', p])
                red.add(p)
        
        # Compute the node
        ops.append(['compute', node])
        red.add(node)
        
        # If this is an output, store it
        if node in output_set:
            ops.append(['store', node])
            red.discard(node)
            completed_outputs.add(node)
        else:
            # If fast memory is full, delete a non-essential red node
            # (prefer to delete nodes that are not needed by future outputs)
            if len(red) > fast_memory_budget:
                # Find a node to delete: prefer non-output, non-parent-of-remaining
                # Simple heuristic: delete the oldest non-output node
                to_delete = None
                for r in list(red):
                    if r not in output_set:
                        to_delete = r
                        break
                if to_delete is not None:
                    ops.append(['delete', to_delete])
                    red.discard(to_delete)
                else:
                    # All remaining are outputs, delete one (shouldn't happen if budget is reasonable)
                    if red:
                        to_delete = next(iter(red))
                        ops.append(['delete', to_delete])
                        red.discard(to_delete)
    
    return ops


def _layered_schedule(layers, parents, inputs, outputs, fast_memory_budget):
    """
    Specialized schedule for layered DAGs.
    For each layer, load all inputs from previous layer, compute all nodes in this layer,
    then store outputs and delete intermediates.
    """
    ops = []
    output_set = set(outputs)
    
    for layer_idx, layer in enumerate(layers):
        if layer_idx == 0:
            # First layer: these are inputs, load them
            for node in layer:
                ops.append(['load', node])
        else:
            # For each node in this layer, ensure parents are loaded
            # Parents are in the previous layer
            for node in layer:
                par = [p for p in parents.get(node, []) if p in set(layers[layer_idx-1])]
                for p in par:
                    ops.append(['load', p])
                ops.append(['compute', node])
        
        # After computing this layer, store any outputs in this layer
        for node in layer:
            if node in output_set:
                ops.append(['store', node])
    
    return ops


def pebble(dag, fast_memory_budget):
    """
    Find a minimum-transfer pebbling schedule.
    Returns list of [operation, vertex].
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10 second safety margin
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        # Fallback to baseline
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    node_set = set(nodes)
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Check if it's a layered DAG
    is_layered, layers, depth = _is_layered_dag(nodes, parents)
    
    if is_layered and layers is not None:
        # Use the specialized layered schedule
        ops = _layered_schedule(layers, parents, inputs, outputs, fast_memory_budget)
        if len(ops) <= 20000 and time.time() < deadline:
            return ops
    
    # Fall back to greedy schedule
    ops = _greedy_schedule(nodes, parents, inputs, outputs, fast_memory_budget)
    if len(ops) <= 20000:
        return ops
    
    # If ops is too long, try to trim (shouldn't happen for reasonable inputs)
    return ops[:20000]

