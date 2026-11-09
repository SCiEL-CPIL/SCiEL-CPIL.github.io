"""Communication-minimizing DAG pebbling with layered divide-and-conquer."""
import time
import heapq
from collections import defaultdict

def _extract_dag_info(dag):
    """
    Extract DAG structure. Handles common formats:
    - dag as dict with 'nodes', 'edges', 'inputs', 'outputs', 'layers'
    - dag as list of (node, parents) tuples
    - dag as list of dicts
    """
    if isinstance(dag, dict):
        nodes = dag.get('nodes', [])
        if not nodes and 'vertices' in dag:
            nodes = dag['vertices']
        edges = dag.get('edges', [])
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        parents = dag.get('parents', {})
        
        # Build parent map if not provided
        if not parents:
            parents = {n: [] for n in nodes}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    parents[e[1]].append(e[0])
        
        # Determine inputs/outputs if not provided
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
                par = item.get('parents', item.get('inputs', []))
                is_input = item.get('is_input', item.get('input', False))
                is_output = item.get('is_output', item.get('output', False))
                nodes.append(node)
                parents[node] = list(par)
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

def _layered_pebble(dag, fast_memory_budget):
    """
    Layered pebbling strategy: process layer by layer.
    For each layer, load needed inputs, compute, store outputs.
    This avoids the exponential state space of global search.
    """
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        return None
    
    if not nodes:
        return []
    
    # Build children map
    children = defaultdict(list)
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
    
    # Compute topological layers using BFS from inputs
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    layer_of = {}
    queue = []
    for n in nodes:
        if in_degree[n] == 0:
            layer_of[n] = 0
            queue.append(n)
    
    while queue:
        n = queue.pop(0)
        for c in children[n]:
            new_layer = layer_of[n] + 1
            if c not in layer_of or layer_of[c] < new_layer:
                layer_of[c] = new_layer
            queue.append(c)
    
    # Group nodes by layer
    layers = defaultdict(list)
    for n in nodes:
        layers[layer_of[n]].append(n)
    
    max_layer = max(layers.keys()) if layers else 0
    layer_order = sorted(layers.keys())
    
    # Pebbling strategy:
    # For each layer, we need to have all parent values available.
    # We load inputs as needed, compute layer by layer, and store results.
    # Key insight: in a layered DAG, we can process one layer at a time.
    
    ops = []
    # Track which nodes are currently in fast memory (red)
    red = set()
    # Track which nodes have been stored (blue, available for reuse)
    stored = set()
    
    for layer_idx in layer_order:
        layer_nodes = layers[layer_idx]
        
        # For each node in this layer, ensure its parents are available
        for node in layer_nodes:
            par = parents.get(node, [])
            for p in par:
                # If parent is not in red and not stored, we need to load it
                # But for inputs, they start blue, so we load them
                if p not in red and p not in stored:
                    # Load the parent
                    ops.append(['load', p])
                    red.add(p)
                    if len(red) > fast_memory_budget:
                        # Need to free space: delete a non-essential node
                        # Delete nodes that are not needed for current or future layers
                        # Simple heuristic: delete oldest loaded non-output node
                        for r in list(red):
                            if r != p and r not in outputs:
                                red.discard(r)
                                break
        
        # Compute the node
        ops.append(['compute', node])
        red.add(node)
        
        # Store the node if it's an output or if we need to free space
        if node in outputs:
            ops.append(['store', node])
            red.discard(node)
            stored.add(node)
        elif len(red) > fast_memory_budget:
            # Free space by storing or deleting
            # Store if it might be needed later (has children)
            if children.get(node):
                ops.append(['store', node])
                red.discard(node)
                stored.add(node)
            else:
                red.discard(node)
                ops.append(['delete', node])
    
    # Ensure all outputs are stored
    for out in outputs:
        if out not in stored:
            if out in red:
                ops.append(['store', out])
                red.discard(out)
                stored.add(out)
    
    return ops

def _naive_schedule(dag, fast_memory_budget):
    """
    Naive schedule: load all inputs, compute in topological order, store all outputs.
    """
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        return []
    
    ops = []
    # Load all inputs
    for inp in inputs:
        ops.append(['load', inp])
    
    # Compute in topological order
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    children = defaultdict(list)
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
    
    queue = [n for n in nodes if in_degree[n] == 0]
    topo_order = []
    while queue:
        n = queue.pop(0)
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    for n in topo_order:
        if n not in inputs:
            ops.append(['compute', n])
    
    # Store all outputs
    for out in outputs:
        ops.append(['store', out])
    
    return ops

def _estimate_cost(ops):
    """Estimate transfer cost of an operation sequence."""
    cost = 0
    for op, v in ops:
        if op in ('load', 'store'):
            cost += 1
    return cost

def pebble(dag, fast_memory_budget):
    """
    Find a minimum-transfer pebbling schedule.
    Uses layered divide-and-conquer for structured DAGs,
    with fallback to simple strategies.
    """
    deadline = time.time() + 80.0  # Safety margin
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if not nodes:
        return []
    
    # Strategy 1: Layered pebbling (exploits structure)
    layered_ops = _layered_pebble(dag, fast_memory_budget)
    if layered_ops is not None and len(layered_ops) <= 20000:
        layered_cost = _estimate_cost(layered_ops)
    
    # Strategy 2: Naive schedule
    naive_ops = _naive_schedule(dag, fast_memory_budget)
    naive_cost = _estimate_cost(naive_ops)
    
    # Choose the better strategy
    if layered_ops is not None and layered_cost <= naive_cost:
        return layered_ops
    else:
        return naive_ops

# EVOLVE-BLOCK-END
