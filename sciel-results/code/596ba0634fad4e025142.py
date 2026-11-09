"""Optimized DAG pebbling with layered greedy and bounded Dijkstra."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

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

def _naive_schedule(nodes, parents, inputs, outputs, fast_memory_budget):
    """
    Naive schedule: process nodes in topological order, load/store as needed.
    Returns (operations, transfer_count).
    """
    ops = []
    transfers = 0
    red = set()
    computed = set()
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Topological sort
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
    
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Process in topological order
    for n in topo_order:
        par = parents.get(n, [])
        # Load parents if not red
        for p in par:
            if p not in red:
                if p in input_set:
                    ops.append(['load', p])
                    transfers += 1
                    red.add(p)
                else:
                    # p should have been computed and stored; reload
                    ops.append(['load', p])
                    transfers += 1
                    red.add(p)
        
        # Compute n
        ops.append(['compute', n])
        red.add(n)
        computed.add(n)
        
        # If n is an output, store it
        if n in output_set:
            ops.append(['store', n])
            transfers += 1
            red.discard(n)
        
        # If fast memory is full, delete a non-output, non-parent-needed node
        # Simple heuristic: delete nodes that won't be needed again
        if len(red) > fast_memory_budget:
            # Find nodes in red that are not needed by any future node
            future_needed = set()
            for m in topo_order[topo_order.index(n)+1:]:
                for p in parents.get(m, []):
                    future_needed.add(p)
            
            deletable = [r for r in red if r not in future_needed and r not in output_set]
            while len(red) > fast_memory_budget and deletable:
                d = deletable.pop()
                ops.append(['delete', d])
                red.discard(d)
    
    # Store any remaining red outputs
    for n in red:
        if n in output_set and n not in computed:
            ops.append(['store', n])
            transfers += 1
            red.discard(n)
    
    return ops, transfers

def _greedy_layered_schedule(nodes, parents, inputs, outputs, fast_memory_budget):
    """
    Greedy layered schedule: process layer by layer, minimizing transfers.
    """
    ops = []
    transfers = 0
    red = set()
    computed = set()
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Compute layers (distance from inputs)
    layer = {}
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
    
    queue = deque([n for n in nodes if in_degree[n] == 0])
    for n in queue:
        layer[n] = 0
    while queue:
        n = queue.popleft()
        for c in children[n]:
            if c not in layer or layer[c] < layer[n] + 1:
                layer[c] = layer[n] + 1
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Group by layer
    max_layer = max(layer.values()) if layer else 0
    layers = defaultdict(list)
    for n in nodes:
        layers[layer[n]].append(n)
    
    # Process layer by layer
    for L in range(max_layer + 1):
        layer_nodes = layers[L]
        
        # Determine which nodes in this layer are needed by future layers
        future_needed = set()
        for L2 in range(L + 1, max_layer + 1):
            for n in layers[L2]:
                for p in parents.get(n, []):
                    future_needed.add(p)
        
        # Load inputs for this layer if not already red
        for n in layer_nodes:
            par = parents.get(n, [])
            for p in par:
                if p not in red:
                    ops.append(['load', p])
                    transfers += 1
                    red.add(p)
            
            # Compute n
            ops.append(['compute', n])
            red.add(n)
            computed.add(n)
            
            # If n is an output, store it
            if n in output_set:
                ops.append(['store', n])
                transfers += 1
                red.discard(n)
        
        # Clean up red nodes that won't be needed in future layers
        if len(red) > fast_memory_budget:
            deletable = [r for r in red if r not in future_needed and r not in output_set]
            deletable.sort()  # Deterministic
            while len(red) > fast_memory_budget and deletable:
                d = deletable.pop()
                ops.append(['delete', d])
                red.discard(d)
        
        # Even if not over budget, delete nodes not needed in future
        # to keep memory clean
        for r in list(red):
            if r not in future_needed and r not in output_set:
                ops.append(['delete', r])
                red.discard(r)
    
    return ops, transfers

def _bounded_dijkstra(nodes, parents, inputs, outputs, fast_memory_budget, time_limit=10.0):
    """
    Bounded Dijkstra over red-pebble states.
    Returns (operations, transfers) or (None, None) if no solution found.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # State: frozenset of red nodes
    # We don't track completed separately; a node is "done" if it's been stored
    # but for simplicity, we just need all outputs to be stored at the end.
    
    initial_state = frozenset()
    
    def is_goal(state):
        # All outputs must be blue (not in red) and must have been computed
        # Since we can only store outputs, if an output is not in red, it's blue
        # But we need to ensure it was computed. Track computed separately?
        # For now, assume if output is not in red, it's done.
        # Actually, we need to track which outputs have been stored.
        # Let's use state = (frozenset(red), frozenset(stored_outputs))
        pass
    
    # Simpler: state = frozenset(red)
    # Goal: all outputs are not in red (they've been stored)
    # But we need to ensure they were computed. Since we can only store if computed,
    # and compute requires parents red, the logic handles it.
    
    # However, we might store an output, then later need it again? No, outputs are sinks.
    # So once stored, they're done.
    
    # But the state doesn't track which outputs have been stored.
    # We can infer: if an output is not in red, it's either not computed yet or stored.
    # We need to distinguish.
    
    # Let's use state = (frozenset(red), frozenset(done_outputs))
    
    def make_state(red, done):
        return (frozenset(red), frozenset(done))
    
    initial_state = make_state(set(), set())
    
    def is_goal(state):
        return state[1] == output_set
    
    dist = {initial_state: 0}
    parent_map = {initial_state: None}
    pq = [(0, initial_state)]
    
    best_score = float('inf')
    best_state = None
    
    while pq:
        if time.time() > deadline:
            break
        
        d, state = heapq.heappop(pq)
        
        if d > dist.get(state, float('inf')):
            continue
        
        if is_goal(state):
            best_score = d
            best_state = state
            break
        
        red = set(state[0])
        done = set(state[1])
        
        # Generate next states
        
        # 1. Load an input
        for inp in inputs:
            if inp not in red and inp not in done:
                if len(red) < fast_memory_budget:
                    new_red = red | {inp}
                    new_state = make_state(new_red, done)
                    new_d = d + 1
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent_map[new_state] = (state, 'load', inp)
                        heapq.heappush(pq, (new_d, new_state))
        
        # 2. Compute a node
        for node in nodes:
            if node in done or node in red:
                continue
            par = parents.get(node, [])
            if all(p in red for p in par):
                if len(red) < fast_memory_budget:
                    new_red = red | {node}
                    new_done = done | {node} if node in output_set else done
                    new_state = make_state(new_red, new_done)
                    new_d = d  # compute costs 0
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent_map[new_state] = (state, 'compute', node)
                        heapq.heappush(pq, (new_d, new_state))
        
        # 3. Store a red output
        for node in red:
            if node in output_set and node not in done:
                new_red = red - {node}
                new_done = done | {node}
                new_state = make_state(new_red, new_done)
                new_d = d + 1
                if new_d < dist.get(new_state, float('inf')):
                    dist[new_state] = new_d
                    parent_map[new_state] = (state, 'store', node)
                    heapq.heappush(pq, (new_d, new_state))
        
        # 4. Delete a red node
        for node in red:
            if node not in output_set or node in done:
                # Can delete if not an output that needs to be stored
                # Actually, if it's an output not done, we should store it, not delete
                if node in output_set and node not in done:
                    continue
                new_red = red - {node}
                new_state = make_state(new_red, done)
                new_d = d  # delete costs 0
                if new_d < dist.get(new_state, float('inf')):
                    dist[new_state] = new_d
                    parent_map[new_state] = (state, 'delete', node)
                    heapq.heappush(pq, (new_d, new_state))
    
    if best_state is not None:
        ops = []
        state = best_state
        while parent_map[state] is not None:
            prev_state, op, vertex = parent_map[state]
            ops.append([op, vertex])
            state = prev_state
        ops.reverse()
        if len(ops) <= 20000:
            return ops, best_score
    
    return None, None

def pebble(dag, fast_memory_budget):
    """
    Find a minimum-transfer pebbling schedule.
    """
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        return []
    
    if not nodes:
        return []
    
    # Try greedy layered schedule first
    greedy_ops, greedy_transfers = _greedy_layered_schedule(
        nodes, parents, inputs, outputs, fast_memory_budget)
    
    # Try bounded Dijkstra with limited time
    dijkstra_ops, dijkstra_transfers = _bounded_dijkstra(
        nodes, parents, inputs, outputs, fast_memory_budget, time_limit=5.0)
    
    # Choose the better one
    if dijkstra_ops is not None and dijkstra_transfers <= greedy_transfers:
        return dijkstra_ops
    
    return greedy_ops
# EVOLVE-BLOCK-END
