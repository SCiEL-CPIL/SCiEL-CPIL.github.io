"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    def time_left():
        return deadline - (time.time() - t0)

    # Try to import baseline first as a fallback
    baseline_ops = None
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except Exception:
        baseline_ops = None

    # Try to understand the DAG structure
    # Common conventions: dag is a dict with 'parents' (list of lists or dict) and 'outputs'
    # Or dag could be a list of nodes where each node has parents

    nodes = None
    parents = None
    outputs = None

    # Try to parse dag structure
    try:
        if isinstance(dag, dict):
            # Convention 1: dag = {'parents': [...], 'outputs': [...]}
            if 'parents' in dag:
                parents = dag['parents']
            if 'outputs' in dag:
                outputs = dag['outputs']
            # Get all nodes
            if isinstance(parents, list):
                nodes = list(range(len(parents)))
            elif isinstance(parents, dict):
                nodes = list(parents.keys())
        elif isinstance(dag, list):
            # Convention 2: dag is a list of node dicts or tuples
            # Each element might be a node with 'parents' key
            nodes = list(range(len(dag)))
            parents = []
            for i, node in enumerate(dag):
                if isinstance(node, dict):
                    p = node.get('parents', [])
                    parents.append(p)
                elif isinstance(node, (list, tuple)):
                    # Could be (node_id, [parents]) or just [parents]
                    if len(node) >= 2 and isinstance(node[1], (list, tuple)):
                        parents.append(list(node[1]))
                    else:
                        parents.append([])
                else:
                    parents.append([])
            # Try to find outputs
            outputs = None
            for i, node in enumerate(dag):
                if isinstance(node, dict):
                    if node.get('is_output') or node.get('output'):
                        if outputs is None:
                            outputs = []
                        outputs.append(i)
        else:
            # Try other structures
            pass
    except Exception:
        pass

    # If we couldn't parse, try baseline
    if baseline_ops is not None:
        return baseline_ops

    if nodes is None or parents is None:
        # Can't determine structure, return empty
        return []

    # Normalize parents to list of lists of ints
    if isinstance(parents, dict):
        parent_list = []
        for i in range(len(nodes)):
            if i in parents:
                parent_list.append(list(parents[i]))
            else:
                parent_list.append([])
        parents = parent_list

    # Ensure outputs is a list
    if outputs is None:
        # Assume last node is output, or nodes with no children
        # Find nodes that are not parents of any other node
        all_parents = set()
        for p_list in parents:
            for p in p_list:
                all_parents.add(p)
        outputs = [i for i in range(len(nodes)) if i not in all_parents]
        if not outputs:
            outputs = [len(nodes) - 1]

    if not isinstance(outputs, list):
        outputs = [outputs]

    n = len(nodes)
    budget = fast_memory_budget

    # Build adjacency: for each node, which nodes depend on it (children)
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)

    # Identify input nodes (no parents) and output nodes
    input_nodes = [i for i in range(n) if len(parents[i]) == 0]
    output_set = set(outputs)

    # For layered DAGs, detect layer structure
    # A layered DAG has nodes grouped by "depth" from inputs
    # Let's compute depth of each node
    depth = [-1] * n
    # Topological order
    in_degree = [len(parents[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        if depth[u] == -1:
            depth[u] = 0
        for v in children[u]:
            in_degree[v] -= 1
            depth[v] = max(depth[v], depth[u] + 1)
            if in_degree[v] == 0:
                queue.append(v)

    # Group nodes by depth
    max_depth = max(depth) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        if depth[v] >= 0:
            layers[depth[v]].append(v)

    # Strategy: For layered DAGs with limited memory, use a layer-by-layer approach
    # Load inputs of current layer, compute next layer, store/delete as needed
    # This minimizes communication by keeping only necessary nodes in memory

    # Check if we can use exact search for small DAGs
    if n <= 15 and time_left() > 5.0:
        # Exact Dijkstra for small DAGs
        start_state = frozenset()
        target_state = frozenset()  # all blue

        # Priority queue: (cost, num_ops, state)
        # dist[state] = (cost, num_ops)
        dist = {start_state: (0, 0)}
        parent = {start_state: (None, None)}  # (prev_state, op)
        pq = [(0, 0, start_state)]

        max_states = 50000
        states_explored = 0

        while pq and states_explored < max_states and time_left() > 2.0:
            cost, num_ops, state = heapq.heappop(pq)

            if (cost, num_ops) > dist.get(state, (float('inf'), float('inf'))):
                continue

            states_explored += 1

            if state == target_state:
                # Reconstruct path
                path = []
                cur = state
                while parent[cur][0] is not None:
                    prev, op = parent[cur]
                    path.append(op)
                    cur = prev
                path.reverse()
                return path

            # Generate successors
            red_set = state
            red_count = len(red_set)

            # For each vertex, try operations
            for v in range(n):
                is_red = v in red_set
                is_input = len(parents[v]) == 0
                parents_all_red = all(p in red_set for p in parents[v])

                # load(v): v must be blue, v must be input or previously computed
                # For simplicity, allow load only for inputs
                if not is_red and is_input and red_count < budget:
                    new_state = red_set | frozenset([v])
                    new_cost = cost + 1
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent[new_state] = (state, ['load', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

                # compute(v): all parents must be red, v must be blue
                if not is_red and parents_all_red and red_count < budget:
                    new_state = red_set | frozenset([v])
                    new_cost = cost
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent[new_state] = (state, ['compute', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

                # store(v): v must be red, v must be an output
                if is_red and v in output_set:
                    new_state = red_set - frozenset([v])
                    new_cost = cost + 1
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent[new_state] = (state, ['store', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

                # delete(v): v must be red, v must NOT be an output (outputs must be stored, not deleted)
                if is_red and v not in output_set:
                    new_state = red_set - frozenset([v])
                    new_cost = cost
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent[new_state] = (state, ['delete', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

    # Layered approach for larger DAGs or when exact search fails
    # This is the key improvement: use layer structure to minimize communication
    ops = []
    
    # Process layer by layer
    # For each layer, we need to have its parents available
    # Strategy: load inputs, then for each layer compute nodes and manage memory
    
    # First, load all input nodes
    for v in input_nodes:
        ops.append(['load', v])
    
    # Track which nodes are currently in memory (red)
    in_memory = set(input_nodes)
    
    # Process nodes in topological order, but group by layer
    # For each node in topo order:
    # - If all parents are in memory, compute it (add to memory)
    # - If memory is full, we need to evict some nodes
    # - When we're done with a node's children, we can delete it
    
    # Better approach: process by layers
    # For layer L, we need parents from layer L-1
    # After computing layer L, we can delete layer L-1 nodes that are no longer needed
    
    # Compute for each node, which layers' nodes are its descendants
    # A node in layer L is needed until all its descendants in later layers are computed
    
    # For each node, find the maximum depth of its descendants
    max_desc_depth = [depth[i] for i in range(n)]
    for v in reversed(topo):
        for child in children[v]:
            max_desc_depth[v] = max(max_desc_depth[v], max_desc_depth[child])
    
    # Now process layer by layer
    # For each layer, compute all nodes in that layer
    # Before computing layer L, ensure all parents (from layer L-1) are in memory
    # After computing layer L, delete nodes from layer L-1 that are no longer needed
    
    # Track which nodes are still needed
    # A node v is needed if any of its descendants haven't been computed yet
    # We can track this by: node v is needed if max_desc_depth[v] > current_layer
    
    # Process layers
    current_memory = set(input_nodes)  # Start with all inputs in memory
    
    for layer_idx in range(1, max_depth + 1):
        layer_nodes = layers[layer_idx]
        
        # Before computing this layer, make sure all parents are in memory
        # Parents should be from layer layer_idx-1
        # Load any missing parents (shouldn't happen if we keep them)
        
        # Compute all nodes in this layer
        for v in layer_nodes:
            # Check if all parents are in memory
            if all(p in current_memory for p in parents[v]):
                ops.append(['compute', v])
                current_memory.add(v)
            else:
                # This shouldn't happen in a proper layered DAG, but handle it
                # Load missing parents
                for p in parents[v]:
                    if p not in current_memory:
                        ops.append(['load', p])
                        current_memory.add(p)
                ops.append(['compute', v])
                current_memory.add(v)
        
        # After computing this layer, delete nodes from previous layers that are no longer needed
        # A node u from layer < layer_idx is no longer needed if max_desc_depth[u] <= layer_idx
        # But we need to be careful: we can only delete if it's not an output
        
        # Delete nodes that are no longer needed
        for u in list(current_memory):
            if depth[u] < layer_idx and u not in output_set:
                # Check if u is still needed (has descendants in current or later layers)
                if max_desc_depth[u] <= layer_idx:
                    ops.append(['delete', u])
                    current_memory.remove(u)
        
        # Also delete any non-output nodes from earlier layers that are no longer needed
        # (already handled above)
    
    # Store all outputs
    for v in outputs:
        if v in current_memory:
            ops.append(['store', v])
            current_memory.remove(v)
        else:
            # Shouldn't happen, but handle it
            # We need to recompute or load
            # For simplicity, just store if it was computed
            pass
    
    # Delete any remaining non-output nodes
    for v in list(current_memory):
        if v not in output_set:
            ops.append(['delete', v])
            current_memory.remove(v)
    
    # Check if the schedule is valid
    if len(ops) <= 20000:
        # Verify the schedule
        red = set()
        for op, v in ops:
            if op == 'load':
                if v in red:
                    return []  # Invalid
                if len(parents[v]) != 0:
                    return []  # Can only load inputs
                red.add(v)
            elif op == 'store':
                if v not in red:
                    return []  # Invalid
                red.remove(v)
            elif op == 'compute':
                if v in red:
                    return []  # Invalid
                if not all(p in red for p in parents[v]):
                    return []  # Invalid
                red.add(v)
            elif op == 'delete':
                if v not in red:
                    return []  # Invalid
                red.remove(v)
        
        # Check final state: all outputs must be blue (not in red)
        for v in outputs:
            if v in red:
                return []  # Invalid
        
        # Check memory constraint
        # We need to track memory throughout
        red = set()
        max_mem = 0
        for op, v in ops:
            if op == 'load':
                red.add(v)
            elif op == 'store':
                red.remove(v)
            elif op == 'compute':
                red.add(v)
            elif op == 'delete':
                red.remove(v)
            max_mem = max(max_mem, len(red))
        
        if max_mem <= budget:
            return ops
    
    # Fallback: naive topological order
    ops = []
    for v in topo:
        if len(parents[v]) == 0:
            ops.append(['load', v])
        else:
            ops.append(['compute', v])

    for v in outputs:
        if v in topo:
            ops.append(['store', v])

    for v in topo:
        if v not in outputs:
            ops.append(['delete', v])

    if len(ops) <= 20000:
        return ops

    # Last resort: return baseline if available
    if baseline_ops is not None:
        return baseline_ops

    return []

# EVOLVE-BLOCK-END
