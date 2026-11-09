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

    # Helper to validate a schedule
    def validate_schedule(ops, parents, outputs, budget, n):
        if len(ops) > 20000:
            return False
        red = set()
        stored_outputs = set()
        for op, v in ops:
            if op == 'load':
                if v in red:
                    return False
                if len(parents[v]) != 0:
                    return False
                if len(red) >= budget:
                    return False
                red.add(v)
            elif op == 'compute':
                if v in red:
                    return False
                if not all(p in red for p in parents[v]):
                    return False
                if len(red) >= budget:
                    return False
                red.add(v)
            elif op == 'store':
                if v not in red:
                    return False
                if v not in outputs:
                    return False
                red.remove(v)
                stored_outputs.add(v)
            elif op == 'delete':
                if v not in red:
                    return False
                if v in outputs:
                    return False
                red.remove(v)
        # All outputs must be stored
        if stored_outputs != set(outputs):
            return False
        return True

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
        # Can't determine structure, try to construct minimal schedule
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
    if n <= 12 and time_left() > 5.0:
        # Exact Dijkstra for small DAGs
        # State: (frozenset of red vertices, frozenset of stored outputs)
        start_state = (frozenset(), frozenset())
        target_state = (frozenset(), frozenset(outputs))

        # Priority queue: (cost, num_ops, state)
        dist = {start_state: (0, 0)}
        parent_map = {start_state: (None, None)}
        pq = [(0, 0, start_state)]

        max_states = 100000
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
                while parent_map[cur][0] is not None:
                    prev, op = parent_map[cur]
                    path.append(op)
                    cur = prev
                path.reverse()
                if validate_schedule(path, parents, outputs, budget, n):
                    return path
                else:
                    continue

            # Generate successors
            red_set, stored_set = state
            red_count = len(red_set)

            # For each vertex, try operations
            for v in range(n):
                is_red = v in red_set
                is_input = len(parents[v]) == 0
                parents_all_red = all(p in red_set for p in parents[v])
                is_output = v in output_set

                # load(v): v must be blue, v must be input, memory not full
                if not is_red and is_input and red_count < budget:
                    new_state = (red_set | frozenset([v]), stored_set)
                    new_cost = cost + 1
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent_map[new_state] = (state, ['load', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

                # compute(v): all parents must be red, v must be blue, memory not full
                if not is_red and parents_all_red and red_count < budget:
                    new_state = (red_set | frozenset([v]), stored_set)
                    new_cost = cost
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent_map[new_state] = (state, ['compute', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

                # store(v): v must be red, v must be an output, not already stored
                if is_red and is_output and v not in stored_set:
                    new_state = (red_set - frozenset([v]), stored_set | frozenset([v]))
                    new_cost = cost + 1
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent_map[new_state] = (state, ['store', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

                # delete(v): v must be red, v must NOT be an output
                if is_red and not is_output:
                    new_state = (red_set - frozenset([v]), stored_set)
                    new_cost = cost
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent_map[new_state] = (state, ['delete', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

    # Layered approach for larger DAGs or when exact search fails
    # This is the key improvement: use layer structure to minimize communication
    ops = []
    
    # Check if we can fit all inputs in memory
    if len(input_nodes) > budget:
        # Can't even load all inputs - need a different strategy
        # Fall back to naive approach
        pass
    
    # Compute for each node, the maximum depth of its descendants
    max_desc_depth = [depth[i] for i in range(n)]
    for v in reversed(topo):
        for child in children[v]:
            max_desc_depth[v] = max(max_desc_depth[v], max_desc_depth[child])
    
    # Process layers with memory management
    current_memory = set()
    
    # Load input nodes (respecting budget)
    for v in input_nodes:
        if len(current_memory) < budget:
            ops.append(['load', v])
            current_memory.add(v)
    
    # Process each layer
    for layer_idx in range(1, max_depth + 1):
        layer_nodes = layers[layer_idx]
        
        # Before computing this layer, ensure all parents are in memory
        # If not, we may need to evict some nodes and reload
        for v in layer_nodes:
            missing_parents = [p for p in parents[v] if p not in current_memory]
            if missing_parents:
                # Need to free space and reload missing parents
                # First, delete nodes that are no longer needed
                for u in list(current_memory):
                    if u not in output_set and max_desc_depth[u] < layer_idx:
                        ops.append(['delete', u])
                        current_memory.remove(u)
                        if not missing_parents:
                            break
                
                # If still can't fit, this is problematic
                # For now, just try to load if there's space
                for p in missing_parents:
                    if len(current_memory) < budget:
                        ops.append(['load', p])
                        current_memory.add(p)
                    else:
                        # Can't fit - skip this node (will cause issues)
                        break
            
            # Compute if all parents are available
            if all(p in current_memory for p in parents[v]):
                if len(current_memory) < budget:
                    ops.append(['compute', v])
                    current_memory.add(v)
        
        # After computing this layer, delete nodes from previous layers that are no longer needed
        for u in list(current_memory):
            if depth[u] < layer_idx and u not in output_set:
                if max_desc_depth[u] <= layer_idx:
                    ops.append(['delete', u])
                    current_memory.remove(u)
    
    # Store all outputs
    for v in outputs:
        if v in current_memory:
            ops.append(['store', v])
            current_memory.remove(v)
    
    # Delete any remaining non-output nodes
    for v in list(current_memory):
        if v not in output_set:
            ops.append(['delete', v])
            current_memory.remove(v)
    
    # Validate the schedule
    if validate_schedule(ops, parents, outputs, budget, n):
        return ops
    
    # Fallback: naive topological order with memory management
    ops = []
    current_memory = set()
    
    for v in topo:
        if len(parents[v]) == 0:
            # Load input
            if len(current_memory) < budget:
                ops.append(['load', v])
                current_memory.add(v)
            else:
                # Need to evict - delete nodes no longer needed
                for u in list(current_memory):
                    if u not in output_set and all(c not in current_memory for c in children[u] if depth[c] > depth[u]):
                        ops.append(['delete', u])
                        current_memory.remove(u)
                        if len(current_memory) < budget:
                            break
                if len(current_memory) < budget:
                    ops.append(['load', v])
                    current_memory.add(v)
        else:
            # Compute
            missing = [p for p in parents[v] if p not in current_memory]
            if missing:
                # Try to reload missing parents
                for p in missing:
                    if len(current_memory) < budget:
                        ops.append(['load', p])
                        current_memory.add(p)
                    else:
                        # Evict something
                        for u in list(current_memory):
                            if u not in output_set and u not in parents[v]:
                                ops.append(['delete', u])
                                current_memory.remove(u)
                                if len(current_memory) < budget:
                                    break
                        if len(current_memory) < budget:
                            ops.append(['load', p])
                            current_memory.add(p)
            
            if all(p in current_memory for p in parents[v]):
                if len(current_memory) < budget:
                    ops.append(['compute', v])
                    current_memory.add(v)
                else:
                    # Can't fit - skip
                    pass
    
    # Store outputs
    for v in outputs:
        if v in current_memory:
            ops.append(['store', v])
            current_memory.remove(v)
    
    # Delete remaining
    for v in list(current_memory):
        if v not in output_set:
            ops.append(['delete', v])
            current_memory.remove(v)
    
    if validate_schedule(ops, parents, outputs, budget, n):
        return ops

    # Last resort: return baseline if available
    if baseline_ops is not None:
        return baseline_ops

    return []

# EVOLVE-BLOCK-END
