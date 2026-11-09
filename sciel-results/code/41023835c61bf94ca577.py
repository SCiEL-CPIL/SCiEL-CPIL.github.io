"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
            # Outputs are nodes with no children
            children = {n: set() for n in nodes}
            for n in nodes:
                for p in parents.get(n, []):
                    children[p].add(n)
            outputs = [n for n in nodes if len(children[n]) == 0]
        
        return nodes, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
        # List of (node, parents) or list of dicts
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
        
        # Determine inputs/outputs
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

def _naive_schedule_cost(dag, fast_memory_budget):
    """
    Compute the cost of a naive sequential schedule for reference.
    This loads all inputs, computes each node once, stores all outputs.
    """
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        return 1.0
    
    # Naive: load each input once, compute each node once, store each output once
    # Cost = len(inputs) + len(outputs)  (computes are free)
    # But we might need to reload if memory is tight - for simplicity, assume we can hold all
    return len(inputs) + len(outputs)

def _get_relevant_nodes(nodes, parents, outputs):
    """
    Get only the nodes that are ancestors of outputs (including the outputs themselves).
    This reduces the state space for search.
    """
    output_set = set(outputs)
    relevant = set(output_set)
    
    # BFS/DFS from outputs to find all ancestors
    queue = list(output_set)
    while queue:
        node = queue.pop()
        for p in parents.get(node, []):
            if p not in relevant:
                relevant.add(p)
                queue.append(p)
    
    # Return in topological order (inputs first)
    relevant_list = [n for n in nodes if n in relevant]
    return relevant_list, relevant

def pebble(dag, fast_memory_budget):
    """
    Find a minimum-transfer pebbling schedule using A* search
    over the state space of red-pebble configurations.
    
    State: frozenset of red nodes
    Goal: all outputs are stored (blue), and we can have any red nodes left
    """
    deadline = time.time() + 85.0  # 5 second safety margin before 90s limit
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        # Fallback to baseline
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Get relevant nodes (ancestors of outputs)
    relevant_nodes, relevant_set = _get_relevant_nodes(nodes, parents, outputs)
    
    # Filter inputs and outputs to relevant ones
    relevant_inputs = [n for n in inputs if n in relevant_set]
    relevant_outputs = [n for n in outputs if n in relevant_set]
    
    # Precompute which nodes can be computed given a set of red nodes
    # For efficiency, precompute parent sets (only relevant parents)
    parent_sets = {}
    for n in relevant_nodes:
        par = parents.get(n, [])
        # Only consider parents that are in the relevant set
        parent_sets[n] = frozenset(p for p in par if p in relevant_set)
    
    # State: frozenset of red nodes
    # We don't need to track completed outputs separately - 
    # an output is "done" when it's been stored (not in red)
    # But we need to ensure it was computed at some point
    # Actually, we can only store a node if it's been computed
    # So we need to track which nodes have been computed
    
    # Revised state: (frozenset(red_nodes), frozenset(computed_nodes))
    # A node is computed if it's in red or was stored (computed and then stored)
    # Actually, let's simplify: track which nodes have been computed at least once
    
    initial_state = (frozenset(), frozenset())
    
    def is_goal(state):
        red, computed = state
        # All outputs must be computed and stored (not in red)
        return output_set.issubset(computed) and not (output_set & red)
    
    # A* search
    # dist[state] = minimum transfers to reach state
    # parent[state] = (prev_state, operation, vertex)
    
    dist = {initial_state: 0}
    parent = {initial_state: None}
    
    # Heuristic: lower bound on remaining transfers
    # At minimum, we need to load each input that hasn't been loaded yet
    # and store each output that hasn't been stored yet
    # But this is complex to compute exactly, so use a simple admissible heuristic
    def heuristic(state):
        red, computed = state
        # Outputs not yet computed need at least 1 store each
        # But they also need their inputs loaded
        # Simple admissible: number of outputs not yet computed
        uncomputed_outputs = output_set - computed
        return len(uncomputed_outputs)
    
    # Use A* with consistent heuristic
    pq = [(heuristic(initial_state), 0, initial_state)]
    
    best_score = float('inf')
    best_state = None
    best_path = None
    
    # Track visited states to avoid re-expansion
    expanded = set()
    
    while pq:
        if time.time() > deadline:
            break
        
        f, d, state = heapq.heappop(pq)
        
        # Skip if already expanded
        if state in expanded:
            continue
        
        expanded.add(state)
        
        if d > dist.get(state, float('inf')):
            continue
        
        if is_goal(state):
            best_score = d
            best_state = state
            break
        
        red = state[0]
        computed = state[1]
        red_set = set(red)
        
        # Generate all valid next states
        # 1. Load an input node (if not already red)
        for inp in relevant_inputs:
            if inp not in red_set:
                if len(red_set) < fast_memory_budget:
                    new_red = red_set | {inp}
                    new_computed = computed  # Loading doesn't compute
                    new_state = (frozenset(new_red), computed)
                    cost = 1
                    new_d = d + cost
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent[new_state] = (state, 'load', inp)
                        heapq.heappush(pq, (new_d + heuristic(new_state), new_d, new_state))
        
        # 2. Compute a node (if all parents are red, and node not yet computed)
        for node in relevant_nodes:
            if node in computed:
                continue
            if node in red_set:
                continue
            # Check if all parents are red
            par = parent_sets.get(node, frozenset())
            if par.issubset(red_set):
                # Can compute
                if len(red_set) < fast_memory_budget:
                    new_red = red_set | {node}
                    new_computed = computed | {node}
                    new_state = (frozenset(new_red), new_computed)
                    cost = 0
                    new_d = d + cost
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent[new_state] = (state, 'compute', node)
                        heapq.heappush(pq, (new_d + heuristic(new_state), new_d, new_state))
        
        # 3. Store a red node (frees space, marks as computed if it was just computed)
        for node in red_set:
            # We can store any red node
            new_red = red_set - {node}
            # Storing means it's been computed (it was red, so it was computed)
            new_computed = computed | {node}
            new_state = (frozenset(new_red), new_computed)
            cost = 1
            new_d = d + cost
            if new_d < dist.get(new_state, float('inf')):
                dist[new_state] = new_d
                parent[new_state] = (state, 'store', node)
                heapq.heappush(pq, (new_d + heuristic(new_state), new_d, new_state))
        
        # 4. Delete a red node (free, but loses the value)
        for node in red_set:
            new_red = red_set - {node}
            # Deleting doesn't mark as computed - the value is lost
            # But if it was an output, we still need to recompute it
            new_state = (frozenset(new_red), computed)
            cost = 0
            new_d = d + cost
            if new_d < dist.get(new_state, float('inf')):
                dist[new_state] = new_d
                parent[new_state] = (state, 'delete', node)
                heapq.heappush(pq, (new_d + heuristic(new_state), new_d, new_state))
    
    # If we found a solution, reconstruct the path
    if best_state is not None:
        ops = []
        state = best_state
        while parent[state] is not None:
            prev_state, op, vertex = parent[state]
            ops.append([op, vertex])
            state = prev_state
        ops.reverse()
        
        # Check if we're within operation limit
        if len(ops) <= 20000:
            return ops
    
    # Fallback: greedy layer-by-layer schedule
    # Load all inputs, compute layer by layer, store outputs
    ops = []
    for inp in relevant_inputs:
        ops.append(['load', inp])
    
    for node in relevant_nodes:
        if node in input_set:
            continue
        ops.append(['compute', node])
        # If memory is tight, store intermediate results
        if len([op for op in ops if op[0] == 'load' or op[0] == 'compute']) > fast_memory_budget * 2:
            # Store some intermediate results to free memory
            for n in relevant_nodes:
                if n not in input_set and n not in output_set:
                    ops.append(['store', n])
                    break
    
    for out in relevant_outputs:
        ops.append(['store', out])
    
    if len(ops) <= 20000:
        return ops
    
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
