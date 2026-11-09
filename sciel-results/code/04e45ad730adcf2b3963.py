"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _extract_dag_info(dag):
    """
    Extract DAG structure. Handles common formats.
    """
    if isinstance(dag, dict):
        nodes = dag.get('nodes', [])
        if not nodes and 'vertices' in dag:
            nodes = dag['vertices']
        edges = dag.get('edges', [])
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        parents = dag.get('parents', {})
        
        if not parents:
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
    Generate a naive topological schedule to serve as a baseline for comparison.
    This mimics the behavior of the seed program's greedy approach but strictly
    follows topological order without lookahead eviction, ensuring a valid baseline
    for the score calculation if the evaluator requires it, or simply as a fallback.
    """
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
                
    if len(topo_order) != len(nodes):
        return None

    red = set()
    completed = set()
    schedule = []
    
    output_set = set(outputs)
    
    for node in topo_order:
        # Ensure parents are red
        for p in parents.get(node, []):
            if p not in red:
                if len(red) >= fast_memory_budget:
                    # Evict any red node that is not an output we haven't stored yet?
                    # Naive strategy: evict the first one we find that isn't needed for current node?
                    # Actually, naive usually just assumes we can evict anything.
                    # To be safe and simple, evict the first non-output node or any node.
                    # In a true naive schedule for baseline, we might just assume infinite memory or
                    # a very simple LRU. Let's just evict the first node in red.
                    # This is just a fallback, so correctness is key, optimality is not.
                    victim = next(iter(red))
                    red.remove(victim)
                    schedule.append(['delete', victim])
                
                red.add(p)
                schedule.append(['load', p])
        
        # Compute node
        if node not in red:
            if len(red) >= fast_memory_budget:
                victim = next(iter(red))
                red.remove(victim)
                schedule.append(['delete', victim])
            
            red.add(node)
            schedule.append(['compute', node])
        
        # If node is output, store it
        if node in output_set:
            if node in red:
                red.remove(node)
            completed.add(node)
            schedule.append(['store', node])
            
    return schedule

def pebble(dag, fast_memory_budget):
    """
    Hybrid pebbling: 
    1. Attempts an exact Dijkstra search on the state space of red pebbles for small/medium DAGs.
    2. Falls back to a greedy heuristic with lookahead eviction for larger DAGs or if time runs out.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    n_nodes = len(nodes)
    
    # Precompute children map
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
            
    # Topological sort
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(nodes):
        # Cycle or error, fallback
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

    # Strategy Selection:
    # If the DAG is small enough, try Dijkstra for an optimal solution.
    # State space size is roughly 2^(n_nodes) but constrained by fast_memory_budget.
    # A rough heuristic: if n_nodes <= 20 and fast_memory_budget is small, Dijkstra is feasible.
    # We will attempt Dijkstra if n_nodes is small (<= 25) and budget allows.
    
    best_schedule = None
    best_cost = float('inf')
    
    # --- Attempt 1: Dijkstra's Algorithm for Exact Optimal Schedule ---
    if n_nodes <= 25:
        try:
            # State: (frozenset(red_nodes), frozenset(completed_outputs))
            # Note: We assume inputs are available in slow memory and can be loaded anytime.
            # We assume computed values are only in fast memory until stored.
            
            initial_state = (frozenset(), frozenset())
            
            # Heuristic for A* (optional, but Dijkstra is safer for correctness if weights are uniform-ish)
            # Let's stick to Dijkstra for robustness.
            
            dist = {initial_state: 0}
            parent_map = {initial_state: None}
            pq = [(0, initial_state)]
            
            found_goal = False
            
            while pq:
                if time.time() > deadline - 5.0: # Leave 5s for fallback
                    break
                    
                d, state = heapq.heappop(pq)
                if d > dist.get(state, float('inf')):
                    continue
                    
                red = set(state[0])
                completed = set(state[1])
                
                if completed == output_set:
                    found_goal = True
                    best_cost = d
                    # Reconstruct path
                    path = []
                    curr = state
                    while parent_map[curr] is not None:
                        prev_state, op, node = parent_map[curr]
                        path.append([op, node])
                        curr = prev_state
                    path.reverse()
                    best_schedule = path
                    break
                
                # Generate neighbors
                
                # 1. Load an input
                for inp in inputs:
                    if inp not in red and inp not in completed:
                        if len(red) < fast_memory_budget:
                            new_red = red | {inp}
                            new_state = (frozenset(new_red), state[1])
                            cost = 1
                            new_d = d + cost
                            if new_d < dist.get(new_state, float('inf')):
                                dist[new_state] = new_d
                                parent_map[new_state] = (state, 'load', inp)
                                heapq.heappush(pq, (new_d, new_state))
                
                # 2. Compute a node
                # A node can be computed if:
                # - It is not completed
                # - All its parents are currently red OR are inputs that have been loaded? 
                #   Wait, if a parent is an input and it's blue, we must load it first.
                #   If a parent is a computed node and it's blue (stored), we must load it?
                #   The problem says "Inputs start blue... all outputs must finish blue".
                #   It implies we can load any blue node (input or previously stored output/intermediate?)
                #   "Recomputations allowed". This implies we can re-load stored values.
                #   So, if a parent is not red, we can 'load' it (cost 1) if it's available in slow memory.
                #   Are intermediates available in slow memory? Only if we 'store' them.
                #   If we never stored an intermediate, it's gone.
                #   So, 'load' is only valid for:
                #   - Inputs (always available)
                #   - Nodes that have been 'store'd (completed? No, completed outputs are done.
                #     What about intermediates? If we store an intermediate, it's blue. Can we load it back?
                #     The task says "all outputs must finish blue". It doesn't explicitly forbid storing intermediates.
                #     However, typically in pebbling, you only store outputs.
                #     Let's assume we can only load Inputs and Stored Outputs/Intermediates.
                #     To simplify: We track `completed` as "stored". 
                #     If a node is in `completed`, it is blue and available for loading.
                
                for node in nodes:
                    if node in completed:
                        continue
                    if node in red:
                        continue
                    
                    # Check if all parents are red
                    can_compute = True
                    for p in parents.get(node, []):
                        if p not in red:
                            can_compute = False
                            break
                    
                    if can_compute:
                        # We need space for the node if it's not already in red (it isn't)
                        # If budget is full, we must evict something.
                        # Dijkstra state space needs to handle eviction.
                        # This makes the state space complex.
                        # If len(red) < budget:
                        if len(red) < fast_memory_budget:
                            new_red = red | {node}
                            new_state = (frozenset(new_red), state[1])
                            cost = 0
                            new_d = d + cost
                            if new_d < dist.get(new_state, float('inf')):
                                dist[new_state] = new_d
                                parent_map[new_state] = (state, 'compute', node)
                                heapq.heappush(pq, (new_d, new_state))
                        else:
                            # Need to evict. This branches into all possible evictions.
                            # This explodes the state space.
                            # For now, we skip this branch in Dijkstra to keep it tractable,
                            # relying on the greedy fallback for cases where budget is tight.
                            # Or, we can restrict Dijkstra to cases where budget is large enough.
                            pass
                
                # 3. Store an output
                for node in nodes:
                    if node in red and node in output_set and node not in completed:
                        new_red = red - {node}
                        new_completed = completed | {node}
                        new_state = (frozenset(new_red), frozenset(new_completed))
                        cost = 1
                        new_d = d + cost
                        if new_d < dist.get(new_state, float('inf')):
                            dist[new_state] = new_d
                            parent_map[new_state] = (state, 'store', node)
                            heapq.heappush(pq, (new_d, new_state))
                
                # 4. Delete a red node (optional, but might be needed if we want to free space without loading)
                # Actually, delete is only useful if we want to free space.
                # If we are about to load, we can just load and the system handles eviction?
                # No, the interface requires explicit delete.
                # In Dijkstra, we should allow delete to free space.
                for node in red:
                    if node in output_set and node in completed:
                        # Shouldn't happen, completed outputs are removed from red on store
                        continue
                    new_red = red - {node}
                    new_state = (frozenset(new_red), state[1])
                    cost = 0
                    new_d = d + cost
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent_map[new_state] = (state, 'delete', node)
                        heapq.heappush(pq, (new_d, new_state))

            if found_goal and best_schedule:
                return best_schedule

        except Exception:
            pass # Fall through to greedy

    # --- Attempt 2: Greedy Heuristic with Lookahead Eviction ---
    # This is the strategy from the current program, refined.
    
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    # Track timestamp for LRU
    timestamp = {}
    current_time = 0
    
    # Precompute "future fan-out" dynamically? 
    # We will use a simple heuristic: evict the node with the fewest remaining children.
    
    for node in topo_order:
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in red
        for p in parents.get(node, []):
            if p not in red:
                # Need to load p.
                # If budget is full, evict.
                if len(red) >= fast_memory_budget:
                    # Find victim
                    # Heuristic: Min future children, then LRU
                    min_val = (float('inf'), float('inf'))
                    victim = None
                    for r in red:
                        if r in completed:
                            # Should not be in red if completed
                            continue
                        # Count children of r that are not completed
                        fut = len([c for c in children[r] if c not in completed])
                        t = timestamp.get(r, 0)
                        if (fut, t) < min_val:
                            victim = r
                            min_val = (fut, t)
                    
                    if victim:
                        red.remove(victim)
                        schedule.append(['delete', victim])
                
                red.add(p)
                timestamp[p] = current_time
                current_time += 1
                schedule.append(['load', p])
                transfers += 1
        
        # 2. Compute node
        if node not in red:
            if len(red) >= fast_memory_budget:
                # Evict
                min_val = (float('inf'), float('inf'))
                victim = None
                for r in red:
                    if r in completed:
                        continue
                    fut = len([c for c in children[r] if c not in completed])
                    t = timestamp.get(r, 0)
                    if (fut, t) < min_val:
                        victim = r
                        min_val = (fut, t)
                
                if victim:
                    red.remove(victim)
                    schedule.append(['delete', victim])
            
            red.add(node)
            timestamp[node] = current_time
            current_time += 1
            schedule.append(['compute', node])
        
        # 3. If node is output, store it
        if node in output_set:
            if node in red:
                red.remove(node)
            completed.add(node)
            schedule.append(['store', node])
            transfers += 1
            # Remove from timestamp to avoid confusion
            if node in timestamp:
                del timestamp[node]
    
    # Check if all outputs are completed
    if completed != output_set:
        # Fallback if we didn't finish (e.g. time limit)
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Verify schedule length
    if len(schedule) > 20000:
        # This shouldn't happen for reasonable DAGs with greedy
        pass
        
    return schedule

# EVOLVE-BLOCK-END
