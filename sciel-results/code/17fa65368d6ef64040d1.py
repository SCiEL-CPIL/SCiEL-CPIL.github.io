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
    # We will attempt Dijkstra if n_nodes is small (<= 22) to keep state space tractable.
    
    best_schedule = None
    best_cost = float('inf')
    
    # --- Attempt 1: Dijkstra's Algorithm for Exact Optimal Schedule ---
    if n_nodes <= 22:
        try:
            # State: (frozenset(red_nodes), frozenset(completed_outputs))
            initial_state = (frozenset(), frozenset())
            
            dist = {initial_state: 0}
            parent_map = {initial_state: None}
            pq = [(0, initial_state)]
            
            found_goal = False
            nodes_list = list(nodes)
            inputs_list = list(inputs)
            outputs_list = list(outputs)
            
            while pq:
                if time.time() > deadline - 8.0:
                    break
                    
                d, state = heapq.heappop(pq)
                if d > dist.get(state, float('inf')):
                    continue
                    
                red = set(state[0])
                completed = set(state[1])
                
                if completed == output_set:
                    found_goal = True
                    best_cost = d
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
                # 1. Load a node (input or previously stored node)
                for inp in inputs_list:
                    if inp not in red:
                        if len(red) < fast_memory_budget:
                            new_red = red | {inp}
                            new_state = (frozenset(new_red), state[1])
                            new_d = d + 1
                            if new_d < dist.get(new_state, float('inf')):
                                dist[new_state] = new_d
                                parent_map[new_state] = (state, 'load', inp)
                                heapq.heappush(pq, (new_d, new_state))
                        else:
                            # Evict then load
                            for victim in red:
                                new_red = red - {victim} | {inp}
                                new_state = (frozenset(new_red), state[1])
                                new_d = d + 2  # delete + load
                                if new_d < dist.get(new_state, float('inf')):
                                    dist[new_state] = new_d
                                    parent_map[new_state] = (state, 'load', inp)
                                    heapq.heappush(pq, (new_d, new_state))
                
                # 2. Compute a node (all parents must be red)
                for node in nodes_list:
                    if node in completed or node in red:
                        continue
                    
                    can_compute = True
                    for p in parents.get(node, []):
                        if p not in red:
                            can_compute = False
                            break
                    
                    if can_compute:
                        if len(red) < fast_memory_budget:
                            new_red = red | {node}
                            new_state = (frozenset(new_red), state[1])
                            new_d = d  # compute costs 0 transfers
                            if new_d < dist.get(new_state, float('inf')):
                                dist[new_state] = new_d
                                parent_map[new_state] = (state, 'compute', node)
                                heapq.heappush(pq, (new_d, new_state))
                        else:
                            # Evict then compute
                            for victim in red:
                                new_red = red - {victim} | {node}
                                new_state = (frozenset(new_red), state[1])
                                new_d = d + 1  # delete only
                                if new_d < dist.get(new_state, float('inf')):
                                    dist[new_state] = new_d
                                    parent_map[new_state] = (state, 'compute', node)
                                    heapq.heappush(pq, (new_d, new_state))
                
                # 3. Store an output
                for node in outputs_list:
                    if node in red and node not in completed:
                        new_red = red - {node}
                        new_completed = completed | {node}
                        new_state = (frozenset(new_red), frozenset(new_completed))
                        new_d = d + 1
                        if new_d < dist.get(new_state, float('inf')):
                            dist[new_state] = new_d
                            parent_map[new_state] = (state, 'store', node)
                            heapq.heappush(pq, (new_d, new_state))
                
                # 4. Delete a red node (free space)
                for node in red:
                    new_red = red - {node}
                    new_state = (frozenset(new_red), state[1])
                    new_d = d  # delete costs 0 transfers
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent_map[new_state] = (state, 'delete', node)
                        heapq.heappush(pq, (new_d, new_state))

            if found_goal and best_schedule:
                return best_schedule

        except Exception:
            pass # Fall through to greedy

    # --- Attempt 2: Greedy Heuristic with Improved Eviction ---
    
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    # Precompute for each node: number of times it will be needed (fan-out count in remaining topo)
    # We'll use a dynamic "remaining uses" counter
    remaining_uses = {}
    for n in nodes:
        remaining_uses[n] = len(children[n])
    
    # Track timestamp for tie-breaking
    timestamp = {}
    current_time = 0
    
    for node in topo_order:
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in red
        for p in parents.get(node, []):
            if p not in red:
                if len(red) >= fast_memory_budget:
                    # Find best victim: minimize remaining_uses, then LRU
                    min_val = (float('inf'), float('inf'))
                    victim = None
                    for r in red:
                        if r in completed:
                            continue
                        ru = remaining_uses.get(r, 0)
                        t = timestamp.get(r, 0)
                        if (ru, t) < min_val:
                            victim = r
                            min_val = (ru, t)
                    
                    if victim is not None:
                        red.remove(victim)
                        schedule.append(['delete', victim])
                        if victim in timestamp:
                            del timestamp[victim]
                
                red.add(p)
                timestamp[p] = current_time
                current_time += 1
                schedule.append(['load', p])
                transfers += 1
        
        # 2. Compute node
        if node not in red:
            if len(red) >= fast_memory_budget:
                min_val = (float('inf'), float('inf'))
                victim = None
                for r in red:
                    if r in completed:
                        continue
                    ru = remaining_uses.get(r, 0)
                    t = timestamp.get(r, 0)
                    if (ru, t) < min_val:
                        victim = r
                        min_val = (ru, t)
                
                if victim is not None:
                    red.remove(victim)
                    schedule.append(['delete', victim])
                    if victim in timestamp:
                        del timestamp[victim]
            
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
            if node in timestamp:
                del timestamp[node]
        
        # Decrement remaining uses for this node's children (node is now processed)
        # Actually, remaining_uses tracks how many children still need this node.
        # After computing node, it's done being a parent for future computations.
        # But we should decrement the parent's remaining_uses when a child is processed.
        # Let's update: after processing node, decrement remaining_uses of each parent
        for p in parents.get(node, []):
            if p in remaining_uses:
                remaining_uses[p] = max(0, remaining_uses[p] - 1)
    
    # Check if all outputs are completed
    if completed != output_set:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if len(schedule) > 20000:
        pass
        
    return schedule

# EVOLVE-BLOCK-END
