import heapq
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Extract DAG info
    if isinstance(dag, dict):
        parents_dict = dict(dag.get('parents', {}))
        inputs = list(dag.get('inputs', []))
        outputs = list(dag.get('outputs', []))
    else:
        parents_dict = dict(getattr(dag, 'parents', {}))
        inputs = list(getattr(dag, 'inputs', []))
        outputs = list(getattr(dag, 'outputs', []))
    
    # Get all vertices
    all_vertices = set(inputs)
    for v, ps in parents_dict.items():
        all_vertices.add(v)
        all_vertices.update(ps)
    for v in outputs:
        all_vertices.add(v)
        
    # Map vertices to indices
    vertex_list = list(all_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    N = len(vertex_list)
    
    # Bitmasks
    inputs_mask = 0
    for v in inputs:
        inputs_mask |= (1 << vertex_to_idx[v])
        
    outputs_mask = 0
    for v in outputs:
        outputs_mask |= (1 << vertex_to_idx[v])
        
    parents_mask = [0] * N
    for v in vertex_list:
        idx = vertex_to_idx[v]
        ps = parents_dict.get(v, [])
        pm = 0
        for p in ps:
            if p in vertex_to_idx:
                pm |= (1 << vertex_to_idx[p])
        parents_mask[idx] = pm
        
    # Dijkstra over (mask, stored_mask)
    # mask: vertices in fast memory
    # stored_mask: vertices available in slow memory (inputs + stored computed)
    
    # State: (cost, mask, stored_mask)
    # We want min cost to reach a state where outputs are in stored_mask
    
    # Initial state
    init_mask = 0
    init_stored = inputs_mask
    init_state = (init_mask, init_stored)
    
    # If N is too large for bitmask DP, we might need to fallback
    # But for "small" DAGs, this should work.
    # Max states: 2^N * 2^N = 4^N. For N=15, 4^15 ~ 1e9, too big.
    # However, reachable states are much fewer due to DAG structure and budget.
    # Budget K limits mask to combinations of size <= K.
    # stored_mask can be anything.
    
    # Let's try Dijkstra.
    # To keep it fast, we limit the number of states visited.
    
    # Heuristic: A* might be better, but Dijkstra is exact.
    # Let's use Dijkstra with a cap on visited states.
    
    # dist[(mask, stored_mask)] = min_cost
    dist = {}
    prev = {} # To reconstruct path
    
    pq = []
    heapq.heappush(pq, (0, init_mask, init_stored))
    dist[init_state] = 0
    
    goal_found = False
    goal_state = None
    visited_count = 0
    
    while pq:
        if time.time() > deadline:
            break
        if visited_count > 500000: # Cap to prevent explosion
            break
            
        cost, mask, stored_mask = heapq.heappop(pq)
        state = (mask, stored_mask)
        
        if dist.get(state, float('inf')) < cost:
            continue
            
        visited_count += 1
        
        # Check goal
        if (stored_mask & outputs_mask) == outputs_mask:
            goal_found = True
            goal_state = state
            break
            
        # Generate neighbors
        # 1. Load v
        # 2. Store v
        # 3. Compute v
        # 4. Delete v
        
        # Helper to get set bits
        def bits(m):
            while m:
                lsb = m & -m
                yield lsb
                m ^= lsb
        
        # Load
        # v not in mask, v in stored_mask (or input)
        # space available
        popcount = bin(mask).count('1')
        if popcount < fast_memory_budget:
            # Candidates for load: in stored_mask, not in mask
            load_candidates = stored_mask & ~mask
            for lsb in bits(load_candidates):
                v_idx = lsb.bit_length() - 1
                new_mask = mask | lsb
                new_stored = stored_mask
                new_cost = cost + 1
                new_state = (new_mask, new_stored)
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    prev[new_state] = (state, 'load', v_idx)
                    heapq.heappush(pq, (new_cost, new_mask, new_stored))
                    
        # Store
        # v in mask
        store_candidates = mask
        for lsb in bits(store_candidates):
            v_idx = lsb.bit_length() - 1
            new_mask = mask & ~lsb
            new_stored = stored_mask | lsb
            new_cost = cost + 1
            new_state = (new_mask, new_stored)
            if dist.get(new_state, float('inf')) > new_cost:
                dist[new_state] = new_cost
                prev[new_state] = (state, 'store', v_idx)
                heapq.heappush(pq, (new_cost, new_mask, new_stored))
                
        # Compute
        # v not in mask
        # parents in mask
        # space available
        if popcount < fast_memory_budget:
            # v not in mask
            not_in_mask = ~mask
            # Iterate all v? No, only those whose parents are in mask.
            # We can check all v not in mask.
            for v_idx in range(N):
                lsb = 1 << v_idx
                if mask & lsb:
                    continue
                # Check parents
                if (parents_mask[v_idx] & mask) == parents_mask[v_idx]:
                    new_mask = mask | lsb
                    new_stored = stored_mask
                    new_cost = cost # Compute is free
                    new_state = (new_mask, new_stored)
                    if dist.get(new_state, float('inf')) > new_cost:
                        dist[new_state] = new_cost
                        prev[new_state] = (state, 'compute', v_idx)
                        heapq.heappush(pq, (new_cost, new_mask, new_stored))
                        
        # Delete
        # v in mask
        delete_candidates = mask
        for lsb in bits(delete_candidates):
            v_idx = lsb.bit_length() - 1
            new_mask = mask & ~lsb
            new_stored = stored_mask
            new_cost = cost # Delete is free
            new_state = (new_mask, new_stored)
            if dist.get(new_state, float('inf')) > new_cost:
                dist[new_state] = new_cost
                prev[new_state] = (state, 'delete', v_idx)
                heapq.heappush(pq, (new_cost, new_mask, new_stored))
                
    if not goal_found:
        # Fallback to greedy if Dijkstra fails or times out
        return _greedy_pebble(dag, fast_memory_budget)
        
    # Reconstruct path
    operations = []
    state = goal_state
    while state in prev:
        p_state, op, v_idx = prev[state]
        v = vertex_list[v_idx]
        operations.append([op, v])
        state = p_state
        
    operations.reverse()
    return operations

def _greedy_pebble(dag, fast_memory_budget):
    # Fallback implementation similar to original
    if isinstance(dag, dict):
        parents_dict = dict(dag.get('parents', {}))
        inputs = list(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        parents_dict = dict(getattr(dag, 'parents', {}))
        inputs = list(getattr(dag, 'inputs', []))
        outputs = set(getattr(dag, 'outputs', []))
        
    all_vertices = set(inputs)
    for v, ps in parents_dict.items():
        all_vertices.add(v)
        all_vertices.update(ps)
        
    in_degree = {v: len(parents_dict.get(v, [])) for v in all_vertices}
    children = {v: [] for v in all_vertices}
    for v, ps in parents_dict.items():
        for p in ps:
            children[p].append(v)
            
    from collections import deque
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    future_uses = {v: 0 for v in all_vertices}
    for v in topo_order:
        for c in children[v]:
            future_uses[v] += 1
            
    in_fast_memory = set()
    operations = []
    
    for v in topo_order:
        if v in inputs:
            if v not in in_fast_memory:
                in_fast_memory.add(v)
                operations.append(['load', v])
        else:
            ps = parents_dict.get(v, [])
            for p in ps:
                if p not in in_fast_memory:
                    if len(in_fast_memory) >= fast_memory_budget:
                        for u in list(in_fast_memory):
                            if u not in outputs and future_uses.get(u, 0) == 0:
                                in_fast_memory.discard(u)
                                operations.append(['delete', u])
                                break
                        else:
                            for u in list(in_fast_memory):
                                if u not in outputs:
                                    in_fast_memory.discard(u)
                                    operations.append(['delete', u])
                                    break
                    in_fast_memory.add(p)
                    operations.append(['load', p])
                    
            in_fast_memory.add(v)
            operations.append(['compute', v])
            
        if future_uses.get(v, 0) == 0 and v not in outputs:
            if v in in_fast_memory:
                in_fast_memory.discard(v)
                operations.append(['delete', v])
        elif v in outputs:
            if v in in_fast_memory:
                in_fast_memory.discard(v)
                operations.append(['store', v])
                
    for v in outputs:
        if v in in_fast_memory:
            in_fast_memory.discard(v)
            operations.append(['store', v])
            
    return operations
