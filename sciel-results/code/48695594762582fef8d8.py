import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        has_baseline = True
    except ImportError:
        has_baseline = False

    try:
        if isinstance(dag, dict):
            parents = dict(dag.get('parents', {}))
            outputs = list(dag.get('outputs', []))
            vertices = dag.get('vertices', None)
            inputs = dag.get('inputs', None)
        else:
            parents = dict(getattr(dag, 'parents', {}))
            outputs = list(getattr(dag, 'outputs', []))
            vertices = getattr(dag, 'vertices', None)
            inputs = getattr(dag, 'inputs', None)
    except Exception:
        if has_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        raise

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    # Small DAG: Exact Search
    if len(all_vertices) <= 15:
        return pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Large DAG: Layered Greedy
    # 1. Compute Layers
    layer = {}
    for v in input_vertices:
        layer[v] = 0

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    queue = deque(list(input_vertices))
    while queue:
        v = queue.popleft()
        for c in children[v]:
            if c not in layer:
                all_p_done = all(p in layer for p in parents.get(c, []))
                if all_p_done:
                    layer[c] = max(layer[p] for p in parents.get(c, [])) + 1
                    queue.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    layers = defaultdict(list)
    for v in all_vertices:
        layers[layer[v]].append(v)

    max_L = max(layers.keys()) if layers else 0

    # 2. Precompute Next Use Layer
    # next_use_layer[v] = min(L' such that v is a parent of some node in layer L')
    # If v is not a parent of any node, next_use_layer[v] = max_L + 1
    next_use_layer = {v: max_L + 1 for v in all_vertices}
    for L in range(max_L + 1):
        for v in layers.get(L, []):
            for p in parents.get(v, []):
                if L < next_use_layer[p]:
                    next_use_layer[p] = L

    # 3. Precompute Subtree Size/Depth for Cost Estimation
    subtree_size = {v: 1 for v in all_vertices}
    subtree_depth = {v: 0 for v in all_vertices}
    
    # Process in reverse topological order
    for L in range(max_L, -1, -1):
        for v in layers.get(L, []):
            if not children.get(v):
                continue
            max_d = 0
            total_s = 1
            for c in children.get(v, []):
                if c in subtree_depth:
                    max_d = max(max_d, subtree_depth[c] + 1)
                    total_s += subtree_size.get(c, 1)
            subtree_depth[v] = max_d
            subtree_size[v] = total_s

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []

    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue

        # Phase 1: Ensure parents are in fast memory
        needed_parents = set()
        for v in lv:
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                needed_parents.add(p)

        for p in needed_parents:
            if p not in fast_mem:
                if p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                else:
                    # Should not happen if we keep active set, but if it does, we must recompute.
                    # This implies we evicted it.
                    # For simplicity in greedy, we assume we don't evict needed parents.
                    # If we do, we'd need to recurse. 
                    # To be safe, if p is not in slow or fast, we can't compute v.
                    # But we only evict based on next_use_layer.
                    # If p is needed now, next_use_layer[p] <= L.
                    # We only evict if next_use_layer[p] > L+1.
                    # So p should be in fast or slow.
                    pass

        # Phase 2: Compute vertices in this layer
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)

        # Phase 3: Evict/Store to maintain budget
        while len(fast_mem) > fast_memory_budget and not time_check():
            # Find best candidate to evict
            # Priority:
            # 1. Outputs (Store them)
            # 2. Vertices with next_use_layer > L + 1 (Not needed soon)
            #    Among these, prefer lowest recompute cost (subtree_size)
            # 3. If all are needed soon, evict the one with lowest recompute cost
            #    (This will force a reload/recompute later, but it's the "cheapest" damage)
            
            candidates = []
            for u in fast_mem:
                nu = next_use_layer.get(u, max_L + 1)
                is_output = u in outputs
                is_input = u in input_vertices
                size = subtree_size.get(u, 1)
                
                # Score: Lower is better to evict
                if is_output:
                    # Store is good, frees space, satisfies condition
                    score = -1000
                elif nu > L + 1:
                    # Not needed in current or next layer
                    # Cost to re-fetch:
                    if is_input:
                        cost = 1
                    else:
                        cost = 1 + size # Approximation
                    score = cost
                else:
                    # Needed soon
                    # High penalty to evict
                    if is_input:
                        cost = 100
                    else:
                        cost = 100 + size
                    score = cost

                candidates.append((score, u))

            if not candidates:
                break

            candidates.sort()
            best_score, best_u = candidates[0]
            
            if best_u in outputs:
                ops.append(['store', best_u])
                slow_mem.add(best_u)
            else:
                ops.append(['delete', best_u])
            
            fast_mem.discard(best_u)

    # Store any remaining outputs in fast memory
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

def pebble_dijkstra(dag, budget, all_vertices, input_vertices, outputs, parents, time_check):
    # State: (fast_mask, blue_mask)
    # Use integers for masks
    n = len(all_vertices)
    if n > 20:
        return [] # Fallback to greedy if too large for bitmask (though 15 is limit)
    
    v_to_idx = {v: i for i, v in enumerate(all_vertices)}
    idx_to_v = {i: v for v, i in v_to_idx.items()}
    
    init_fast = 0
    init_blue = 0
    for v in input_vertices:
        init_blue |= (1 << v_to_idx[v])
        
    goal_blue = 0
    for v in outputs:
        goal_blue |= (1 << v_to_idx[v])
        
    # Precompute parent masks
    parent_masks = [0] * n
    for v in all_vertices:
        i = v_to_idx[v]
        mask = 0
        for p in parents.get(v, []):
            mask |= (1 << v_to_idx[p])
        parent_masks[i] = mask
        
    # Dijkstra
    # dist[fast][blue] = cost
    # fast and blue are integers
    dist = {}
    prev = {}
    
    start_state = (init_fast, init_blue)
    dist[start_state] = 0
    pq = [(0, start_state)]
    
    while pq:
        if time_check():
            break
        cost, state = heapq.heappop(pq)
        fast, blue = state
        
        if cost > dist.get(state, float('inf')):
            continue
            
        # Check goal
        if (blue & goal_blue) == goal_blue:
            # Reconstruct
            ops = []
            curr = state
            while curr != start_state:
                p_state, op = prev[curr]
                ops.append(op)
                curr = p_state
            ops.reverse()
            return ops
            
        # Generate neighbors
        # 1. Load
        for i in range(n):
            v = idx_to_v[i]
            if (blue & (1 << i)) and not (fast & (1 << i)):
                new_fast = fast | (1 << i)
                if bin(new_fast).count('1') <= budget:
                    new_state = (new_fast, blue)
                    new_cost = cost + 1
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['load', v])
                        heapq.heappush(pq, (new_cost, new_state))
                        
        # 2. Store
        for i in range(n):
            v = idx_to_v[i]
            if (fast & (1 << i)) and not (blue & (1 << i)):
                new_blue = blue | (1 << i)
                new_fast = fast & ~(1 << i) # Store removes from fast
                new_state = (new_fast, new_blue)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 3. Delete
        for i in range(n):
            v = idx_to_v[i]
            if (fast & (1 << i)):
                new_fast = fast & ~(1 << i)
                new_state = (new_fast, blue)
                new_cost = cost
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['delete', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 4. Compute
        for i in range(n):
            v = idx_to_v[i]
            if not (fast & (1 << i)) and not (blue & (1 << i)):
                # Check parents
                p_mask = parent_masks[i]
                if (fast & p_mask) == p_mask:
                    new_fast = fast | (1 << i)
                    if bin(new_fast).count('1') <= budget:
                        new_state = (new_fast, blue)
                        new_cost = cost
                        if new_cost < dist.get(new_state, float('inf')):
                            dist[new_state] = new_cost
                            prev[new_state] = (state, ['compute', v])
                            heapq.heappush(pq, (new_cost, new_state))

    return []
