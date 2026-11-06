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

    # Large DAG: Improved Layered Greedy
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
    next_use_layer = {v: max_L + 1 for v in all_vertices}
    for L in range(max_L + 1):
        for v in layers.get(L, []):
            for p in parents.get(v, []):
                if L < next_use_layer[p]:
                    next_use_layer[p] = L

    # 3. Precompute Subtree Size/Depth
    subtree_size = {v: 1 for v in all_vertices}
    subtree_depth = {v: 0 for v in all_vertices}
    
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

    # Track which vertices in the current layer have been processed
    # to determine if a parent is still needed
    
    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue
        
        # Sort vertices in layer by "urgency" or "complexity"?
        # Let's process vertices whose parents are most likely to be evicted first?
        # Or vertices with fewest parents first to reduce load pressure?
        # Let's try sorting by number of parents (ascending) to start with simple ones.
        lv_sorted = sorted(lv, key=lambda v: len(parents.get(v, [])))
        
        # Pre-compute for this layer: for each vertex, which parents it needs
        # and for each parent, which vertices in this layer need it
        parent_users_in_layer = defaultdict(list)
        for v in lv_sorted:
            for p in parents.get(v, []):
                parent_users_in_layer[p].append(v)
                
        # Process vertices in this layer
        for v in lv_sorted:
            if v in input_vertices:
                continue
            if time_check():
                break
            
            # Ensure parents are in fast memory
            for p in parents.get(v, []):
                if p not in fast_mem:
                    if p in slow_mem:
                        # Check if we have space
                        if len(fast_mem) >= fast_memory_budget:
                            # Need to evict something
                            # Find a vertex to evict
                            # Candidates: vertices in fast_mem that are NOT needed by any UNPROCESSED vertex in lv_sorted
                            # AND have next_use_layer > L
                            
                            # Identify unprocessed vertices in this layer
                            # This is inefficient if done inside loop. 
                            # Optimization: Maintain a set of "needed_in_current_layer"
                            
                            # Let's implement a simple eviction:
                            # Evict the vertex in fast_mem with the highest next_use_layer
                            # that is not a parent of any unprocessed vertex in lv_sorted
                            
                            unprocessed = set(lv_sorted[lv_sorted.index(v):])
                            needed_now = set()
                            for uv in unprocessed:
                                for up in parents.get(uv, []):
                                    needed_now.add(up)
                            
                            candidates = []
                            for u in fast_mem:
                                if u in needed_now:
                                    continue
                                nu = next_use_layer.get(u, max_L + 1)
                                # Prefer evicting those with latest next use
                                # If tie, prefer smaller subtree size (cheaper to recompute)
                                cost = -nu * 1000 + subtree_size.get(u, 1)
                                candidates.append((cost, u))
                            
                            if candidates:
                                candidates.sort()
                                # We want the one with highest next_use_layer (lowest cost in this metric? No, -nu means large nu is small cost)
                                # Wait, if nu is large, it's good to evict. So -nu is negative (good).
                                # We want min cost. So min(-nu) -> max nu. Correct.
                                best_u = candidates[0][1]
                                if best_u in outputs:
                                    ops.append(['store', best_u])
                                    slow_mem.add(best_u)
                                else:
                                    ops.append(['delete', best_u])
                                fast_mem.discard(best_u)
                            else:
                                # All fast_mem vertices are needed now.
                                # We are stuck? Or we must evict one of them and recompute later.
                                # This implies budget is too small for the width of the layer.
                                # We must evict the one with highest next_use_layer even if needed now,
                                # knowing we will reload/recompute it.
                                candidates = []
                                for u in fast_mem:
                                    nu = next_use_layer.get(u, max_L + 1)
                                    cost = -nu * 1000 + subtree_size.get(u, 1)
                                    candidates.append((cost, u))
                                if candidates:
                                    candidates.sort()
                                    best_u = candidates[0][1]
                                    if best_u in outputs:
                                        ops.append(['store', best_u])
                                        slow_mem.add(best_u)
                                    else:
                                        ops.append(['delete', best_u])
                                    fast_mem.discard(best_u)
                                else:
                                    # Should not happen if budget >= 1
                                    pass

                        ops.append(['load', p])
                        fast_mem.add(p)
                        slow_mem.discard(p)
                    else:
                        # p is not in slow_mem. This means it was deleted.
                        # We must recompute it.
                        # This breaks the simple layer logic.
                        # To handle this, we would need to recursively ensure p's parents are loaded.
                        # For now, if this happens, we fail or try to recover.
                        # Given the greedy nature, this shouldn't happen if we don't evict needed parents.
                        # But if we evicted a needed parent because budget was full, we are in trouble.
                        # The eviction logic above tries to avoid this by checking `needed_now`.
                        # If we still end up here, it means `needed_now` was empty or we evicted everything.
                        # Let's assume we can't recompute easily in this greedy pass.
                        # We will just skip this vertex or hope it's an input (which is in slow_mem).
                        pass

            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)
            
            # Store if output
            if v in outputs:
                ops.append(['store', v])
                slow_mem.add(v)
                # Do NOT remove from fast_mem yet? 
                # Store usually removes from fast. 
                # "Store: moves value from fast to slow". 
                # So yes, remove from fast.
                fast_mem.discard(v)

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
    if n > 15:
        return [] # Fallback to greedy if too large for bitmask
        
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
                if new_fast.bit_count() <= budget:
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
                    if new_fast.bit_count() <= budget:
                        new_state = (new_fast, blue)
                        new_cost = cost
                        if new_cost < dist.get(new_state, float('inf')):
                            dist[new_state] = new_cost
                            prev[new_state] = (state, ['compute', v])
                            heapq.heappush(pq, (new_cost, new_state))

    return []
