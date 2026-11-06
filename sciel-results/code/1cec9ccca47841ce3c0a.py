"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    # If the DAG is small, use Dijkstra for optimal solution
    if len(all_vertices) <= 12:
        return pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Otherwise, use improved greedy
    # Compute layer for each vertex
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

    # Precompute usage info for better eviction heuristics
    use_count = {v: len(children.get(v, [])) for v in all_vertices}
    next_use_layer = {}
    for v in all_vertices:
        next_use_layer[v] = max_L + 1
        for c in children.get(v, []):
            cl = layer.get(c, 0)
            if cl < next_use_layer[v]:
                next_use_layer[v] = cl

    # Compute subtree depth and size for cost estimation
    # Depth: max distance to leaf in children
    # Size: number of nodes in subtree
    # We process in reverse topological order (layers from max to 0)
    subtree_depth = {v: 0 for v in all_vertices}
    subtree_size = {v: 1 for v in all_vertices}
    
    # Topological sort is effectively given by layers, but we need to ensure parents are processed after children for DP?
    # No, depth depends on children. So we process from max layer down to 0.
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

        # Load needed parents
        needed_parents = set()
        for v in lv:
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                needed_parents.add(p)

        for p in needed_parents:
            if p not in fast_mem and p in slow_mem:
                ops.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)

        # Compute vertices in this layer
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)

        # Evict if over budget
        while len(fast_mem) > fast_memory_budget and not time_check():
            best = None
            best_score = float('inf')
            for u in fast_mem:
                # We want to evict the vertex with the LOWEST retention value.
                # Retention value = Cost of re-fetching/recomputing if evicted.
                
                # 1. If it's an output, we MUST store it eventually. 
                #    If we evict it now, we can store it (cost 1) and it's done.
                #    If we keep it, we use a slot.
                #    Storing an output is usually good to free space, unless we need it for something else (outputs are sinks).
                #    So, outputs have low retention priority if we are over budget, UNLESS we haven't stored them yet?
                #    Actually, if we evict an output, we must store it. So it becomes blue.
                #    If we keep it, it stays red (and blue if stored? No, store makes it blue, delete makes it red gone).
                #    Let's assume we prefer to store outputs to free memory.
                
                if u in outputs:
                    # We should store this to free memory. It's a "safe" eviction.
                    score = 0 
                elif u in input_vertices:
                    # Inputs are cheap to reload (cost 1) but recompute is impossible.
                    # However, if we delete an input, we must reload it later.
                    # If we keep it, we use space.
                    # Inputs have moderate retention.
                    score = 100 
                else:
                    # Internal vertex.
                    # Cost to recompute = subtree_size[u] (approx)
                    # Cost to reload = 1 (if it was stored/blue) or impossible (if deleted).
                    # If we delete it, it's gone. We must recompute.
                    # If we store it, it's in slow mem. We can load it later (cost 1).
                    # 
                    # Heuristic:
                    # High subtree_size -> expensive to recompute -> high retention.
                    # High use_count_remaining -> used often -> high retention.
                    # Early next_use_layer -> needed soon -> high retention.
                    
                    nu = next_use_layer.get(u, max_L + 1)
                    uc = use_count.get(u, 0)
                    depth = subtree_depth.get(u, 0)
                    size = subtree_size.get(u, 1)
                    
                    # If not used in future, retention is low.
                    if nu > max_L:
                        score = 10 # Can be deleted safely, no future use.
                    else:
                        # Score represents "value" of keeping. We want to evict lowest value.
                        # Value = (use_count * 10) + (depth * 5) + (size * 1)
                        # But we also penalize if it's needed very soon?
                        # If needed soon, we might have to reload/recompute immediately.
                        # Actually, if it's needed soon, keeping it saves the immediate transfer.
                        
                        value = 0
                        value += uc * 50
                        value += depth * 10
                        value += size * 2
                        
                        # If it is needed in the current or next layer, boost value significantly
                        if nu <= L + 1:
                            value += 1000
                        
                        score = value

                if score < best_score:
                    best_score = score
                    best = u

            if best is None:
                break

            if best in outputs:
                ops.append(['store', best])
                slow_mem.add(best)
            else:
                ops.append(['delete', best])
            fast_mem.discard(best)

    # Store outputs
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
    # State: (cost, frozenset(fast), frozenset(black))
    # Initial: fast = set(), black = input_vertices
    init_black = frozenset(input_vertices)
    init_fast = frozenset()
    
    # Dijkstra
    # dist[fast][black] = cost
    # Since N is small, we can use a dictionary
    dist = {}
    prev = {} # state -> (prev_state, op)
    
    start_state = (init_fast, init_black)
    dist[start_state] = 0
    
    pq = [(0, start_state)]
    
    while pq:
        if time_check():
            break
        cost, state = heapq.heappop(pq)
        fast, black = state
        
        if cost > dist.get(state, float('inf')):
            continue
            
        # Check if goal reached
        if all(o in black for o in outputs):
            # Reconstruct path
            ops = []
            curr = state
            while curr != start_state:
                p_state, op = prev[curr]
                ops.append(op)
                curr = p_state
            ops.reverse()
            return ops
            
        # Generate neighbors
        # 1. Compute
        for v in all_vertices:
            if v in fast or v in black:
                continue
            # Check if parents are available
            ps = parents.get(v, [])
            if all(p in fast or p in black for p in ps):
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost # Compute is free
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['compute', v])
                        heapq.heappush(pq, (new_cost, new_state))
                        
        # 2. Store
        for v in fast:
            if v not in black:
                new_black = black | frozenset([v])
                # Fast remains the same (vertex is both red and blue now? Or just blue?)
                # If it stays red, fast is unchanged.
                # If it becomes blue only, fast removes v.
                # Let's assume it stays red to allow later deletion without reload.
                new_state = (fast, new_black)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 3. Delete
        for v in fast:
            # Can delete if we don't need it in fast? 
            # Deleting is always valid if we have space issues, but it might force reloads.
            # In Dijkstra, we allow all valid moves.
            new_fast = fast - frozenset([v])
            new_state = (new_fast, black)
            new_cost = cost
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                prev[new_state] = (state, ['delete', v])
                heapq.heappush(pq, (new_cost, new_state))
                
        # 4. Load
        for v in black:
            if v not in fast:
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost + 1
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['load', v])
                        heapq.heappush(pq, (new_cost, new_state))

    # Fallback if Dijkstra fails or times out
    return []

# EVOLVE-BLOCK-END
