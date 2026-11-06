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

    # If the DAG is small enough, use Dijkstra for optimal solution
    if len(all_vertices) <= 25:
        return pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Otherwise, use improved greedy
    # Precompute children and future use counts
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Count total uses (fan-out) for each vertex
    use_count = {v: len(children[v]) for v in all_vertices}
    
    # Topological sort to process in order
    in_deg = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_deg[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
        
    # If topo sort didn't include all, there's a cycle or disconnected graph, 
    # but DAG implies no cycles. 
    # Ensure all vertices are in topo_order
    if len(topo_order) < len(all_vertices):
        # Add missing vertices (should not happen in valid DAG)
        for v in all_vertices:
            if v not in topo_order:
                topo_order.append(v)

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    
    # To allow "reloading" of computed nodes, we need to know if they are in slow_mem.
    # We will simulate the state.
    
    # We process vertices in topological order.
    # However, we can choose the order of computation within topological constraints 
    # to optimize memory usage. 
    # A simple greedy: Process vertices in topological order. 
    # When a vertex is ready (all parents in fast_mem), compute it.
    # But we might want to delay computation to keep parents in memory?
    # Actually, computing a vertex adds it to fast_mem. 
    # If we compute too many, we evict.
    
    # Let's use a priority queue for "ready" vertices.
    # Priority: High fan-out first? Or Low fan-out first?
    # If we compute a high fan-out vertex early, it stays in memory and can be used by many children, 
    # reducing loads for those children.
    # So, compute high fan-out vertices first?
    # But we need their parents.
    
    # Let's stick to Topological Order but with a twist:
    # We maintain a set of "ready" vertices (parents in fast_mem).
    # We pick the one with the highest "value" to compute next.
    # Value: use_count[v] (how many children it has).
    
    ready = set()
    for v in topo_order:
        if v in input_vertices:
            # Inputs are initially in slow_mem. 
            # We need to load them to fast_mem to compute children.
            # But we don't compute inputs. We just load them when needed.
            pass

    # Initialize: Inputs are in slow_mem. 
    # We don't load them until needed.
    
    # To make this work, we need to dynamically determine readiness.
    # A vertex is ready if all its parents are in fast_mem.
    
    # We iterate through topo_order.
    # For each v in topo_order:
    #   If v is input: 
    #     It's "ready" to be loaded.
    #   If v is not input:
    #     It becomes ready when all parents are in fast_mem.
    
    # This is tricky because loading an input takes space.
    
    # Alternative: 
    # 1. Start with all inputs in slow_mem.
    # 2. Fast_mem is empty.
    # 3. We have a set of "available" vertices to compute/load.
    #    - Inputs are available to load.
    #    - Non-inputs are available to compute if parents are in fast_mem.
    
    # Let's use a loop that continues until all outputs are stored.
    # State: fast_mem, slow_mem, computed (set of vertices that have been computed at least once)
    
    # Actually, the standard greedy for pebbling is:
    # 1. Identify vertices whose parents are in fast_mem.
    # 2. Among those, pick one to compute.
    # 3. If fast_mem is full, evict.
    
    # Let's implement a simple greedy that processes in topological order but 
    # loads parents just-in-time.
    
    # This is essentially the same as the current program but with better eviction.
    # Let's refine the eviction logic.
    
    # Precompute layers (topological depth from inputs)
    layer = {}
    for v in input_vertices:
        layer[v] = 0
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

    # Precompute future use layers for each vertex
    future_use_layers = {v: [] for v in all_vertices}
    for v in all_vertices:
        for c in children.get(v, []):
            cl = layer.get(c, 0)
            future_use_layers[v].append(cl)
        future_use_layers[v].sort()

    # Precompute next_use_index for efficient lookup
    # next_use_idx[v] = list of indices into future_use_layers[v]
    # We'll maintain a pointer or just iterate.
    
    # We process vertices in topological order (layer by layer, then within layer by topo order).
    # We maintain a set of 'active' vertices to process.
    
    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    
    # We need to process all non-input vertices.
    # We iterate through layers.
    
    # To handle eviction and reloading properly, we simulate the pebbling.
    # A vertex is 'ready' if all its parents are in fast_mem.
    # We maintain a set of ready vertices.
    
    # Initial ready: inputs are in slow_mem, so they are not in fast_mem.
    # We need to load them.
    
    # Let's use a priority queue for ready vertices.
    # Priority: based on some heuristic.
    # For now, we process in topological order.
    
    # Actually, the previous layer-based approach was sequential.
    # Let's stick to that but fix the eviction logic.
    
    # The issue with the previous approach:
    # 1. It didn't handle the case where a parent was deleted and not in slow_mem (recomputation needed).
    # 2. The eviction scoring was heuristic and might not be optimal.
    
    # Improvement:
    # When evicting, if a vertex is not in slow_mem and has future uses, we MUST store it first.
    # If it is in slow_mem, we can just delete it.
    # If it is an output, we store it.
    
    # Let's refine the eviction cost:
    # - If u in slow_mem: cost to evict = 0 (delete). Cost to reload later = 1.
    # - If u not in slow_mem: cost to evict = 1 (store) + 0 (delete) = 1. Cost to reload later = 1.
    #   Wait, if we store it, it goes to slow_mem. So next time we load it, cost is 1.
    #   If we don't store it and delete it, we must recompute it later.
    #   Recomputation cost is not just 1, it's the cost of computing all its parents again if they are not in memory.
    #   So it is generally better to store it.
    
    # Heuristic for eviction:
    # We want to evict the vertex that is least likely to be needed soon, or whose eviction is cheapest.
    # Score = (future_use_distance) * (recomputation_risk)
    
    # Let's use a simpler heuristic:
    # 1. Never evict a vertex that is needed for the next vertex to be computed (if possible).
    # 2. Evict the vertex with the furthest next use.
    # 3. If tie, prefer evicting from slow_mem (cheaper to reload) over not in slow_mem.
    
    # Let's implement a greedy that processes vertices in topological order.
    # For each vertex v:
    #   1. Ensure all parents are in fast_mem.
    #      - If parent p is in slow_mem, load it.
    #      - If parent p is not in fast_mem and not in slow_mem, this is an error (should not happen if we store before delete).
    #   2. Compute v.
    #   3. If fast_mem is full, evict.
    
    # We need to be careful about the order of vertices within a layer.
    # Let's just use the topo_order we computed earlier.
    
    # But we need to handle the case where a vertex is not ready yet.
    # Since we process in topo order, all parents are processed before children.
    # So if we ensure parents are in fast_mem, we are good.
    
    # The problem is that parents might have been evicted.
    # If a parent was evicted and stored in slow_mem, we load it.
    # If a parent was evicted and NOT stored, we have a problem.
    # To avoid this, we must ensure that any vertex with future uses is stored before being evicted.
    
    # So the eviction logic is:
    # If we must evict v:
    #   If v has future uses:
    #     If v not in slow_mem: store v.
    #     Delete v.
    #   Else:
    #     Delete v.
    
    # This ensures that any evicted vertex with future uses is in slow_mem.
    
    # Let's refine the choice of which vertex to evict.
    # We want to minimize the cost of future loads/recomputations.
    # A good heuristic is to evict the vertex with the furthest next use.
    
    # Let's implement this.
    
    for v in topo_order:
        if time_check():
            break
        if v in input_vertices:
            continue
        
        # Ensure parents are in fast_mem
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                else:
                    # This should not happen if we store before delete.
                    # If it does, we have to recompute p, which is complex.
                    # For now, we assume it doesn't happen.
                    pass
        
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # Evict if needed
        while len(fast_mem) > fast_memory_budget and not time_check():
            best = None
            best_score = float('inf')
            
            for u in fast_mem:
                # Calculate eviction score
                # Lower score is better to evict
                
                # 1. If u is an output, we must store it.
                #    Storing it is mandatory.
                #    Score: 0 (we will store and delete)
                if u in outputs:
                    score = 0.0
                else:
                    # 2. Find next use layer
                    nu = max_L + 1
                    for nl in future_use_layers[u]:
                        if nl > layer[v]: # Current layer
                            nu = nl
                            break
                    
                    if nu > max_L:
                        # Never used again
                        score = 0.1
                    else:
                        # Used in future
                        dist = nu - layer[v]
                        # Prefer to evict nodes that are in slow_mem (cheaper to reload)
                        # and nodes with further next use.
                        if u in slow_mem:
                            score = 1.0 + dist * 0.1
                        else:
                            # Must store before evicting
                            # Higher penalty
                            score = 10.0 + dist * 0.1
                
                if score < best_score:
                    best_score = score
                    best = u

            if best is None:
                break

            if best in outputs:
                ops.append(['store', best])
                slow_mem.add(best)
                fast_mem.discard(best)
            else:
                if best not in slow_mem:
                    ops.append(['store', best])
                    slow_mem.add(best)
                ops.append(['delete', best])
                fast_mem.discard(best)

    # Store remaining outputs
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
    # State: (frozenset(fast_red), frozenset(black))
    init_black = frozenset(input_vertices)
    init_fast = frozenset()
    
    dist = {}
    prev = {}
    
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
            
        if all(o in black for o in outputs):
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
            ps = parents.get(v, [])
            if all(p in fast or p in black for p in ps):
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['compute', v])
                        heapq.heappush(pq, (new_cost, new_state))
                        
        # 2. Store
        for v in fast:
            if v not in black:
                new_black = black | frozenset([v])
                new_state = (fast, new_black)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 3. Delete
        for v in fast:
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

    return []

# EVOLVE-BLOCK-END
