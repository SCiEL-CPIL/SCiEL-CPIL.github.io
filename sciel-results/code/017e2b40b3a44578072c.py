"""Improved DAG pebbling with better heuristics and layer-based scheduling."""
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

    # Parse DAG
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

    # Small DAG: use Dijkstra
    if len(all_vertices) <= 15:
        return pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Compute topological order
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
    if len(topo_order) < len(all_vertices):
        for v in all_vertices:
            if v not in topo_order:
                topo_order.append(v)

    # Compute layer for each vertex
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
    max_L = max(layer.values()) if layer else 0

    # Precompute: for each vertex, the layers of its children
    child_layers = {v: sorted(layer[c] for c in children[v]) for v in all_vertices}

    # State
    slow_mem = set(input_vertices)  # blue vertices
    fast_mem = set()                # red vertices
    ops = []

    # Track remaining computations
    to_compute = set(all_vertices - input_vertices)

    max_steps = len(all_vertices) * 10 + 100
    step = 0

    while step < max_steps and not time_check():
        step += 1

        # Check if done
        if all(o in slow_mem for o in outputs):
            break

        # Find ready vertices: all parents in fast_mem
        ready = set()
        for v in to_compute:
            if v in fast_mem or v in slow_mem:
                continue
            ps = parents.get(v, [])
            if all(p in fast_mem for p in ps):
                ready.add(v)

        if ready:
            # Pick the best vertex to compute using improved heuristic
            best_v = None
            best_score = -float('inf')
            for v in ready:
                fanout = len(children[v])
                # Prefer vertices that are used soon and have high fanout
                if children[v]:
                    min_child_layer = min(child_layers[v])
                    # How many children are in the next few layers?
                    upcoming = sum(1 for cl in child_layers[v] if cl <= layer.get(v, 0) + 3)
                else:
                    min_child_layer = max_L + 1
                    upcoming = 0
                
                # Score: high fanout, early use, many upcoming uses
                score = fanout * 10.0 - min_child_layer * 2.0 + upcoming * 5.0
                # Bonus if it's an output
                if v in outputs:
                    score += 100.0
                
                if score > best_score:
                    best_score = score
                    best_v = v

            # Evict if needed to make room
            while len(fast_mem) >= fast_memory_budget and best_v not in fast_mem:
                if time_check():
                    break
                best_evict = None
                best_evict_score = float('inf')
                for u in fast_mem:
                    if u in outputs:
                        # Prefer to store outputs
                        score = -10.0
                    else:
                        # Check future uses
                        future_uses = [nl for nl in child_layers[u] if nl > layer.get(best_v, 0)]
                        if not future_uses:
                            score = 0.0
                        else:
                            min_future = min(future_uses)
                            dist = min_future - layer.get(best_v, 0)
                            if u in slow_mem:
                                score = 1.0 + dist * 0.1
                            else:
                                score = 10.0 + dist * 0.1
                    
                    if score < best_evict_score:
                        best_evict_score = score
                        best_evict = u
                
                if best_evict is None:
                    break
                
                if best_evict in outputs:
                    ops.append(['store', best_evict])
                    slow_mem.add(best_evict)
                    fast_mem.discard(best_evict)
                elif best_evict in slow_mem:
                    ops.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
                else:
                    ops.append(['store', best_evict])
                    slow_mem.add(best_evict)
                    ops.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
            
            # Compute the vertex
            ops.append(['compute', best_v])
            fast_mem.add(best_v)
            to_compute.discard(best_v)

        else:
            # No ready vertices. Need to load a parent.
            blocked = set()
            for v in to_compute:
                if v in fast_mem or v in slow_mem:
                    continue
                ps = parents.get(v, [])
                if not all(p in fast_mem for p in ps):
                    blocked.add(v)
            
            if not blocked:
                break
            
            # Find the most urgent blocked vertex
            best_blocked = None
            best_blocked_score = -float('inf')
            for v in blocked:
                fanout = len(children[v])
                if children[v]:
                    min_child_layer = min(child_layers[v])
                    upcoming = sum(1 for cl in child_layers[v] if cl <= layer.get(v, 0) + 3)
                else:
                    min_child_layer = max_L + 1
                    upcoming = 0
                score = fanout * 10.0 - min_child_layer * 2.0 + upcoming * 5.0
                if v in outputs:
                    score += 100.0
                if score > best_blocked_score:
                    best_blocked_score = score
                    best_blocked = v
            
            # Load a parent of best_blocked that is in slow_mem but not in fast_mem
            ps = parents.get(best_blocked, [])
            load_candidate = None
            for p in ps:
                if p in slow_mem and p not in fast_mem:
                    load_candidate = p
                    break
            
            if load_candidate is None:
                break
            
            # Evict if needed
            while len(fast_mem) >= fast_memory_budget:
                if time_check():
                    break
                best_evict = None
                best_evict_score = float('inf')
                for u in fast_mem:
                    if u in outputs:
                        score = -10.0
                    else:
                        future_uses = [nl for nl in child_layers[u] if nl > layer.get(load_candidate, 0)]
                        if not future_uses:
                            score = 0.0
                        else:
                            min_future = min(future_uses)
                            dist = min_future - layer.get(load_candidate, 0)
                            if u in slow_mem:
                                score = 1.0 + dist * 0.1
                            else:
                                score = 10.0 + dist * 0.1
                    
                    if score < best_evict_score:
                        best_evict_score = score
                        best_evict = u
                
                if best_evict is None:
                    break
                
                if best_evict in outputs:
                    ops.append(['store', best_evict])
                    slow_mem.add(best_evict)
                    fast_mem.discard(best_evict)
                elif best_evict in slow_mem:
                    ops.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
                else:
                    ops.append(['store', best_evict])
                    slow_mem.add(best_evict)
                    ops.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
            
            ops.append(['load', load_candidate])
            fast_mem.add(load_candidate)
        
        # Store any outputs that are in fast_mem
        for o in list(fast_mem):
            if o in outputs:
                ops.append(['store', o])
                slow_mem.add(o)
                fast_mem.discard(o)

    # Final: ensure all outputs are in slow_mem
    for o in outputs:
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

def pebble_dijkstra(dag, budget, all_vertices, input_vertices, outputs, parents, time_check):
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
        for v in all_vertices:
            if v in fast or v in black:
                continue
            ps = parents.get(v, [])
            if all(p in fast for p in ps):
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['compute', v])
                        heapq.heappush(pq, (new_cost, new_state))
                        
        for v in fast:
            if v not in black:
                new_black = black | frozenset([v])
                new_state = (fast, new_black)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        for v in fast:
            new_fast = fast - frozenset([v])
            new_state = (new_fast, black)
            new_cost = cost
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                prev[new_state] = (state, ['delete', v])
                heapq.heappush(pq, (new_cost, new_state))
                
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
