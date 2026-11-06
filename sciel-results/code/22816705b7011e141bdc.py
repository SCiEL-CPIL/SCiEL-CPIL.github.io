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

    # For small graphs use BFS over states
    if len(all_vertices) <= 12:
        ops = pebble_bfs_small(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)
        if ops is not None:
            return ops

    # For larger graphs, use greedy layer-based approach
    ops = pebble_greedy_layer(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)
    return ops


def pebble_bfs_small(dag, budget, all_vertices, input_vertices, outputs, parents, time_check):
    """
    BFS over states: (frozenset(red), frozenset(blue))
    red = values in fast memory, blue = values already stored
    """
    init_red = frozenset()
    init_blue = frozenset(input_vertices)
    start = (init_red, init_blue)
    goal_blue = frozenset(outputs)

    from collections import deque
    dist = {start: 0}
    parent_map = {}  # state -> (prev_state, op)
    q = deque([start])

    while q:
        if time_check():
            return None
        red, blue = q.popleft()
        cost = dist[(red, blue)]

        # Check goal
        if goal_blue.issubset(blue):
            # reconstruct
            ops = []
            cur = (red, blue)
            while cur in parent_map:
                prev_state, op = parent_map[cur]
                ops.append(op)
                cur = prev_state
            ops.reverse()
            return ops

        # Generate moves
        # 1) Compute: for v not in red|blue, parents in red|blue, len(red)<budget
        for v in all_vertices:
            if v in red or v in blue:
                continue
            ps = parents.get(v, [])
            if all(p in red or p in blue for p in ps):
                if len(red) < budget:
                    new_state = (red | frozenset({v}), blue)
                    if new_state not in dist:
                        dist[new_state] = cost
                        parent_map[new_state] = ((red, blue), ['compute', v])
                        q.append(new_state)

        # 2) Load: v in blue, v not in red, len(red)<budget
        for v in blue:
            if v not in red and len(red) < budget:
                new_state = (red | frozenset({v}), blue)
                if new_state not in dist:
                    dist[new_state] = cost + 1
                    parent_map[new_state] = ((red, blue), ['load', v])
                    q.append(new_state)

        # 3) Store: v in red, v not in blue
        for v in red:
            if v not in blue:
                new_state = (red, blue | frozenset({v}))
                if new_state not in dist:
                    dist[new_state] = cost + 1
                    parent_map[new_state] = ((red, blue), ['store', v])
                    q.append(new_state)

        # 4) Delete: v in red (only if not last needed? we just allow)
        # In classic pebbling, delete is free. We allow delete any red.
        for v in red:
            new_state = (red - frozenset({v}), blue)
            if new_state not in dist:
                dist[new_state] = cost
                parent_map[new_state] = ((red, blue), ['delete', v])
                q.append(new_state)

    return None


def pebble_greedy_layer(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check):
    # Build layer assignment
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    layer = {}
    for v in input_vertices:
        layer[v] = 0

    q = deque(list(input_vertices))
    while q:
        v = q.popleft()
        for c in children[v]:
            if c not in layer:
                ps = parents.get(c, [])
                if all(p in layer for p in ps):
                    layer[c] = max(layer[p] for p in ps) + 1
                    q.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    layers = defaultdict(list)
    for v in all_vertices:
        layers[layer[v]].append(v)
    max_L = max(layers.keys()) if layers else 0

    # Compute future use info
    use_count = {v: len(children.get(v, [])) for v in all_vertices}
    next_use_layer = {v: max_L + 1 for v in all_vertices}
    for v in all_vertices:
        nxl = max_L + 1
        for c in children.get(v, []):
            cl = layer.get(c, 0)
            if cl < nxl:
                nxl = cl
        next_use_layer[v] = nxl

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []

    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue

        # Load needed parents for this layer
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
            # If we don't have all parents fast (could have been deleted), load them
            for p in parents.get(v, []):
                if p not in fast_mem and p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)

        # Evict if over budget
        while len(fast_mem) > fast_memory_budget and not time_check():
            best = None
            best_score = float('inf')
            for u in fast_mem:
                if u in outputs:
                    score = 1e9
                elif u in input_vertices:
                    score = 1
                else:
                    nu = next_use_layer.get(u, max_L + 1)
                    uc = use_count.get(u, 0)
                    if nu > max_L:
                        score = 2
                    else:
                        score = 10000 - (nu * 100 + uc * 10)
                        if nu <= L + 1:
                            score += 50000
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
        elif o in slow_mem:
            ops.append(['load', o])
            fast_mem.add(o)
            slow_mem.discard(o)
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)

    if len(ops) > 20000:
        ops = ops[:20000]
    return ops

# EVOLVE-BLOCK-END
