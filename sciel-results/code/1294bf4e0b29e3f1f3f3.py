"""DAG pebbling with A* search and deadline guard."""
# EVOLVE-BLOCK-START
import time
import heapq


def pebble(dag, fast_memory_budget):
    deadline = time.time() + 85.0
    MAX_OPS = 20000

    vertices = dag['vertices']
    outputs = dag['outputs']
    n = len(vertices)

    parents = [tuple(v['parents']) for v in vertices]
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)

    inputs = [i for i in range(n) if len(parents[i]) == 0]
    is_output = [False] * n
    for o in outputs:
        is_output[o] = True

    initial_state = 0
    for i in inputs:
        initial_state |= (1 << i)

    def is_goal(state):
        for o in outputs:
            if not (state & (1 << o)):
                return False
        return True

    def heuristic(state):
        h = 0
        for o in outputs:
            if not (state & (1 << o)):
                h += 1
        return h

    def can_compute(state, v):
        for p in parents[v]:
            if not (state & (1 << p)):
                return False
        return True

    best_g = {initial_state: 0}
    parent_map = {}
    heap = [(heuristic(initial_state), 0, initial_state)]

    found_goal = None

    while heap:
        if time.time() > deadline - 5.0:
            break

        f, g, state = heapq.heappop(heap)
        if g > best_g.get(state, float('inf')):
            continue

        if is_goal(state):
            found_goal = state
            break

        red_count = bin(state).count('1')

        # Load
        if red_count < fast_memory_budget:
            for v in range(n):
                if not (state & (1 << v)):
                    ns = state | (1 << v)
                    ng = g + 1
                    if ng < best_g.get(ns, float('inf')) and ng <= MAX_OPS:
                        best_g[ns] = ng
                        parent_map[ns] = (state, 'load', v)
                        heapq.heappush(heap, (ng + heuristic(ns), ng, ns))

        # Store
        for v in range(n):
            if state & (1 << v):
                ns = state & ~(1 << v)
                ng = g + 1
                if ng < best_g.get(ns, float('inf')) and ng <= MAX_OPS:
                    best_g[ns] = ng
                    parent_map[ns] = (state, 'store', v)
                    heapq.heappush(heap, (ng + heuristic(ns), ng, ns))

        # Compute
        for v in range(n):
            if not (state & (1 << v)) and can_compute(state, v):
                if red_count < fast_memory_budget:
                    ns = state | (1 << v)
                    ng = g + 1
                    if ng < best_g.get(ns, float('inf')) and ng <= MAX_OPS:
                        best_g[ns] = ng
                        parent_map[ns] = (state, 'compute', v)
                        heapq.heappush(heap, (ng + heuristic(ns), ng, ns))

        # Delete
        for v in range(n):
            if state & (1 << v):
                ns = state & ~(1 << v)
                ng = g + 1
                if ng < best_g.get(ns, float('inf')) and ng <= MAX_OPS:
                    best_g[ns] = ng
                    parent_map[ns] = (state, 'delete', v)
                    heapq.heappush(heap, (ng + heuristic(ns), ng, ns))

    if found_goal is not None:
        ops = []
        state = found_goal
        while state != initial_state:
            prev, op, v = parent_map[state]
            ops.append([op, v])
            state = prev
        ops.reverse()
        return ops

    return _fallback(dag, fast_memory_budget)


def _fallback(dag, fast_memory_budget):
    vertices = dag['vertices']
    outputs = dag['outputs']
    n = len(vertices)
    parents = [tuple(v['parents']) for v in vertices]
    inputs = [i for i in range(n) if len(parents[i]) == 0]

    ops = []
    red = set(inputs)

    for o in outputs:
        _ensure(o, red, parents, ops, fast_memory_budget)

    return ops


def _ensure(v, red, parents, ops, budget):
    if v in red:
        return
    for p in parents[v]:
        _ensure(p, red, parents, ops, budget)
    if len(red) >= budget:
        # Delete something not in red that we don't need
        for x in list(red):
            ops.append(['delete', x])
            red.discard(x)
            break
    red.add(v)
    ops.append(['compute', v])
# EVOLVE-BLOCK-END
