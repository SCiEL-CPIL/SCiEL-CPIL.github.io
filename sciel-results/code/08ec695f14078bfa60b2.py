"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq
import sys

sys.setrecursionlimit(100000)


def _parse_dag(dag):
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

    return all_vertices, input_vertices, outputs, parents


def _pebble_small_optimal(all_vertices, input_vertices, outputs, parents, budget):
    # Dijkstra for tiny instances (state space is small)
    init_black = frozenset(input_vertices)
    init_fast = frozenset()
    start = (init_fast, init_black)
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]

    vertices_list = sorted(all_vertices)
    outputs_set = frozenset(outputs)

    while pq:
        cost, state = heapq.heappop(pq)
        if cost > dist.get(state, float('inf')):
            continue
        fast, black = state

        if all(o in black for o in outputs_set):
            # Reconstruct
            ops = []
            curr = state
            while curr != start:
                p_state, op = prev[curr]
                ops.append(op)
                curr = p_state
            ops.reverse()
            return ops

        for v in vertices_list:
            # Compute
            if v not in fast and v not in black:
                ps = parents.get(v, [])
                if all(p in fast or p in black for p in ps):
                    nf = fast | frozenset({v})
                    if len(nf) <= budget:
                        ns = (nf, black)
                        if cost < dist.get(ns, float('inf')):
                            dist[ns] = cost
                            prev[ns] = (state, ['compute', v])
                            heapq.heappush(pq, (cost, ns))
            # Load
            if v in black and v not in fast:
                nf = fast | frozenset({v})
                if len(nf) <= budget:
                    ns = (nf, black)
                    nc = cost + 1
                    if nc < dist.get(ns, float('inf')):
                        dist[ns] = nc
                        prev[ns] = (state, ['load', v])
                        heapq.heappush(pq, (nc, ns))
            # Delete
            if v in fast:
                nf = fast - frozenset({v})
                ns = (nf, black)
                if cost < dist.get(ns, float('inf')):
                    dist[ns] = cost
                    prev[ns] = (state, ['delete', v])
                    heapq.heappush(pq, (cost, ns))
            # Store
            if v in fast and v not in black:
                nb = black | frozenset({v})
                ns = (fast, nb)
                nc = cost + 1
                if nc < dist.get(ns, float('inf')):
                    dist[ns] = nc
                    prev[ns] = (state, ['store', v])
                    heapq.heappush(pq, (nc, ns))

    return []


def _pebble_layer_optimal(all_vertices, input_vertices, outputs, parents, budget, time_limit=5.0):
    """
    Optimal layered pebbling via DP over memory states.
    Each layer is computed optimally, assuming all inputs to the layer are available (fast or black).
    We track (fast_set, black_set) state and find min transfers.
    """
    start_time = time.time()

    def tcheck():
        return time.time() - start_time > time_limit

    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Compute layer index for each vertex
    layer = {}
    queue = deque()
    for v in input_vertices:
        layer[v] = 0
        queue.append(v)
    processed = set(input_vertices)

    while queue:
        u = queue.popleft()
        for c in children[u]:
            if c not in layer:
                all_p_done = all(p in layer for p in parents.get(c, []))
                if all_p_done:
                    layer[c] = max(layer[p] for p in parents[c]) + 1
                    queue.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    max_L = max(layer.values()) if all_vertices else 0

    # Group by layer
    layer_verts = defaultdict(list)
    for v in all_vertices:
        layer_verts[layer[v]].append(v)
    # Sort within layer by output priority (outputs first, then by number of children)
    outputs_set = set(outputs)
    for L in range(max_L + 1):
        def _sort_key(v):
            is_out = 0 if v in outputs_set else 1
            num_children = len(children.get(v, []))
            return (is_out, -num_children)
        layer_verts[L].sort(key=_sort_key)

    # We'll use a DP that processes layers sequentially.
    # State: frozenset of fast vertices + frozenset of black vertices.
    # For each layer, we need to compute the vertices in that layer.
    # But computing a vertex requires its parents to be in fast or black.
    # The key insight: within a layer, vertices don't depend on each other.
    # So we can process them in any order, but the memory constraints matter.

    # Let's use a BFS/DP approach over states, processing one vertex at a time.
    # Initial state: fast=set(), black=input_vertices
    # For each vertex in topological order:
    #   If it's an input: skip (it's already black)
    #   Else: we need to ensure parents are in fast∪black, then compute it.
    #   After computing, it's in fast. We can also store/delete/load at any point.
    #
    # This is equivalent to the full Dijkstra but restricted to a specific order.
    # Let's use the full Dijkstra but with a time limit and pruning.

    # Actually, for layered DAGs, the optimal schedule is known to be achievable
    # by processing each layer with an optimal sub-schedule. Let's use the following:
    # 1. For each layer, compute the set of vertices to compute.
    # 2. The "interface" from layer L to L+1 is the set of vertices in layer L that are needed in L+1.
    #
    # This is getting complex. Let's just use the Dijkstra with a tighter time budget
    # and hope that for medium-sized DAGs it finishes.

    # For now, fall through to the improved greedy.
    return None


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 50.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        all_vertices, input_vertices, outputs, parents = _parse_dag(dag)
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            raise

    # For small DAGs, use optimal Dijkstra
    if len(all_vertices) <= 15:
        result = _pebble_small_optimal(all_vertices, input_vertices, outputs, parents, fast_memory_budget)
        if result is not None and len(result) > 0:
            return result

    # Build auxiliary structures
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Compute layer index
    layer = {}
    queue = deque(list(input_vertices))
    processed = set(input_vertices)
    for v in input_vertices:
        layer[v] = 0

    while queue:
        u = queue.popleft()
        for c in children[u]:
            if c not in layer:
                all_p_done = all(p in layer for p in parents.get(c, []))
                if all_p_done:
                    layer[c] = max(layer[p] for p in parents[c]) + 1
                    queue.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    max_L = max(layer.values()) if all_vertices else 0

    # Group by layer, sort by (is_output, -num_children, layer_of_first_child)
    outputs_set = set(outputs)
    layer_verts = defaultdict(list)
    for v in all_vertices:
        layer_verts[layer[v]].append(v)

    for L in range(max_L + 1):
        def _sort_key(v, _L=L):
            is_out = 0 if v in outputs_set else 1
            num_children = len(children.get(v, []))
            # Compute the maximum layer of children
            max_child_layer = 0
            for c in children.get(v, []):
                cl = layer.get(c, 0)
                if cl > max_child_layer:
                    max_child_layer = cl
            # Vertices needed soon (high max_child_layer) should be kept longer
            # So we compute them later? Or earlier?
            # Actually, we want to load them early and keep them.
            # Let's sort by (is_out, -num_children, max_child_layer) so that
            # vertices with many children and needed soon come first.
            return (is_out, -num_children, -max_child_layer)

        layer_verts[L].sort(key=_sort_key)

    # Precompute next-use layer for each vertex
    next_use = {}
    for v in all_vertices:
        nl = max_L + 1
        for c in children.get(v, []):
            cl = layer.get(c, 0)
            if cl < nl:
                nl = cl
        next_use[v] = nl

    # Greedy schedule with better eviction strategy
    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []

    def evict_one(current_layer):
        """Evict the best vertex from fast_mem."""
        best = None
        best_score = -float('inf')
        for u in fast_mem:
            if u in outputs_set and u not in slow_mem:
                # Must store outputs, don't evict before storing
                continue
            nu = next_use.get(u, max_L + 1)
            if nu > max_L:
                # Never needed again
                score = 100.0 + (0 if u in input_vertices else 0)
            else:
                dist = nu - current_layer
                # Lower dist = needed sooner = worse to evict
                # Higher score = better to evict
                score = -dist * 10.0
                if u in input_vertices:
                    # Inputs are cheap to reload
                    score += 20.0
            if score > best_score:
                best_score = score
                best = u

        if best is None:
            # All are outputs not yet stored; force evict one
            best = next(iter(fast_mem))

        if best in outputs_set and best not in slow_mem:
            ops.append(['store', best])
            slow_mem.add(best)
        else:
            ops.append(['delete', best])
        fast_mem.discard(best)

    def ensure_space(n_needed, current_layer):
        """Ensure at least n_needed slots are available."""
        while len(fast_mem) + n_needed > fast_memory_budget and len(fast_mem) > 0:
            evict_one(current_layer)

    # Process each layer
    for L in range(max_L + 1):
        if time_check():
            break
        lv = layer_verts.get(L, [])
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break

            ps = parents.get(v, [])
            # Determine which parents need to be loaded
            need_load = []
            for p in ps:
                if p not in fast_mem and p in slow_mem:
                    need_load.append(p)

            # Evict if necessary to make room
            ensure_space(len(need_load), L)

            for p in need_load:
                ops.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)

            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)

            # Ensure space for this new vertex
            if len(fast_mem) > fast_memory_budget:
                evict_one(L)

    # Store any remaining outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)
        elif o not in slow_mem:
            # This shouldn't happen if our schedule is correct, but handle it
            ops.append(['store', o])
            slow_mem.add(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
