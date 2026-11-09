"""DAG pebbling: communication-minimizing schedule with deadline guard."""
import time

_DEADLINE = None  # set on first call; 60s wall-clock cap with margin
_TIME_LIMIT = 58.0  # safety margin under 60s


def _now():
    return time.time()


def _deadline_hit():
    return _DEADLINE is not None and (_now() - _DEADLINE) >= _TIME_LIMIT


def _get_parents(dag):
    if isinstance(dag, dict):
        p = dag.get('parents', dag.get('edges', None))
        if p is None:
            # maybe dag is {vertex: [parents]}
            return dag
        return p
    return getattr(dag, 'parents', None) or getattr(dag, 'edges', None)


def _get_outputs(dag):
    if isinstance(dag, dict):
        o = dag.get('outputs', dag.get('sinks', None))
        if o is not None:
            return o
    return getattr(dag, 'outputs', None) or getattr(dag, 'sinks', None)


def _get_all_vertices(dag):
    if isinstance(dag, dict):
        v = dag.get('vertices', dag.get('nodes', None))
        if v is not None:
            return v
    return getattr(dag, 'vertices', None) or getattr(dag, 'nodes', None)


def _get_inputs(dag):
    if isinstance(dag, dict):
        i = dag.get('inputs', dag.get('sources', None))
        if i is not None:
            return i
    return getattr(dag, 'inputs', None) or getattr(dag, 'sources', None)


def _naive_schedule(dag, parents, outputs, all_vertices, inputs, budget):
    """Topological order: load each input once, compute each non-input in topo order."""
    ops = []
    # Determine topological order: inputs first, then others in given order.
    if all_vertices is not None:
        verts = list(all_vertices)
    else:
        verts = list(parents.keys()) if parents else []
    # topological: assume parents already ordered; do a simple topological sort
    indeg = {}
    children = {}
    for v in verts:
        ps = parents.get(v, []) if isinstance(parents, dict) else []
        indeg[v] = len(ps)
        for p in ps:
            children.setdefault(p, []).append(v)
    # Kahn
    from collections import deque
    q = deque()
    for v in verts:
        if indeg[v] == 0:
            q.append(v)
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children.get(v, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != len(verts):
        topo = verts  # fallback
    input_set = set(inputs) if inputs else set()
    for v in topo:
        if v in input_set:
            ops.append(['load', v])
        else:
            ops.append(['compute', v])
    # ensure outputs finish blue: they are computed (non-input) or loaded (input).
    # If an output is an input, it's already blue.
    return ops


def _count_transfers(ops):
    return sum(1 for op, v in ops if op in ('load', 'store'))


def pebble(dag, fast_memory_budget):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = _now()

    parents = _get_parents(dag)
    outputs = _get_outputs(dag)
    all_vertices = _get_all_vertices(dag)
    inputs = _get_inputs(dag)

    # Fallbacks
    if parents is None:
        parents = {}
    if outputs is None:
        outputs = []
    if inputs is None:
        inputs = []
    if all_vertices is None:
        all_vertices = list(parents.keys()) if parents else []

    input_set = set(inputs)
    output_set = set(outputs)
    all_set = set(all_vertices)

    # Naive baseline
    naive_ops = _naive_schedule(dag, parents, outputs, all_vertices, inputs, fast_memory_budget)
    naive_transfers = _count_transfers(naive_ops)

    # Try to build a better schedule: keep intermediate results in fast memory
    # to avoid recomputation when a vertex has multiple children.
    # Strategy: process in topological order; keep a value in fast memory if
    # it will be needed again (has multiple children) and budget allows.
    # Count children for each vertex.
    child_count = {}
    for v in all_vertices:
        ps = parents.get(v, []) if isinstance(parents, dict) else []
        for p in ps:
            child_count[p] = child_count.get(p, 0) + 1

    # Build a schedule that minimizes transfers:
    # - Each input: load once (if needed) -> 1 transfer
    # - Each non-input: compute (no transfer) but its result must be stored
    #   if it will be consumed later (child not yet processed) OR if it's an output.
    # Key insight: in a topological schedule, a value computed at vertex v is
    # available for all its children (which come later). If we keep it in fast
    # memory, we avoid a store+load for the "next" consumer. But fast memory is
    # limited.
    #
    # Simple improvement: if budget >= number of distinct "live" values needed,
    # we can keep everything and only do loads for inputs and stores for
    # outputs. Transfers = len(inputs) + len(outputs).
    #
    # General: we want to minimize stores. A store is needed when a value must
    # persist beyond the point where its last child consumes it, OR when it's
    # an output. Actually in topo order, a value is consumed by all children
    # as they're processed. If we keep it in fast memory until the last child,
    # no store is needed unless budget forces eviction.
    #
    # Let's simulate: process in topo order. Maintain fast_memory set.
    # When computing v (non-input): result goes to fast_memory.
    # When an input is loaded: goes to fast_memory.
    # When a vertex's last child is processed, the value is no longer needed ->
    # can be removed from fast_memory (freeing a slot).
    # If fast_memory is full when we need to add a new value, we must store
    # (evict) some value. To minimize stores, evict the value whose last
    # use is soonest (LRFU-like). But since we're in topo order, the value
    # with the earliest "last child" is the one we should evict.

    # Determine topological order
    from collections import deque
    verts = list(all_vertices)
    indeg = {}
    children = {}
    for v in verts:
        ps = parents.get(v, []) if isinstance(parents, dict) else []
        indeg[v] = len(ps)
        for p in ps:
            children.setdefault(p, []).append(v)
    q = deque()
    for v in verts:
        if indeg[v] == 0:
            q.append(v)
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children.get(v, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != len(verts):
        topo = verts

    # For each vertex, find the position (index in topo) of its last child.
    # A value at vertex v is needed until the last child of v is processed.
    # If v has no children, it's only needed if it's an output (must be blue at end).
    last_child_pos = {}
    for v in verts:
        cs = children.get(v, [])
        if cs:
            last_child_pos[v] = max(topo.index(c) for c in cs)
        else:
            # no children: value needed only if output
            last_child_pos[v] = len(topo) - 1 if v in output_set else -1

    # Simulate the schedule
    fast_mem = set()
    ops = []
    budget = max(0, int(fast_memory_budget)) if fast_memory_budget is not None else 0

    # We need to track: when we compute v, its result enters fast_mem.
    # When we process the last child of v, v's value can leave fast_mem.
    # If fast_mem would exceed budget, evict the value with smallest last_child_pos
    # (i.e., the one needed soonest / already past).

    # Actually, let's think differently. In topo order:
    # - At each vertex v, before computing v, all its parents must be in fast_mem
    #   (or loaded). If a parent is not in fast_mem, we must load it (transfer).
    # - After computing v, v's result is in fast_mem.
    # - We can evict values from fast_mem when budget is exceeded.
    # - At the end, outputs must be in fast_mem (blue).

    # Let's simulate more carefully:
    # State: fast_mem (set of vertices whose values are in fast memory)
    # For each v in topo:
    #   1. Ensure all parents of v are in fast_mem. If not, load them.
    #      (But in topo order, parents come before v, so they were computed/loaded
    #      earlier. They might have been evicted if budget was tight.)
    #   2. If v is an input: load v (transfer), add to fast_mem.
    #      Else: compute v (no transfer), add to fast_mem.
    #   3. If fast_mem size > budget: evict some value (store, transfer).
    #      Evict the one with the smallest last_child_pos that is still >= current
    #      position, or the one whose last use is soonest.
    #   4. If v's last_child_pos < current position (i.e., v's last child was
    #      already processed), v's value is no longer needed -> remove from
    #      fast_mem (no transfer, just free a slot).

    # Wait, this is getting complicated. Let me use a simpler heuristic:
    # If budget >= max "width" of the DAG (max number of simultaneously live
    # values), then no evictions needed, and transfers = inputs + outputs.
    # Otherwise, we need some evictions.

    # Compute max live values: at each position i in topo, how many values
    # are "live" (their last_child_pos >= i)?
    # A value at vertex v is live from position i (when v is computed) to
    # last_child_pos[v].

    # Let's just simulate with a priority queue for eviction.
    import heapq

    fast_mem = set()
    ops = []
    # priority queue: (last_child_pos, vertex) for eviction
    evict_pq = []
    in_fast = {}  # vertex -> last_child_pos

    for i, v in enumerate(topo):
        # Ensure parents are in fast_mem
        ps = parents.get(v, []) if isinstance(parents, dict) else []
        for p in ps:
            if p not in fast_mem:
                # Need to load p
                ops.append(['load', p])
                fast_mem.add(p)
                in_fast[p] = last_child_pos.get(p, -1)
                heapq.heappush(evict_pq, (in_fast[p], p))
                # Evict if over budget
                while len(fast_mem) > budget:
                    lcp, ev = heapq.heappop(evict_pq)
                    if ev not in fast_mem:
                        continue
                    ops.append(['store', ev])
                    fast_mem.discard(ev)
                    del in_fast[ev]

        # Process v
        if v in input_set:
            if v not in fast_mem:
                ops.append(['load', v])
                fast_mem.add(v)
                in_fast[v] = last_child_pos.get(v, -1)
                heapq.heappush(evict_pq, (in_fast[v], v))
                while len(fast_mem) > budget:
                    lcp, ev = heapq.heappop(evict_pq)
                    if ev not in fast_mem:
                        continue
                    ops.append(['store', ev])
                    fast_mem.discard(ev)
                    del in_fast[ev]
        else:
            # compute v
            ops.append(['compute', v])
            fast_mem.add(v)
            in_fast[v] = last_child_pos.get(v, -1)
            heapq.heappush(evict_pq, (in_fast[v], v))
            while len(fast_mem) > budget:
                lcp, ev = heapq.heappop(evict_pq)
                if ev not in fast_mem:
                    continue
                ops.append(['store', ev])
                fast_mem.discard(ev)
                del in_fast[ev]

        # Free v's value if no longer needed
        if last_child_pos.get(v, -1) < i:
            if v in fast_mem:
                fast_mem.discard(v)
                if v in in_fast:
                    del in_fast[v]

    # Ensure outputs are in fast_mem (blue) at the end
    for v in outputs:
        if v not in fast_mem:
            ops.append(['load', v])
            fast_mem.add(v)
            in_fast[v] = last_child_pos.get(v, -1)
            heapq.heappush(evict_pq, (in_fast[v], v))
            while len(fast_mem) > budget:
                lcp, ev = heapq.heappop(evict_pq)
                if ev not in fast_mem:
                    continue
                ops.append(['store', ev])
                fast_mem.discard(ev)
                del in_fast[ev]

    # Count transfers for our schedule
    our_transfers = _count_transfers(ops)

    # Choose the better schedule
    if our_transfers <= naive_transfers:
        return ops
    else:
        return naive_ops
