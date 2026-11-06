"""Rewritten pebbling scheduler: cost-based greedy with recomputation and deadline guard."""
import time

_DEADLINE_FRAC = 0.85  # safety margin before 90s kill

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys like 'vertices' (list), 'parents' (dict v -> list of parent v's),
         'outputs' (list). Inputs = vertices with no parents (start blue).
    fast_memory_budget: int, number of red values allowed in fast memory.
    Returns list of [operation, vertex] with ops in {load, store, compute, delete}.
    """
    start = time.time()
    deadline = start + 90.0 * _DEADLINE_FRAC

    # ---- Parse DAG defensively (accept several common shapes) ----
    vertices = list(dag.get('vertices', []))
    parents = dict(dag.get('parents', {}))
    outputs = list(dag.get('outputs', []))
    if not vertices:
        # fallback: infer from parents keys/values
        vs = set()
        for v, ps in parents.items():
            vs.add(v)
            vs.update(ps)
        vertices = list(vs)
        outputs = list(dag.get('outputs', vertices))

    # children map
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(v)

    inputs = [v for v in vertices if not parents.get(v)]
    # topological order (Kahn)
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    from collections import deque
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for c in children.get(u, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != len(vertices):
        topo = vertices  # fallback

    # ---- Naive schedule baseline (for reference / fallback) ----
    def naive_schedule():
        ops = []
        for v in inputs:
            ops.append(['load', v])
        for v in topo:
            if v in inputs:
                continue
            # ensure parents present (recompute if missing)
            for p in parents.get(v, []):
                if not _present(p, mem_state, fast_memory_budget, ops):
                    pass
            ops.append(['compute', v])
        return ops

    # We'll build a real schedule. State tracking:
    # mem_state: set of vertices currently in fast memory (red)
    # We must start with inputs blue (in memory) and end with outputs blue (in memory).

    # Cost model: each load/store = 1 transfer. compute = 0 transfer but uses memory.
    # Goal: minimize transfers = #load + #store.
    #
    # Strategy: process in topological order. Keep a working set in fast memory.
    # When memory is full and we need a new value, evict (store) the value with
    # the least "future usefulness" (fewest remaining children that still need it),
    # OR recompute if that's cheaper.

    # Precompute: for each vertex, number of children (fanout) and a topological position.
    pos = {v: i for i, v in enumerate(topo)}
    fanout = {v: len(children.get(v, [])) for v in vertices}
    # remaining_children: as we process, decrement.

    # We'll simulate.
    in_mem = set()
    computed = set()  # vertices whose value has been computed at least once
    ops = []

    def remaining_children(v):
        # count children not yet computed
        cnt = 0
        for c in children.get(v, []):
            if c not in computed:
                cnt += 1
        return cnt

    # Start: load all inputs (they start blue, but to be safe we treat them as needing to be in mem)
    # Actually inputs "start blue" meaning they are available in fast memory initially? 
    # The contract: "Inputs start blue, all outputs must finish blue".
    # Interpretation: inputs are initially present (blue) in fast memory. So no load needed for inputs.
    for v in inputs:
        in_mem.add(v)
        computed.add(v)

    # Process each vertex in topo order (skip inputs already done)
    for v in topo:
        if time.time() > deadline:
            break
        if v in inputs:
            continue
        ps = parents.get(v, [])
        # Ensure all parents are in fast memory
        for p in ps:
            if p not in in_mem:
                # Need to bring p into memory.
                if len(in_mem) >= fast_memory_budget:
                    # Evict something or recompute.
                    # Find eviction candidate: among in_mem, the one with lowest remaining usefulness
                    # and that is not an output (outputs must finish blue).
                    candidates = [x for x in in_mem if x not in outputs]
                    if not candidates:
                        # All in memory are outputs; can't evict. Must recompute v's missing parent?
                        # But p is missing and memory full of outputs -> we must have evicted p earlier.
                        # Recompute p: but p's parents may not be in mem. This gets complex.
                        # Fallback: recompute p in-place if possible, else store an output (not allowed to end not blue,
                        # but we can store then reload later).
                        # For safety, just store the output with least remaining children (non-output none), 
                        # we'll reload at end.
                        candidates = list(in_mem)
                    # Score: prefer evicting low remaining_children, and not inputs.
                    best = min(candidates, key=lambda x: (remaining_children(x), pos.get(x, 0)))
                    ops.append(['store', best])
                    in_mem.discard(best)
                # Now load or recompute p
                if p in computed:
                    # value was computed before, just reload
                    ops.append(['load', p])
                else:
                    # never computed: recompute p (requires p's parents in mem)
                    # recursively ensure p's parents, compute p
                    _ensure_and_compute(p, in_mem, computed, ops, fast_memory_budget, parents, children, outputs, pos, remaining_children, deadline)
                in_mem.add(p)
        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        in_mem.add(v)
        # If memory over budget (shouldn't be, but ensure), evict
        while len(in_mem) > fast_memory_budget:
            candidates = [x for x in in_mem if x not in outputs]
            if not candidates:
                candidates = list(in_mem)
            best = min(candidates, key=lambda x: (remaining_children(x), pos.get(x, 0)))
            ops.append(['store', best])
            in_mem.discard(best)

    # End: ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_mem:
            if len(in_mem) >= fast_memory_budget:
                candidates = [x for x in in_mem if x not in outputs]
                if candidates:
                    best = min(candidates, key=lambda x: (remaining_children(x), pos.get(x, 0)))
                    ops.append(['store', best])
                    in_mem.discard(best)
            if v in computed:
                ops.append(['load', v])
            else:
                _ensure_and_compute(v, in_mem, computed, ops, fast_memory_budget, parents, children, outputs, pos, remaining_children, deadline)
            in_mem.add(v)

    # Cap operations
    if len(ops) > 20000:
        ops = ops[:20000]

    return ops


def _present(v, mem_state, budget, ops):
    return v in mem_state


def _ensure_and_compute(v, in_mem, computed, ops, budget, parents, children, outputs, pos, remaining_children, deadline):
    """Ensure v is in in_mem by recomputing if necessary. Modifies in_mem, computed, ops."""
    if v in in_mem:
        return
    if v in computed:
        # just need to reload
        if len(in_mem) >= budget:
            candidates = [x for x in in_mem if x not in outputs]
            if candidates:
                best = min(candidates, key=lambda x: (remaining_children(x), pos.get(x, 0)))
                ops.append(['store', best])
                in_mem.discard(best)
        ops.append(['load', v])
        in_mem.add(v)
        return
    # Recompute v: need all parents in mem
    for p in parents.get(v, []):
        _ensure_and_compute(p, in_mem, computed, ops, budget, parents, children, outputs, pos, remaining_children, deadline)
    ops.append(['compute', v])
    computed.add(v)
    in_mem.add(v)
    while len(in_mem) > budget:
        candidates = [x for x in in_mem if x not in outputs]
        if not candidates:
            candidates = list(in_mem)
        best = min(candidates, key=lambda x: (remaining_children(x), pos.get(x, 0)))
        ops.append(['store', best])
        in_mem.discard(best)
