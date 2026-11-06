"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    import heapq

    # ---- Parse the DAG ----
    # dag is expected to have:
    #   dag['vertices'] or dag['n']  : number of vertices
    #   dag['parents']               : list of lists, parents[v] = list of predecessor vertices
    #   dag['outputs']               : list of output vertices
    #   dag['inputs'] or dag['inputs'] : list of input vertices (start blue)
    # We try several common key names.

    def _get(dag, *names, default=None):
        for n in names:
            if isinstance(dag, dict) and n in dag:
                return dag[n]
        return default

    parents = _get(dag, 'parents', 'preds', 'predecessors', default=None)
    outputs = _get(dag, 'outputs', 'output', 'sinks', default=None)
    n = _get(dag, 'n', 'n_vertices', 'num_vertices', default=None)

    if parents is None:
        # Maybe dag is a dict mapping vertex -> parents
        if isinstance(dag, dict):
            if 'vertices' in dag:
                parents = dag['vertices'].get('parents') if isinstance(dag['vertices'], dict) else None
            # Try to infer: keys are vertices, values are parent lists
            if parents is None:
                for k, v in dag.items():
                    if isinstance(v, (list, tuple)) and all(isinstance(x, int) for x in v):
                        parents = dag
                        break
    if parents is None:
        raise ValueError("Cannot parse dag parents")

    if n is None:
        n = len(parents)
    if outputs is None:
        # outputs are vertices with no children
        children = [[] for _ in range(n)]
        for v in range(n):
            ps = parents[v] if isinstance(parents, (list, tuple)) else parents.get(v, [])
            for p in ps:
                children[p].append(v)
        outputs = [v for v in range(n) if not children[v]]

    # Normalize parents to list of lists
    if not isinstance(parents, (list, tuple)):
        parents = [list(parents.get(v, [])) for v in range(n)]
    else:
        parents = [list(p) for p in parents]

    # Build children and topological order (Kahn)
    children = [[] for _ in range(n)]
    indeg = [0] * n
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            indeg[v] += 1

    # Topological order
    from collections import deque
    q = deque([v for v in range(n) if indeg[v] == 0])
    topo = []
    indeg2 = indeg[:]
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(topo) != n:
        # cycle? fallback
        topo = list(range(n))

    # Input vertices: those with no parents (or explicitly given)
    inputs = _get(dag, 'inputs', 'input', 'sources', default=None)
    if inputs is None:
        inputs = [v for v in range(n) if not parents[v]]

    input_set = set(inputs)
    output_set = set(outputs)

    # ---- Determine value reuse / criticality ----
    # A vertex's value is "reused" if it has multiple children, or is on a path
    # that benefits from keeping it. We compute a simple heuristic weight:
    #   weight[v] = number of descendants that need it (approx by out-degree chain)
    # We'll compute for each vertex the number of times its value is needed
    # in a naive single-pass (i.e., once per child that consumes it).

    # Compute "demand" = number of children + (1 if output)
    demand = [0] * n
    for v in range(n):
        demand[v] = len(children[v])
        if v in output_set:
            demand[v] += 1

    # ---- Schedule ----
    # Strategy: process vertices in topological order. Keep a fast-memory set
    # of currently-resident (red) values. For each vertex in topo order:
    #   - If vertex is an input: it starts blue; we may need to "load" it into
    #     fast memory if we want to compute its children (but inputs are already
    #     available; loading is only needed if we treat them as values to store).
    #     Actually, inputs start blue (in fast memory conceptually), but the
    #     contract says "every red value consumes one fast-memory slot". Inputs
    #     are blue initially. We may need to load them to make them red for
    #     computation? The operations are load/store/compute/delete.
    #
    # Interpretation:
    #   - load: bring a blue value (from slow memory) into fast memory (red)
    #   - store: write a red value back to slow memory (becomes blue)
    #   - compute: compute a vertex's value from its parents (parents must be red)
    #   - delete: remove a red value from fast memory
    #
    # Inputs start blue. To compute a child, parents must be red. So we must
    # load inputs (and any previously stored values) before computing.
    # Outputs must finish blue: so after computing an output, we store it.
    #
    # Naive schedule: for each vertex in topo order, load all parents not in
    # fast memory, compute, then store/delete parents as needed to fit budget.
    #
    # We'll use a priority-based eviction: keep values with high remaining
    # demand (number of future children that need them) resident.

    # Precompute for each vertex the number of children that appear later in
    # topo order (i.e., future demand).
    pos = {v: i for i, v in enumerate(topo)}
    future_demand = [0] * n
    for v in range(n):
        fd = 0
        for c in children[v]:
            if pos[c] > pos[v]:
                fd += 1
        future_demand[v] = fd

    # Fast memory: dict vertex -> True (resident red)
    fast = set()
    operations = []

    # Inputs are initially blue. We need to load them when we first need them.
    # We'll load a vertex on demand.

    def ensure_resident(v):
        """Ensure vertex v is in fast memory (red). Load if not."""
        if v not in fast:
            operations.append(['load', v])
            fast.add(v)

    def evict_if_needed():
        """Evict low-priority values if fast memory exceeds budget."""
        while len(fast) > fast_memory_budget:
            # Evict the resident value with the lowest future_demand
            # (and not an output that must finish blue soon)
            best_v = None
            best_score = float('inf')
            for u in list(fast):
                # Score: lower future_demand = evict first
                # Also prefer evicting values that are outputs (already stored)
                score = future_demand[u]
                if u in output_set:
                    score -= 1000  # prefer to evict outputs (they're done)
                if score < best_score:
                    best_score = score
                    best_v = u
            if best_v is None:
                break
            operations.append(['delete', best_v])
            fast.discard(best_v)

    # Process in topological order
    # We need to ensure each vertex's value is computed exactly once (or recomputed).
    # computed[v] = True once we've computed v's value.
    computed = [False] * n

    # We'll iterate through topo order. For each vertex, if it's not an input,
    # we compute it. If it's an input, it's already available (blue), but we
    # may need to load it to make it red for its children.

    for v in topo:
        # If v is an input, its value is already in slow memory (blue).
        # We don't "compute" it; we load it when needed.
        if v in input_set:
            # Don't compute; just make sure it's available when children need it.
            # We'll load it lazily. For now, mark it as "available" (blue).
            continue

        # For non-input vertices, compute v.
        # First, ensure all parents are resident (red).
        for p in parents[v]:
            ensure_resident(p)
            evict_if_needed()

        # Now compute v.
        operations.append(['compute', v])
        computed[v] = True
        fast.add(v)
        evict_if_needed()

        # If v is an output, it must finish blue: store it.
        if v in output_set:
            operations.append(['store', v])
            fast.discard(v)

    # After processing, ensure all outputs are blue (stored).
    for v in outputs:
        if v in fast:
            operations.append(['store', v])
            fast.discard(v)

    # Also, we may have leftover red values; delete them to be clean.
    for v in list(fast):
        operations.append(['delete', v])
        fast.discard(v)

    return operations
# EVOLVE-BLOCK-END
