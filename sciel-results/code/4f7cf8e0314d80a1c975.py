"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using layer-based scheduling
    with future-use analysis for smart memory eviction.
    """
    from collections import deque

    # --- Parse the DAG structure robustly ---
    vertices = None
    parents = None
    outputs = None
    inputs = None

    if isinstance(dag, dict):
        for key in ['vertices', 'nodes', 'V', 'vertex_list']:
            if key in dag:
                vertices = list(dag[key])
                break

        for key in ['parents', 'preds', 'in_edges', 'predecessors']:
            if key in dag:
                parents = {v: list(p) for v, p in dag[key].items()}
                break

        if parents is None and 'edges' in dag:
            edges = dag['edges']
            parents = {}
            for e in edges:
                if len(e) == 2:
                    a, b = e
                    parents.setdefault(b, []).append(a)

        for key in ['outputs', 'sinks', 'targets', 'out_vertices']:
            if key in dag:
                outputs = list(dag[key])
                break

        for key in ['inputs', 'sources', 'roots', 'in_vertices']:
            if key in dag:
                inputs = list(dag[key])
                break

    if vertices is None:
        return []

    if parents is None:
        parents = {v: [] for v in vertices}

    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            if p in children:
                children[p].append(v)

    if outputs is None:
        outputs = [v for v in vertices if not children[v]]

    if inputs is None:
        inputs = [v for v in vertices if not parents[v]]

    # --- Topological sort (Kahn's algorithm) ---
    in_degree = {v: len(parents[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) < len(vertices):
        remaining = [v for v in vertices if v not in set(topo_order)]
        topo_order.extend(remaining)

    # --- Determine layers (distance from inputs) ---
    layer = {}
    for v in inputs:
        layer[v] = 0

    for v in topo_order:
        if v in inputs:
            continue
        if parents[v]:
            layer[v] = max(layer[p] for p in parents[v]) + 1
        else:
            layer[v] = 0

    max_layer = max(layer.values()) if layer else 0

    # Group vertices by layer
    layers = {l: [] for l in range(max_layer + 1)}
    for v in vertices:
        if v in layer:
            layers[layer[v]].append(v)

    # --- Pebbling strategy ---
    fast_mem = set(inputs)   # blue vertices in fast memory
    red_mem = set()          # vertices in red memory
    computed = set(inputs)   # vertices computed at least once
    ops = []

    # Count direct future uses (out-degree)
    future_uses = {v: len(children[v]) for v in vertices}

    # Process each layer in order
    for l in range(max_layer + 1):
        if l not in layers or not layers[l]:
            continue

        layer_vertices = layers[l]

        # Compute all vertices in this layer
        for v in layer_vertices:
            # Ensure parents are available in fast memory
            for p in parents[v]:
                if p not in fast_mem and p in red_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    red_mem.discard(p)

            # Compute v
            ops.append(['compute', v])
            computed.add(v)
            fast_mem.add(v)

        # Determine which vertices will be needed in future layers
        future_needed = set()
        for fl in range(l + 1, max_layer + 1):
            if fl in layers:
                for fv in layers[fl]:
                    for p in parents[fv]:
                        future_needed.add(p)

        # Store vertices that won't be needed in future (and aren't outputs/inputs)
        for v in list(fast_mem):
            if v not in outputs and v not in future_needed and v not in inputs:
                if v not in red_mem:
                    ops.append(['store', v])
                    red_mem.add(v)
                fast_mem.discard(v)

        # If fast memory is still over budget, evict low-future-use vertices
        while len(fast_mem) > fast_memory_budget:
            candidates = [x for x in fast_mem if x not in outputs]
            if not candidates:
                break
            to_store = min(candidates, key=lambda x: future_uses[x])
            if to_store not in red_mem:
                ops.append(['store', to_store])
                red_mem.add(to_store)
            fast_mem.discard(to_store)

    # Ensure all outputs are blue at the end
    for o in outputs:
        if o not in fast_mem:
            if o in red_mem:
                ops.append(['load', o])
                fast_mem.add(o)
                red_mem.discard(o)

    return ops
# EVOLVE-BLOCK-END
