"""DAG pebbling: communication-minimizing schedule with deadline guard."""
import time

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 90.0
    MARGIN = 1.5

    # ---- Normalize the DAG structure ----
    nodes = list(dag.get("nodes", dag.get("vertices", [])))
    if not nodes:
        return []

    # Build adjacency info
    children = {v: [] for v in nodes}
    parents = {v: [] for v in nodes}
    for v in nodes:
        par = dag.get("parents", {}).get(v, []) if isinstance(dag.get("parents", {}), dict) else []
        if not par:
            par = dag.get("parents_of", {}).get(v, []) if isinstance(dag.get("parents_of", {}), dict) else []
        for p in par:
            parents[v].append(p)
            children[p].append(v)

    # Determine input and output vertices
    inputs = [v for v in nodes if len(parents[v]) == 0]
    outputs = [v for v in nodes if len(children[v]) == 0]

    # Topological order
    from collections import deque
    indeg = {v: len(parents[v]) for v in nodes}
    q = deque([v for v in nodes if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for c in children[u]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != len(nodes):
        topo = list(nodes)  # fallback

    # ---- Build a layer-based schedule ----
    # Each node's "depth" = longest path from an input
    depth = {v: 0 for v in nodes}
    for v in topo:
        for c in children[v]:
            if depth[v] + 1 > depth[c]:
                depth[c] = depth[v] + 1

    # Group by depth
    max_depth = max(depth.values()) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in topo:
        layers[depth[v]].append(v)

    # Fast memory budget: number of red slots available
    B = max(1, int(fast_memory_budget)) if fast_memory_budget else 1

    ops = []
    # Track which vertices are currently "red" (in fast memory)
    red = set()
    # Inputs start blue (in fast memory) - they're already loaded
    for v in inputs:
        red.add(v)

    def is_red(v):
        return v in red

    def make_red(v):
        red.add(v)

    def make_blue(v):
        red.discard(v)

    def spill_oldest(need):
        """Evict oldest red values to make room. Returns list of evicted."""
        # Priority: evict values with smallest remaining use (fewest children not yet computed)
        evicted = []
        while len(red) >= B and need:
            # Find red value with fewest uncomputed children
            best = None
            best_score = float('inf')
            for r in list(red):
                # count children not yet processed
                remaining = sum(1 for c in children[r] if not is_red(c))
                score = remaining
                if score < best_score:
                    best_score = score
                    best = r
            if best is None:
                break
            if best in inputs:
                # inputs can be reloaded, lower priority
                continue
            make_blue(best)
            ops.append(["store", best])
            evicted.append(best)
            need -= 1
        return evicted

    # Process layer by layer
    for d in range(max_depth + 1):
        if time.time() - t0 > DEADLINE - MARGIN:
            # Safety: finalize outputs
            for v in outputs:
                if not is_red(v):
                    ops.append(["load", v])
                    make_red(v)
            return ops

        for v in layers[d]:
            if time.time() - t0 > DEADLINE - MARGIN:
                for v2 in outputs:
                    if not is_red(v2):
                        ops.append(["load", v2])
                        make_red(v2)
                return ops

            # Ensure parents are red
            for p in parents[v]:
                if not is_red(p):
                    # Need to load parent
                    spill_oldest(1)
                    ops.append(["load", p])
                    make_red(p)

            # Compute v
            if len(parents[v]) > 0:
                ops.append(["compute", v])
            else:
                # Input node, already red
                pass
            make_red(v)

            # Free parents if no more children need them
            for p in parents[v]:
                remaining = sum(1 for c in children[p] if c != v and not is_red(c))
                if remaining == 0 and p not in inputs:
                    # Check if p is an output
                    if p not in outputs:
                        make_blue(p)
                        ops.append(["delete", p])

    # Finalize: ensure all outputs are red (blue)
    for v in outputs:
        if not is_red(v):
            # Need to load
            spill_oldest(1)
            ops.append(["load", v])
            make_red(v)

    # Limit operations
    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
