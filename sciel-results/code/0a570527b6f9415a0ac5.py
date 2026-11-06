"""Communication-minimizing DAG pebbling with a bounded candidate interface."""

# EVOLVE-BLOCK-START
import time as _time

def pebble(dag, fast_memory_budget):
    """Return a list of [operation, vertex] operations.

    dag is a dict with:
      - 'n': number of vertices
      - 'topo': list of vertices in topological order
      - 'parents': list of lists; parents[v] = list of parent vertex indices
      - 'outputs': list of output vertex indices
    """
    t0 = _time.time()
    DEADLINE = 70.0

    n = dag['n']
    topo = dag['topo']
    parents = dag['parents']
    outputs = dag['outputs']

    # Build children map
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)

    is_input = [len(parents[v]) == 0 for v in range(n)]

    # Count how many times each vertex is used as a parent
    reuse = [0] * n
    for v in range(n):
        for p in parents[v]:
            reuse[p] += 1

    # Next use in topological order
    next_use = [float('inf')] * n
    for v in range(n):
        for c in children[v]:
            if c > v:
                next_use[v] = min(next_use[v], c)

    ops = []
    fast_mem = set()  # vertices currently in fast memory

    def evict_one():
        """Evict the least important value from fast memory."""
        if not fast_mem:
            return None
        worst = None
        worst_p = float('inf')
        for v in fast_mem:
            # Priority: higher reuse and sooner next_use = keep longer
            p = reuse[v] * 10000 - next_use[v]
            if p < worst_p:
                worst_p = p
                worst = v
        if worst is not None:
            fast_mem.discard(worst)
            ops.append(['delete', worst])
        return worst

    def ensure_red(v):
        """Ensure vertex v is in fast memory (red)."""
        if v in fast_mem:
            return
        if is_input[v]:
            # Need to load v
            if len(fast_mem) >= fast_memory_budget:
                evict_one()
            ops.append(['load', v])
            fast_mem.add(v)
        else:
            # Need to compute v: ensure all parents are red
            for p in parents[v]:
                ensure_red(p)
            if len(fast_mem) >= fast_memory_budget:
                evict_one()
            ops.append(['compute', v])
            fast_mem.add(v)

    # Process all vertices in topo order
    for v in topo:
        ensure_red(v)

    # Ensure all outputs are blue (stored to slow memory)
    for v in outputs:
        if v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)

    if _time.time() - t0 > DEADLINE:
        return ops

    return ops
# EVOLVE-BLOCK-END
