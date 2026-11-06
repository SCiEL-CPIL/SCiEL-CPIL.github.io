"""Rewritten DAG pebbling: memory-aware topological schedule with recompute-on-eviction."""

import time
from collections import deque

# EVOLVE-BLOCK-START

_DEADLINE_MARGIN = 0.5  # seconds before the 90s hard limit to stop and return best


def _parse_dag(dag):
    if isinstance(dag, dict):
        n = dag.get('n', dag.get('num_vertices', dag.get('num_nodes', None)))
        parents = dag.get('parents', dag.get('parent_list', []))
        outputs = dag.get('outputs', dag.get('output_vertices', dag.get('sinks', [])))
        if n is None:
            n = len(parents)
        parents = [list(p) for p in parents]
        while len(parents) < n:
            parents.append([])
        parents = parents[:n]
        outputs = list(outputs)
    else:
        raise ValueError("dag must be a dict")
    inputs = [i for i in range(n) if len(parents[i]) == 0]
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    return n, parents, children, inputs, outputs


def pebble(dag, fast_memory_budget):
    t0 = time.time()

    def deadline_ok():
        return (time.time() - t0) < (90.0 - _DEADLINE_MARGIN)

    n, parents, children, inputs, outputs = _parse_dag(dag)
    is_output = [False] * n
    for o in outputs:
        is_output[o] = True

    # Topological order
    indeg = [len(parents[v]) for v in range(n)]
    q = deque([v for v in range(n) if indeg[v] == 0])
    topo = []
    indeg2 = indeg[:]
    while q:
        u = q.popleft()
        topo.append(u)
        for c in children[u]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(topo) != n:
        topo = list(range(n))

    # remaining_children[v] = number of children of v not yet computed
    remaining_children = [len(children[v]) for v in range(n)]

    red = set()
    ops = []
    budget = max(0, int(fast_memory_budget))

    def emit(op, v):
        ops.append([op, v])

    def load(v):
        emit('load', v)
        red.add(v)

    def store(v):
        emit('store', v)
        red.discard(v)

    def delete(v):
        emit('delete', v)
        red.discard(v)

    def compute(v):
        emit('compute', v)
        red.add(v)

    def ensure_red(v, depth=0):
        if v in red:
            return
        if depth > n + 2:
            return
        if len(parents[v]) == 0:
            load(v)
            return
        for p in parents[v]:
            ensure_red(p, depth + 1)
        compute(v)

    def evict_to_budget():
        # Free red vertices until len(red) <= budget.
        # Priority: non-output with remaining_children==0 first (safe, no recompute needed
        # for their own value), then non-output, then outputs (last resort).
        while len(red) > budget:
            # safe candidates
            cands = [x for x in red if not is_output[x] and remaining_children[x] == 0]
            if not cands:
                cands = [x for x in red if not is_output[x]]
            if not cands:
                cands = [x for x in red]
            if not cands:
                break
            # evict the one with the largest topo index (latest, least likely needed soon)
            cands.sort(key=lambda x: topo.index(x), reverse=True)
            delete(cands[0])

    # Forward pass
    for v in topo:
        if not deadline_ok():
            break
        for p in parents[v]:
            if p not in red:
                ensure_red(p)
        compute(v)
        # decrement remaining_children of parents
        for p in parents[v]:
            remaining_children[p] -= 1
        # if a parent is an output and just lost its last child, store it now
        for p in parents[v]:
            if is_output[p] and remaining_children[p] == 0 and p in red:
                store(p)
        evict_to_budget()

    # Store any remaining outputs that are red
    for o in outputs:
        if not deadline_ok():
            break
        if o in red:
            store(o)
        else:
            # recompute then store (should be rare)
            ensure_red(o)
            store(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

# EVOLVE-BLOCK-END
