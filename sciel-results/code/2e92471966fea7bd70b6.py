# EVOLVE-BLOCK-START
import time


def _extract(dag):
    parents = None
    outputs = None
    n = None
    if isinstance(dag, dict):
        for k in ("parents", "preds", "in_edges", "ancestors"):
            if k in dag:
                parents = dag[k]; break
        for k in ("outputs", "sinks", "sinks_list", "output_vertices"):
            if k in dag:
                outputs = dag[k]; break
        for k in ("n", "num_vertices", "num_nodes", "size"):
            if k in dag:
                n = int(dag[k]); break
    else:
        for attr in ("parents", "preds", "in_edges", "ancestors"):
            if hasattr(dag, attr):
                parents = getattr(dag, attr); break
        for attr in ("outputs", "sinks", "sinks_list", "output_vertices"):
            if hasattr(dag, attr):
                outputs = getattr(dag, attr); break
        for attr in ("n", "num_vertices", "num_nodes", "size"):
            if hasattr(dag, attr):
                try: n = int(getattr(dag, attr)); break
                except Exception: pass
    if parents is None and isinstance(dag, (list, tuple)):
        if len(dag) and all(isinstance(x, (list, tuple, set)) for x in dag):
            parents = dag
    if n is None and parents is not None:
        n = len(parents)
    if n is None:
        return None, None, None
    parents = [list(p) for p in parents]
    if len(parents) < n:
        parents += [[] for _ in range(n - len(parents))]
    if outputs is None:
        children = [[] for _ in range(n)]
        for v in range(n):
            for p in parents[v]:
                children[p].append(v)
        outputs = [v for v in range(n) if not children[v]]
    outputs = list(outputs)
    return parents, outputs, n


def _topo(n, parents):
    indeg = [len(parents[v]) for v in range(n)]
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    from collections import deque
    q = deque(v for v in range(n) if indeg[v] == 0)
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != n:
        seen = set(order)
        order += [v for v in range(n) if v not in seen]
    return order, children


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 85.0
    parents, outputs, n = _extract(dag)
    if n is None:
        return []
    budget = max(1, int(fast_memory_budget))
    order, children = _topo(n, parents)

    is_input = [len(parents[v]) == 0 for v in range(n)]
    output_set = set(outputs)

    # needed set: outputs and their ancestors
    needed = set(output_set)
    for v in reversed(order):
        if v in needed:
            for p in parents[v]:
                needed.add(p)

    # future use count = number of children that are needed
    uses = [0] * n
    for v in range(n):
        if v not in needed:
            continue
        for c in children[v]:
            if c in needed:
                uses[v] += 1

    in_mem = set()
    on_disk = set()   # values that exist (computed and stored) but not in memory
    ops = []
    MAX_OPS = 20000

    def time_up():
        return (time.time() - t0) > DEADLINE

    def evict():
        # choose victim: min (uses, is_output)
        best = None; best_key = None
        for x in in_mem:
            key = (uses[x], 1 if x in output_set else 0)
            if best_key is None or key < best_key:
                best_key = key; best = x
        if best is None:
            return
        in_mem.discard(best)
        # decide store vs delete: if all parents of best are in memory (or inputs),
        # we can recompute cheaply -> delete (0 transfers). Else store (1 transfer).
        ps = parents[best]
        can_recompute = all((p in in_mem) or is_input[p] for p in ps)
        if can_recompute:
            on_disk.discard(best)
            ops.append(["delete", best])
        else:
            on_disk.add(best)
            ops.append(["store", best])

    def ensure(v):
        if v in in_mem:
            return True
        if v in on_disk:
            if len(in_mem) >= budget:
                evict()
            in_mem.add(v)
            ops.append(["load", v])
            return True
        # recompute: need parents in memory
        for p in parents[v]:
            if not ensure(p):
                return False
        if len(in_mem) >= budget:
            evict()
        in_mem.add(v)
        on_disk.add(v)
        ops.append(["compute", v])
        return True

    for v in order:
        if len(ops) >= MAX_OPS or time_up():
            break
        if v not in needed:
            continue
        # inputs are pre-loaded
        if is_input[v]:
            if v not in in_mem and len(in_mem) < budget:
                in_mem.add(v)
            continue
        ensure(v)

    for o in outputs:
        if len(ops) >= MAX_OPS or time_up():
            break
        ensure(o)

    if len(ops) > MAX_OPS:
        ops = ops[:MAX_OPS]
    return ops
# EVOLVE-BLOCK-END
