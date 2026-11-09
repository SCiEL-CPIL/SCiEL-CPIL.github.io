"""Communication-minimizing DAG pebbling; EVOLVE-BLOCK is the searchable region."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque


def _parse_dag(dag):
    """Parse various possible DAG formats into a unified structure."""
    if isinstance(dag, dict):
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        inputs = dag.get('inputs', None)
        vertices = dag.get('vertices', None)
        n = dag.get('n', None)

        if parents is None:
            if isinstance(dag, list):
                parents = [list(p) for p in dag]
                n = len(dag)
                if outputs is None:
                    outputs = [n - 1]
                if inputs is None:
                    inputs = [i for i in range(n) if not parents[i]]
            else:
                return None, None, None, None, None
        else:
            if isinstance(parents, list):
                parents = [list(p) for p in parents]
                n = len(parents)
            elif isinstance(parents, dict):
                n = max(parents.keys()) + 1 if parents else 0
                parents_list = [list(parents.get(i, [])) for i in range(n)]
                parents = parents_list

            if outputs is None:
                children = [[] for _ in range(n)]
                for v in range(n):
                    for p in parents[v]:
                        children[p].append(v)
                outputs = [v for v in range(n) if not children[v]]

            if inputs is None:
                inputs = [v for v in range(n) if not parents[v]]

        return parents, outputs, inputs, vertices, n

    elif isinstance(dag, list):
        parents = [list(p) if isinstance(p, (list, tuple)) else [] for p in dag]
        n = len(dag)
        outputs = [n - 1] if n > 0 else []
        inputs = [i for i in range(n) if not parents[i]]
        return parents, outputs, inputs, None, n

    return None, None, None, None, None


def _build_children(parents, n):
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    return children


def _compute_depth(parents, n):
    depth = [0] * n
    for v in range(n):
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    return depth


def _count_transfers(ops):
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _validate_ops(ops, parents, outputs, fast_memory_budget):
    n = len(parents)
    in_fast = set()
    stored = set()
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                continue
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            stored.add(v)
        elif op == 'compute':
            for p in parents[v]:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        else:
            return False
    for o in outputs:
        if o not in stored:
            return False
    return True


def _lru_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Topological-order pebbling with precise in_fast tracking and
    least-recently-needed eviction. Loads a parent only when it is absent
    from fast memory, and evicts the non-output value whose next use is
    furthest in the future (smallest next-use index wins).
    """
    n = len(parents)
    if n == 0:
        return []

    children = _build_children(parents, n)
    is_output = set(outputs)

    # next-use index: the smallest child index (in topological order) that
    # still needs this vertex. This is the "distance to last remaining use".
    next_use = [-1] * n
    for v in range(n):
        if children[v]:
            next_use[v] = min(children[v])

    ops = []
    in_fast = set()

    def evict_one():
        """Evict the non-output in_fast value with the smallest next_use (furthest
        future use). Returns True if something was evicted."""
        best = None
        best_key = None
        for u in in_fast:
            if u in is_output:
                continue
            nu = next_use[u]
            if nu < 0:
                nu = n  # never needed again -> evict first
            # prefer smallest next_use (most urgent to keep is largest next_use)
            if best_key is None or nu < best_key:
                best_key = nu
                best = u
        if best is not None:
            ops.append(['delete', best])
            in_fast.discard(best)
            return True
        # all in fast are outputs; evict any
        if in_fast:
            u = next(iter(in_fast))
            ops.append(['delete', u])
            in_fast.discard(u)
            return True
        return False

    def ensure_room():
        while len(in_fast) >= fast_memory_budget:
            if not evict_one():
                return False
        return True

    for v in range(n):
        if time.time() > deadline:
            return None

        # Load parents that are not already in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if not ensure_room():
                    return None
                ops.append(['load', p])
                in_fast.add(p)

        # Compute v if it has parents
        if parents[v]:
            if v not in in_fast:
                if not ensure_room():
                    return None
                ops.append(['compute', v])
                in_fast.add(v)
            # advance next_use of parents: v is now consumed
            for p in parents[v]:
                # if v was the next use, move to the child after v
                if next_use[p] == v:
                    later = [c for c in children[p] if c > v]
                    next_use[p] = min(later) if later else -1

        # Inputs that are outputs: load and store directly
        if not parents[v] and v in is_output:
            if v not in in_fast:
                if not ensure_room():
                    return None
                ops.append(['load', v])
                in_fast.add(v)
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)

        # Store computed outputs
        if v in is_output and parents[v] and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)

        # Delete parents that are no longer needed (keep working set clean)
        for p in parents[v]:
            if p in in_fast and p not in is_output and next_use[p] < 0:
                ops.append(['delete', p])
                in_fast.discard(p)

    # Final: store any outputs still in fast memory (defensive)
    for o in outputs:
        if o in in_fast:
            ops.append(['store', o])
            in_fast.discard(o)

    return ops


def _layered_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Layer-by-layer pebbling: for each depth layer, load the distinct parents
    once, compute the whole layer, store non-outputs, and delete the previous
    layer. Reuses in_fast across the layer to avoid redundant loads.
    """
    n = len(parents)
    if n == 0:
        return []

    children = _build_children(parents, n)
    depth = _compute_depth(parents, n)
    is_output = set(outputs)

    max_depth = max(depth) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        layers[depth[v]].append(v)

    ops = []
    in_fast = set()

    def evict_one():
        best = None
        for u in in_fast:
            if u not in is_output:
                best = u
                break
        if best is None and in_fast:
            best = next(iter(in_fast))
        if best is not None:
            ops.append(['delete', best])
            in_fast.discard(best)
            return True
        return False

    def ensure_room():
        while len(in_fast) >= fast_memory_budget:
            if not evict_one():
                return False
        return True

    for d in range(max_depth + 1):
        if time.time() > deadline:
            return None
        layer = layers[d]
        if not layer:
            continue

        if d == 0:
            # inputs: load each (only if not already present)
            for v in layer:
                if v not in in_fast:
                    if not ensure_room():
                        return None
                    ops.append(['load', v])
                    in_fast.add(v)
            # store any that are outputs
            for v in layer:
                if v in is_output and v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
            continue

        # Load distinct parents of this layer that are absent
        needed = set()
        for v in layer:
            for p in parents[v]:
                needed.add(p)
        for p in sorted(needed):
            if p not in in_fast:
                if not ensure_room():
                    return None
                ops.append(['load', p])
                in_fast.add(p)

        # Compute the layer
        for v in layer:
            if v not in in_fast:
                if not ensure_room():
                    return None
                ops.append(['compute', v])
                in_fast.add(v)

        # Store non-outputs of this layer (so next layer can load them)
        for v in layer:
            if v not in is_output:
                ops.append(['store', v])
                in_fast.discard(v)
            # else keep in fast in case it's needed by a later layer

        # Delete previous layer values that are no longer needed
        if d - 1 >= 0:
            for v in layers[d - 1]:
                if v in in_fast and v not in is_output:
                    # check if still needed by a later layer
                    still = any(c > layers[d - 1] and False for c in [])  # placeholder
                    ops.append(['delete', v])
                    in_fast.discard(v)

    # Store any outputs still in fast memory
    for o in outputs:
        if o in in_fast:
            ops.append(['store', o])
            in_fast.discard(o)

    return ops


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        result = pebble_baseline(dag, fast_memory_budget)
        if result is not None:
            return result
    except Exception:
        pass

    parents, outputs, inputs, vertices, n = _parse_dag(dag)

    if parents is None or n is None:
        return []

    if outputs is None:
        outputs = [n - 1]

    if fast_memory_budget is None:
        fast_memory_budget = n
    if fast_memory_budget <= 0:
        fast_memory_budget = 1

    best_ops = None
    best_transfers = float('inf')

    candidates = [
        _lru_pebbling,
        _layered_pebbling,
    ]

    for fn in candidates:
        try:
            ops = fn(parents, outputs, fast_memory_budget, deadline)
            if ops is not None and len(ops) <= 20000 and \
               _validate_ops(ops, parents, outputs, fast_memory_budget):
                transfers = _count_transfers(ops)
                if transfers < best_transfers:
                    best_transfers = transfers
                    best_ops = ops
        except Exception:
            pass

    if best_ops is not None:
        return best_ops

    # Fallback: simple topological load/store/compute
    try:
        ops = []
        in_fast = set()
        is_output = set(outputs)
        children = _build_children(parents, n)

        for v in range(n):
            if time.time() > deadline:
                break
            for p in parents[v]:
                if p not in in_fast:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', p])
                        in_fast.add(p)
            if parents[v]:
                while len(in_fast) >= fast_memory_budget:
                    evicted = False
                    for u in list(in_fast):
                        if u not in is_output:
                            ops.append(['delete', u])
                            in_fast.discard(u)
                            evicted = True
                            break
                    if not evicted:
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
            if v in is_output:
                if v not in in_fast and not parents[v]:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', v])
                        in_fast.add(v)
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)

        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass

    return []

# EVOLVE-BLOCK-END
