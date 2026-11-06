"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0

    # Parse DAG
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        vertices = dag.get('vertices', None)
        inputs = dag.get('inputs', None)
    else:
        raise ValueError("Unknown dag format")

    if isinstance(parents, (list, tuple)):
        parents = {i: list(p) for i, p in enumerate(parents)}

    if vertices is None:
        all_v = set(parents.keys())
        for p_list in parents.values():
            for p in p_list:
                all_v.add(p)
        vertices = sorted(all_v)

    if inputs is None:
        inputs = [v for v in vertices if len(parents.get(v, [])) == 0]

    # Build children map
    children = defaultdict(list)
    for v, p_list in parents.items():
        for p in p_list:
            children[p].append(v)

    # Topological order
    topo_order = list(vertices)

    # Precompute: for each vertex, the next time it's needed (in topo order)
    # This helps with eviction decisions
    next_use = {}
    # For each vertex, find the index of its last child in topo_order
    # Actually, we need to know when each value is last needed
    # A value v is needed by all its children. The last time v is needed
    # is the maximum index of v's children in topo_order.
    for v in vertices:
        if children[v]:
            # Find the last child in topo_order
            last_idx = -1
            for c in children[v]:
                idx = topo_order.index(c) if c in topo_order else -1
                if idx > last_idx:
                    last_idx = idx
            next_use[v] = last_idx
        else:
            next_use[v] = -1  # never needed again after compute

    # Also, for inputs, they are "loaded" and then needed by their children
    # For computed vertices, they are "computed" and then needed by their children

    def count_transfers(sch):
        return sum(1 for op, v in sch if op in ('load', 'store'))

    def verify_schedule(sch):
        fast = set()
        computed = set()
        for op, v in sch:
            if op == 'load':
                if v in fast:
                    return False
                if len(fast) >= fast_memory_budget:
                    return False
                fast.add(v)
            elif op == 'store':
                if v not in fast:
                    return False
                fast.discard(v)
            elif op == 'compute':
                p_list = parents.get(v, [])
                for p in p_list:
                    if p not in fast:
                        return False
                if v in fast:
                    return False
                if len(fast) >= fast_memory_budget:
                    return False
                fast.add(v)
                computed.add(v)
            elif op == 'delete':
                if v not in fast:
                    return False
                fast.discard(v)
            else:
                return False
        for v in outputs:
            if v not in computed:
                return False
        return True

    # Strategy: Process in topological order, use next_use for eviction
    def schedule_with_next_use():
        sch = []
        fast = set()

        for i, v in enumerate(topo_order):
            p_list = parents.get(v, [])
            for p in p_list:
                if p not in fast:
                    # Need to make room if over budget
                    while len(fast) >= fast_memory_budget:
                        # Evict the value with the latest next_use (or -1)
                        # among non-outputs
                        best_evict = None
                        best_next = -2  # -1 means never needed, so -2 is worse
                        for fv in fast:
                            if fv in outputs:
                                continue
                            nu = next_use.get(fv, -1)
                            if nu > best_next:
                                best_next = nu
                                best_evict = fv
                        if best_evict is None:
                            # All are outputs, can't evict
                            break
                        sch.append(['store', best_evict])
                        fast.discard(best_evict)
                    sch.append(['load', p])
                    fast.add(p)

            sch.append(['compute', v])
            fast.add(v)

        # Store outputs
        for v in outputs:
            if v in fast:
                sch.append(['store', v])
                fast.discard(v)

        return sch

    # Alternative: keep values that are needed soon
    def schedule_keep_soon():
        sch = []
        fast = set()

        for i, v in enumerate(topo_order):
            p_list = parents.get(v, [])
            for p in p_list:
                if p not in fast:
                    while len(fast) >= fast_memory_budget:
                        # Evict the value with the furthest next_use
                        best_evict = None
                        best_dist = -1
                        for fv in fast:
                            if fv in outputs:
                                continue
                            nu = next_use.get(fv, -1)
                            if nu == -1:
                                dist = float('inf')
                            else:
                                dist = nu - i
                            if dist > best_dist:
                                best_dist = dist
                                best_evict = fv
                        if best_evict is None:
                            break
                        sch.append(['store', best_evict])
                        fast.discard(best_evict)
                    sch.append(['load', p])
                    fast.add(p)

            sch.append(['compute', v])
            fast.add(v)

        for v in outputs:
            if v in fast:
                sch.append(['store', v])
                fast.discard(v)

        return sch

    # Generate candidates
    candidates = []
    for name, func in [("next_use", schedule_with_next_use),
                        ("keep_soon", schedule_keep_soon)]:
        try:
            sch = func()
            if verify_schedule(sch):
                t = count_transfers(sch)
                candidates.append((t, sch))
        except Exception:
            pass

    # Fallback: simple load-all-inputs, compute, store-outputs
    def simple_schedule():
        sch = []
        fast = set()
        for v in inputs:
            sch.append(['load', v])
            fast.add(v)
        for v in topo_order:
            p_list = parents.get(v, [])
            for p in p_list:
                if p not in fast:
                    while len(fast) >= fast_memory_budget:
                        for fv in list(fast):
                            if fv not in outputs:
                                sch.append(['store', fv])
                                fast.discard(fv)
                                break
                        else:
                            break
                    sch.append(['load', p])
                    fast.add(p)
            sch.append(['compute', v])
            fast.add(v)
        for v in outputs:
            if v in fast:
                sch.append(['store', v])
                fast.discard(v)
        return sch

    if not candidates:
        sch = simple_schedule()
        if verify_schedule(sch):
            candidates.append((count_transfers(sch), sch))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])
    best_sch = candidates[0][1]

    if time.time() - start_time > DEADLINE:
        return best_sch

    return best_sch
# EVOLVE-BLOCK-END
