"""Communication-minimizing DAG pebbling with greedy reuse-aware scheduling."""
import time

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 55.0  # safety margin before 60s kill

    # --- Parse DAG structure ---
    n = len(dag['parents'])
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    if not outputs:
        children = [[] for _ in range(n)]
        for v in range(n):
            for p in parents[v]:
                children[p].append(v)
        outputs = set(v for v in range(n) if not children[v])

    # Build children (reverse edges)
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)

    # Inputs: vertices with no parents
    inputs = set(v for v in range(n) if not parents[v])

    # Compute height (distance to outputs)
    height = [0] * n
    for v in range(n - 1, -1, -1):
        if v in outputs:
            height[v] = 0
        elif children[v]:
            height[v] = 1 + max(height[c] for c in children[v])
        else:
            height[v] = 0

    # remaining_deps[v]: how many children of v still need to be computed
    remaining_deps = [len(children[v]) for v in range(n)]

    # --- State ---
    # fast: set of vertex indices currently in fast memory
    # ext: set of vertex indices stored in external memory
    # computed: set of vertex indices whose value has been computed
    fast = set()
    ext = set()
    computed = set()
    ops = []

    def elapsed_ok():
        return (time.time() - t0) < DEADLINE

    def do_op(op_type, v):
        ops.append([op_type, v])

    # Inputs start blue = in fast memory
    for v in inputs:
        fast.add(v)
        computed.add(v)

    # To compute: all non-input vertices
    to_compute = set(range(n)) - inputs

    # Greedy loop
    while to_compute and elapsed_ok():
        # Find ready vertices (all parents in fast memory)
        ready = []
        for v in to_compute:
            if all(p in fast for p in parents[v]):
                ready.append(v)

        if not ready:
            # Need to load a parent into fast memory
            # Find a vertex in to_compute with a parent not in fast
            found = False
            for v in list(to_compute):
                for p in parents[v]:
                    if p not in fast:
                        # Need to load p
                        # Check if p is in external
                        if p in ext:
                            # Need room in fast
                            if len(fast) >= fast_memory_budget:
                                # Evict least valuable
                                min_cost = float('inf')
                                ev = None
                                for fv in fast:
                                    rd = remaining_deps[fv]
                                    # Value: how much more this will be needed
                                    if rd == 0 and fv not in outputs:
                                        cost = 0
                                    else:
                                        cost = rd * (1 + height[fv])
                                    # Don't evict outputs not yet stored
                                    if fv in outputs and fv not in ext:
                                        cost += 10000
                                    if cost < min_cost:
                                        min_cost = cost
                                        ev = fv
                                if ev is not None:
                                    fast.discard(ev)
                                    ext.add(ev)
                                    do_op('store', ev)
                            fast.add(p)
                            do_op('load', p)
                        elif p in computed:
                            # p was computed but not in fast/ext - shouldn't happen
                            # Recompute p: add back to to_compute
                            if p not in inputs:
                                to_compute.add(p)
                            # Skip for now
                            continue
                        else:
                            # p not computed - need to compute it first
                            if p not in inputs:
                                to_compute.add(p)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
            continue

        # Pick best vertex to compute
        # Priority: high reuse, high height (urgent), low depth
        best = max(ready, key=lambda v: (
            remaining_deps[v] * (1 + height[v]),  # persistence cost
            len(children[v]),
            -height[v]
        ))

        # Compute best
        fast.add(best)
        computed.add(best)
        to_compute.discard(best)
        do_op('compute', best)

        # Update remaining_deps for parents
        for p in parents[best]:
            remaining_deps[p] -= 1

        # Evict if over budget
        while len(fast) > fast_memory_budget and elapsed_ok():
            min_cost = float('inf')
            ev = None
            for fv in fast:
                rd = remaining_deps[fv]
                if rd == 0 and fv not in outputs:
                    cost = 0
                else:
                    cost = rd * (1 + height[fv])
                if fv in outputs and fv not in ext:
                    cost += 10000
                if cost < min_cost:
                    min_cost = cost
                    ev = fv
            if ev is not None:
                fast.discard(ev)
                ext.add(ev)
                do_op('store', ev)
            else:
                break

    # Ensure outputs are in fast memory (finish blue)
    for v in outputs:
        if v not in fast and elapsed_ok():
            if v in ext:
                if len(fast) >= fast_memory_budget:
                    # Evict non-output
                    min_cost = float('inf')
                    ev = None
                    for fv in fast:
                        if fv in outputs:
                            continue
                        rd = remaining_deps[fv]
                        cost = rd * (1 + height[fv]) if rd > 0 else 0
                        if cost < min_cost:
                            min_cost = cost
                            ev = fv
                    if ev is not None:
                        fast.discard(ev)
                        ext.add(ev)
                        do_op('store', ev)
                fast.add(v)
                do_op('load', v)
            elif v in computed:
                if len(fast) >= fast_memory_budget:
                    min_cost = float('inf')
                    ev = None
                    for fv in fast:
                        if fv in outputs:
                            continue
                        rd = remaining_deps[fv]
                        cost = rd * (1 + height[fv]) if rd > 0 else 0
                        if cost < min_cost:
                            min_cost = cost
                            ev = fv
                    if ev is not None:
                        fast.discard(ev)
                        ext.add(ev)
                        do_op('store', ev)
                fast.add(v)
                do_op('load', v)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

# EVOLVE-BLOCK-END
