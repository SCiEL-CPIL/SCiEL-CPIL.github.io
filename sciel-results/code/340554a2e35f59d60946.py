"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def _parse_dag(dag):
    """Robustly extract vertices, parents, inputs, outputs, children from dag."""
    # vertices
    vertices = None
    if isinstance(dag, dict):
        for key in ('vertices', 'nodes', 'V'):
            if key in dag:
                vertices = list(dag[key]); break
        if vertices is None:
            # maybe dag is a list of vertex dicts
            if isinstance(dag.get('vertices'), list):
                vertices = list(dag['vertices'])
    else:
        for attr in ('vertices', 'nodes', 'V'):
            if hasattr(dag, attr):
                vertices = list(getattr(dag, attr)); break
    if vertices is None:
        # try to treat dag itself as iterable of vertices
        try:
            vertices = list(dag)
        except Exception:
            raise ValueError("Cannot extract vertices")
    # dedupe while preserving order
    seen = set(); verts = []
    for v in vertices:
        if v not in seen:
            seen.add(v); verts.append(v)
    vertices = verts
    n = len(vertices)
    vset = set(vertices)

    # parents
    parents = None
    if isinstance(dag, dict):
        for key in ('parents', 'preds', 'predecessors', 'in_edges', 'inputs_to'):
            if key in dag:
                parents = dag[key]; break
    else:
        for attr in ('parents', 'preds', 'predecessors', 'in_edges'):
            if hasattr(dag, attr):
                parents = getattr(dag, attr); break

    if parents is None:
        parents = {v: [] for v in vertices}
    elif isinstance(parents, dict):
        parents = {v: list(parents.get(v, [])) for v in vertices}
    elif isinstance(parents, (list, tuple)):
        # list of lists aligned with vertices order
        parents = {v: list(parents[i]) if i < len(parents) else [] for i, v in enumerate(vertices)}
    else:
        parents = {v: [] for v in vertices}
    # ensure entries, filter to known vertices
    for v in vertices:
        if v not in parents:
            parents[v] = []
        parents[v] = [p for p in parents[v] if p in vset]

    # inputs
    inputs = None
    if isinstance(dag, dict):
        for key in ('inputs', 'sources', 'input_vertices', 'leaves'):
            if key in dag:
                inputs = dag[key]; break
    else:
        for attr in ('inputs', 'sources', 'input_vertices'):
            if hasattr(dag, attr):
                inputs = getattr(dag, attr); break
    if inputs is None:
        inputs = [v for v in vertices if not parents[v]]
    inputs = [v for v in inputs if v in vset]
    input_set = set(inputs)

    # outputs
    outputs = None
    if isinstance(dag, dict):
        for key in ('outputs', 'sinks', 'output_vertices', 'targets'):
            if key in dag:
                outputs = dag[key]; break
    else:
        for attr in ('outputs', 'sinks', 'output_vertices'):
            if hasattr(dag, attr):
                outputs = getattr(dag, attr); break
    if outputs is None:
        # children map
        children = defaultdict(list)
        for v in vertices:
            for p in parents[v]:
                children[p].append(v)
        outputs = [v for v in vertices if not children[v]]
    outputs = [v for v in outputs if v in vset]
    output_set = set(outputs)

    # children
    children = defaultdict(list)
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)

    return vertices, parents, input_set, output_set, children

def _topo_order(vertices, parents):
    indeg = {v: len(parents[v]) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return order

def _simulate(ops, vertices, parents, input_set, output_set, fast_budget):
    """Return (valid, transfers). transfers = number of load+store ops."""
    fast = set()
    transfers = 0
    for op, v in ops:
        if op == 'load':
            if v in fast:
                continue
            if len(fast) >= fast_budget:
                return False, transfers  # overflow
            fast.add(v); transfers += 1
        elif op == 'store':
            if v not in fast:
                return False, transfers
            fast.discard(v); transfers += 1
        elif op == 'compute':
            for p in parents[v]:
                if p not in fast:
                    return False, transfers
            if v not in fast:
                if len(fast) >= fast_budget:
                    return False, transfers
                fast.add(v)
        elif op == 'delete':
            if v in fast:
                fast.discard(v)
        else:
            return False, transfers
    # all outputs must be blue at end
    for v in output_set:
        if v not in fast:
            return False, transfers
    return True, transfers

def _naive_schedule(dag, fast_memory_budget):
    """Reference naive schedule: load all inputs, compute in topo order, keep everything, store outputs.
    This is the baseline the score is normalized against."""
    vertices, parents, input_set, output_set, children = _parse_dag(dag)
    # naive: load all inputs, compute all in topo order, store outputs
    # keep all in fast (assume budget large enough for naive; if not, naive still defines the ratio)
    order = _topo_order(vertices, parents)
    ops = []
    for v in sorted(input_set):
        ops.append(['load', v])
    in_fast = set(input_set)
    for v in order:
        if v in input_set:
            continue
        ops.append(['compute', v])
        in_fast.add(v)
    for v in sorted(output_set):
        ops.append(['store', v])
    return ops

def _greedy_schedule(dag, fast_memory_budget):
    """Greedy pebbling: process vertices in topological order.
    Keep a vertex in fast memory if it will be needed again (has remaining children)
    and there is space; otherwise delete it. Load parents when needed.
    This minimizes re-loads by keeping useful intermediates."""
    vertices, parents, input_set, output_set, children = _parse_dag(dag)
    order = _topo_order(vertices, parents)
    # remaining children count (children not yet processed)
    remaining_children = {v: len(children[v]) for v in vertices}
    in_fast = set()
    ops = []
    transfers = 0

    def load(v):
        nonlocal transfers
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
            ops.append(['load', v])
            transfers += 1
        return True

    def delete(v):
        nonlocal transfers
        if v in in_fast:
            in_fast.discard(v)
            ops.append(['delete', v])

    # Pre-load inputs that are also outputs (must end blue) - actually just load when needed
    for v in order:
        # ensure all parents are in fast
        for p in parents[v]:
            if p not in in_fast:
                # p was deleted; need to reload. But reload of intermediate requires it was stored.
                # In pebbling, slow memory holds values that were stored. We only store outputs.
                # So intermediates that are deleted CANNOT be reloaded - they must be recomputed.
                # To avoid this complexity, we should NOT delete vertices that have remaining children.
                # If we did delete, we'd need recompute. Let's handle by keeping them.
                # Actually, the greedy should only delete if remaining_children == 0.
                # So this branch should not happen if we're careful.
                # Fallback: try to load (might fail validation)
                load(p)
        # compute v
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Need to free space: delete a vertex that is no longer needed
                freed = False
                for u in list(in_fast):
                    if u not in output_set and remaining_children[u] == 0:
                        delete(u)
                        freed = True
                        break
                if not freed:
                    # delete any non-output
                    for u in list(in_fast):
                        if u not in output_set:
                            delete(u)
                            freed = True
                            break
                if not freed:
                    # cannot proceed; break
                    break
            in_fast.add(v)
            ops.append(['compute', v])
        # decrement remaining children of parents
        for p in parents[v]:
            remaining_children[p] -= 1
        # if v is not an output and has no remaining children, delete it
        if v not in output_set and remaining_children[v] == 0:
            delete(v)

    # store outputs
    for v in sorted(output_set):
        if v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)

    return ops

def _layer_schedule(dag, fast_memory_budget):
    """Layer-based schedule: process by depth, keeping only current and next layer."""
    vertices, parents, input_set, output_set, children = _parse_dag(dag)
    # compute depth
    depth = {}
    order = _topo_order(vertices, parents)
    for v in order:
        if not parents[v]:
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) + 1 for p in parents[v])
    max_depth = max(depth.values()) if depth else 0
    layers = defaultdict(list)
    for v in vertices:
        layers[depth[v]].append(v)

    in_fast = set()
    ops = []

    for d in range(max_depth + 1):
        layer = sorted(layers.get(d, []))
        # load parents needed for this layer that aren't in fast
        for v in layer:
            for p in parents[v]:
                if p not in in_fast:
                    # p is from an earlier layer; if it was deleted, we'd need recompute
                    # For simplicity, load it (assumes it's still available or we keep it)
                    if len(in_fast) < fast_memory_budget:
                        in_fast.add(p)
                        ops.append(['load', p])
        # compute vertices in this layer
        for v in layer:
            if v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    # free space
                    for u in list(in_fast):
                        if u not in output_set and depth[u] < d:
                            in_fast.discard(u)
                            ops.append(['delete', u])
                            break
                in_fast.add(v)
                ops.append(['compute', v])
        # after processing layer d, delete vertices from layer d that are not outputs
        # and not needed by layer d+1
        next_layer = set(layers.get(d + 1, []))
        for v in layer:
            if v not in output_set and v not in next_layer:
                # check if any future layer needs v
                needed = False
                for dd in range(d + 1, max_depth + 1):
                    for w in layers.get(dd, []):
                        if v in parents[w]:
                            needed = True
                            break
                    if needed:
                        break
                if not needed:
                    if v in in_fast:
                        in_fast.discard(v)
                        ops.append(['delete', v])

    # store outputs
    for v in sorted(output_set):
        if v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)

    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 80  # Safety margin

    try:
        vertices, parents, input_set, output_set, children = _parse_dag(dag)
        n = len(vertices)

        # Try multiple strategies and pick the best valid one
        candidates = []

        # Strategy 1: Greedy (keep vertices with remaining children)
        if time.time() < DEADLINE:
            ops1 = _greedy_schedule(dag, fast_memory_budget)
            if len(ops1) <= 20000:
                valid, transfers = _simulate(ops1, vertices, parents, input_set, output_set, fast_memory_budget)
                if valid:
                    candidates.append((transfers, ops1, 'greedy'))

        # Strategy 2: Layer-based
        if time.time() < DEADLINE:
            ops2 = _layer_schedule(dag, fast_memory_budget)
            if len(ops2) <= 20000:
                valid, transfers = _simulate(ops2, vertices, parents, input_set, output_set, fast_memory_budget)
                if valid:
                    candidates.append((transfers, ops2, 'layer'))

        # Strategy 3: Naive (load all, compute, store) - only valid if budget >= n
        if fast_memory_budget >= n:
            ops3 = _naive_schedule(dag, fast_memory_budget)
            if len(ops3) <= 20000:
                valid, transfers = _simulate(ops3, vertices, parents, input_set, output_set, fast_memory_budget)
                if valid:
                    candidates.append((transfers, ops3, 'naive'))

        # Pick best
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        else:
            # Fallback: return naive even if invalid (shouldn't happen)
            return _naive_schedule(dag, fast_memory_budget)

    except Exception as e:
        # Last resort: minimal valid schedule if possible
        try:
            vertices, parents, input_set, output_set, children = _parse_dag(dag)
            return _naive_schedule(dag, fast_memory_budget)
        except Exception:
            return []
# EVOLVE-BLOCK-END
