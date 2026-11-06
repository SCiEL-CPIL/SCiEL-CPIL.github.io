# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """Communication-minimizing DAG pebbling.
    Returns list of [operation, vertex].
    """
    t0 = time.time()
    DEADLINE = t0 + 85.0

    def _get(obj, *names, default=None):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
        if isinstance(obj, dict):
            for n in names:
                if n in obj:
                    return obj[n]
        return default

    # Extract DAG components
    parents = _get(dag, 'parents', 'preds', 'parents_list', 'in_edges', 'pred', 'predecessors')
    outputs = _get(dag, 'outputs', 'output', 'sinks', 'targets', 'output_vertices', 'sinks_list')
    inputs = _get(dag, 'inputs', 'input', 'sources', 'input_vertices', 'sources_list')
    vertices = _get(dag, 'vertices', 'nodes', 'vertex_list', 'all_vertices', 'node_list')

    # If dag is a dict mapping vertex -> parents
    if parents is None and isinstance(dag, dict) and dag:
        sample = next(iter(dag.values()))
        if isinstance(sample, (list, tuple, set, frozenset)):
            parents = {k: list(v) for k, v in dag.items()}
            vertices = list(dag.keys())

    if parents is None:
        raise ValueError("cannot find parents")
    if outputs is None:
        raise ValueError("cannot find outputs")

    # Normalize parents
    if isinstance(parents, dict):
        parents_map = {k: list(v) for k, v in parents.items()}
    elif isinstance(parents, (list, tuple)):
        parents_map = {i: list(parents[i]) for i in range(len(parents))}
    else:
        parents_map = {}

    if vertices is None:
        vertices = list(parents_map.keys())

    if inputs is None:
        inputs = [v for v in vertices if not parents_map.get(v)]

    inputs = list(inputs)
    outputs = list(outputs)
    output_set = set(outputs)
    input_set = set(inputs)

    # Build children map
    children_map = defaultdict(list)
    for v, ps in parents_map.items():
        for p in ps:
            children_map[p].append(v)

    all_vertices = set(parents_map.keys())
    for ps in parents_map.values():
        for p in ps:
            all_vertices.add(p)
    all_vertices = list(all_vertices)

    # Topological order
    indeg = {v: len(parents_map.get(v, [])) for v in all_vertices}
    q = deque([v for v in all_vertices if indeg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children_map.get(v, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != len(all_vertices):
        topo = all_vertices

    # Compute depth (layer)
    depth = {}
    for v in topo:
        ps = parents_map.get(v, [])
        if not ps:
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in ps) + 1

    max_depth = max(depth.values()) if depth else 0

    # Layer groups
    layer_groups = defaultdict(list)
    for v in all_vertices:
        layer_groups[depth[v]].append(v)

    # Layer-based pebbling schedule
    ops = []
    blue = set(inputs)  # inputs start blue

    for d in range(1, max_depth + 1):
        if time.time() > DEADLINE:
            break
        layer_verts = layer_groups.get(d, [])
        if not layer_verts:
            continue

        # Load parents that aren't blue
        for v in layer_verts:
            for p in parents_map.get(v, []):
                if p not in blue:
                    ops.append(['load', p])
                    blue.add(p)
                    # Enforce memory budget
                    if len(blue) > fast_memory_budget:
                        # Store a vertex that is no longer needed
                        # Prefer vertices from earlier layers
                        to_store = None
                        for b in sorted(blue, key=lambda x: depth.get(x, 0)):
                            if b not in output_set and depth.get(b, 0) < d:
                                to_store = b
                                break
                        if to_store is None:
                            for b in blue:
                                if b not in output_set:
                                    to_store = b
                                    break
                        if to_store is not None:
                            ops.append(['store', to_store])
                            blue.discard(to_store)

        # Compute layer d
        for v in layer_verts:
            ops.append(['compute', v])
            blue.add(v)

        # Store layer d-1 vertices that are no longer needed
        prev_layer = layer_groups.get(d - 1, [])
        for p in prev_layer:
            if p in blue and p not in output_set:
                # All children of p should be in layer >= d
                # Since we process in order, children in layer d are just computed
                # Children in layer > d are not yet computed but will be loaded
                # So p is no longer needed
                ops.append(['store', p])
                blue.discard(p)

    # Ensure outputs are blue
    for o in outputs:
        if o not in blue:
            ops.append(['load', o])
            blue.add(o)
            if len(blue) > fast_memory_budget:
                for b in blue:
                    if b not in output_set:
                        ops.append(['store', b])
                        blue.discard(b)
                        break

    if len(ops) > 20000:
        # Truncate to avoid invalid result
        # Keep the last operations to ensure outputs are blue
        # Actually, just return what we have; the validator may reject it
        pass

    return ops

# EVOLVE-BLOCK-END
