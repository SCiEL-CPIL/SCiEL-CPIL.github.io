"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55  # Safety margin before 60s limit

    # Parse DAG structure
    vertices = []
    parent_map = {}
    output_set = set()
    input_set = set()

    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents_raw = dag.get('parents', {})
        if isinstance(parents_raw, dict):
            parent_map = {v: list(p) for v, p in parents_raw.items()}
        elif isinstance(parents_raw, list):
            parent_map = {i: list(parents_raw[i]) for i in range(len(parents_raw))}
        output_set = set(dag.get('outputs', []))
        input_set = set(dag.get('inputs', []))
    elif hasattr(dag, 'vertices'):
        v = dag.vertices
        vertices = list(v) if isinstance(v, (set, tuple)) else list(v)
        p = getattr(dag, 'parents', {})
        if isinstance(p, dict):
            parent_map = {k: list(val) for k, val in p.items()}
        elif isinstance(p, list):
            parent_map = {i: list(p[i]) for i in range(len(p))}
        output_set = set(getattr(dag, 'outputs', []))
        input_set = set(getattr(dag, 'inputs', []))
    else:
        # Fallback: try common attribute names
        for attr in ['vertices', 'nodes', 'V']:
            if hasattr(dag, attr):
                v = getattr(dag, attr)
                vertices = list(v) if isinstance(v, (set, tuple)) else list(v)
                break
        for attr in ['parents', 'preds', 'in_neighbors']:
            if hasattr(dag, attr):
                p = getattr(dag, attr)
                if isinstance(p, dict):
                    parent_map = {k: list(val) for k, val in p.items()}
                elif isinstance(p, list):
                    parent_map = {i: list(p[i]) for i in range(len(p))}
                break
        for attr in ['outputs', 'sinks', 'targets']:
            if hasattr(dag, attr):
                output_set = set(getattr(dag, attr))
                break
        for attr in ['inputs', 'sources']:
            if hasattr(dag, attr):
                input_set = set(getattr(dag, attr))
                break

    # Build children map
    children_map = defaultdict(list)
    for v, ps in parent_map.items():
        for p in ps:
            children_map[p].append(v)

    # Determine inputs and outputs if not specified
    if not input_set:
        for v in vertices:
            if v not in parent_map or len(parent_map.get(v, [])) == 0:
                input_set.add(v)

    if not output_set:
        for v in vertices:
            if v not in children_map:
                output_set.add(v)

    # Topological order (assume vertices are already in topological order)
    topo_order = list(vertices)

    # Greedy pebbling with lookahead
    fast_mem = set(input_set)
    ops = []

    # Track how many more times each vertex's value will be needed
    remaining_use = {v: len(children_map.get(v, [])) for v in vertices}

    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break

        if v in input_set:
            continue

        # Ensure parents are in fast memory
        for p in parent_map.get(v, []):
            if p not in fast_mem:
                # Need to load p into fast memory
                if len(fast_mem) >= fast_memory_budget:
                    # Evict: choose vertex with least remaining use, prefer non-outputs
                    best_evict = None
                    best_key = None
                    for fm_v in fast_mem:
                        ru = remaining_use.get(fm_v, 0)
                        is_out = 1 if fm_v in output_set else 0
                        key = (ru, is_out)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_evict = fm_v
                    if best_evict is not None:
                        ops.append(['delete', best_evict])
                        fast_mem.remove(best_evict)

                ops.append(['load', p])
                fast_mem.add(p)

        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)

        # Update remaining use for parents
        for p in parent_map.get(v, []):
            remaining_use[p] = remaining_use.get(p, 0) - 1

        # Free memory: delete parents with no remaining use (if not outputs)
        for p in parent_map.get(v, []):
            if remaining_use.get(p, 0) == 0 and p not in output_set:
                if p in fast_mem:
                    ops.append(['delete', p])
                    fast_mem.remove(p)

    # Ensure all outputs are in fast memory at the end
    for out_v in output_set:
        if out_v not in fast_mem:
            if time.time() - start_time > DEADLINE:
                break
            if len(fast_mem) >= fast_memory_budget:
                best_evict = None
                best_key = None
                for fm_v in fast_mem:
                    ru = remaining_use.get(fm_v, 0)
                    is_out = 1 if fm_v in output_set else 0
                    key = (ru, is_out)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_evict = fm_v
                if best_evict is not None:
                    ops.append(['delete', best_evict])
                    fast_mem.remove(best_evict)
            ops.append(['load', out_v])
            fast_mem.add(out_v)

    # Enforce operation limit
    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

# EVOLVE-BLOCK-END
