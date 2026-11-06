"""Communication-minimizing DAG pebbling with recompute strategy."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0

    layers = dag['layers']
    parents = dag['parents']
    outputs = dag['outputs']
    output_set = set(outputs)
    n_layers = len(layers)

    depth = {}
    for li, layer in enumerate(layers):
        for v in layer:
            depth[v] = li

    B = max(1, int(fast_memory_budget))
    MAX_OPS = 20000
    ops = []
    ops_count = 0

    def emit(op, v):
        nonlocal ops_count
        if ops_count >= MAX_OPS:
            raise RuntimeError("too many ops")
        ops.append([op, v])
        ops_count += 1

    if n_layers <= 1:
        return ops

    # Find all needed vertices
    all_needed = set()
    stack = list(outputs)
    while stack:
        v = stack.pop()
        if v in all_needed:
            continue
        all_needed.add(v)
        for p in parents.get(v, []):
            stack.append(p)

    # Build children map
    children = {}
    for v in all_needed:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)

    # Layer-by-layer processing with smart memory management
    fast_memory = set()

    for li in range(1, n_layers):
        if time.time() > deadline:
            break
        layer_vertices = sorted([v for v in all_needed if depth.get(v, -1) == li], key=lambda x: x)
        if not layer_vertices:
            continue

        # Find needed parents
        needed_parents = set()
        for v in layer_vertices:
            for p in parents.get(v, []):
                needed_parents.add(p)

        # Load needed parents
        for p in sorted(needed_parents, key=lambda x: (depth.get(x, -1), x)):
            if p not in fast_memory:
                emit('load', p)
                fast_memory.add(p)

        # Compute
        for v in layer_vertices:
            emit('compute', v)
            fast_memory.add(v)

        # Compute what's needed in future
        still_needed = set()
        for v in all_needed:
            if depth.get(v, -1) > li:
                for p in parents.get(v, []):
                    still_needed.add(p)

        if li == n_layers - 1:
            for v in layer_vertices:
                if v in output_set:
                    emit('store', v)
                    fast_memory.discard(v)
            for v in list(fast_memory):
                emit('delete', v)
                fast_memory.discard(v)
        else:
            for v in list(fast_memory):
                if v not in still_needed and depth.get(v, -1) < li:
                    emit('delete', v)
                    fast_memory.discard(v)
            if len(fast_memory) > B:
                to_store = sorted(fast_memory, key=lambda x: (depth.get(x, -1), x))
                excess = len(fast_memory) - B
                for v in to_store[:excess]:
                    emit('store', v)
                    fast_memory.discard(v)

    for v in outputs:
        if v in fast_memory:
            emit('store', v)
            fast_memory.discard(v)

    return ops
# EVOLVE-BLOCK-END
