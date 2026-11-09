# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using greedy topological-order
    processing with immediate cleanup of no-longer-needed vertices.
    
    Strategy:
      - Process vertices in topological order.
      - Before computing a vertex, load all its parents into fast memory.
      - After computing, free up (store or delete) any vertices whose all
        children have been computed.
      - This minimizes the number of load/store operations by keeping the
        fast-memory working set as small as possible while respecting the
        dependency order.
    
    Args:
        dag: dict with 'n' (int), 'parents' (list of lists), 'outputs' (list of int)
        fast_memory_budget: int
    
    Returns:
        list of [operation, vertex] with operations load/store/compute/delete
    """
    import time
    _t0 = time.time()
    _deadline = _t0 + 85.0

    n = dag['n']
    parents = dag['parents']
    outputs = dag['outputs']
    B = fast_memory_budget

    # Build adjacency
    children = [[] for _ in range(n)]
    for v in range(n):
        for u in parents[v]:
            children[u].append(v)

    # Topological sort (Kahn's algorithm)
    in_deg = [len(parents[v]) for v in range(n)]
    from collections import deque
    dq = deque()
    for i in range(n):
        if in_deg[i] == 0:
            dq.append(i)
    top_order = []
    temp_deg = in_deg[:]
    while dq:
        u = dq.popleft()
        top_order.append(u)
        for w in children[u]:
            temp_deg[w] -= 1
            if temp_deg[w] == 0:
                dq.append(w)

    if len(top_order) < n:
        # Shouldn't happen for valid DAG; fallback
        return [['compute', v] for v in range(n)]

    is_input = [len(parents[v]) == 0 for v in range(n)]
    is_output = [False] * n
    for o in outputs:
        is_output[o] = True

    red = [False] * n
    pending = [len(children[v]) for v in range(n)]
    ops = []
    max_ops = 20000
    step = 0

    for v in top_order:
        if step >= max_ops or time.time() > _deadline:
            break

        # Load all parents not yet in fast memory
        for u in parents[v]:
            if not red[u]:
                ops.append(['load', u])
                red[u] = True
                step += 1
                if step >= max_ops or time.time() > _deadline:
                    break
        if step >= max_ops:
            break

        # Compute v
        ops.append(['compute', v])
        red[v] = True
        step += 1

        # Decrement pending count for all parents
        for u in parents[v]:
            pending[u] -= 1

        # Cleanup: free vertices no longer needed by any future computation
        for _pass in range(n + 1):
            if step >= max_ops or time.time() > _deadline:
                break
            progress = False
            for u in range(n):
                if step >= max_ops or time.time() > _deadline:
                    break
                if not red[u]:
                    continue
                if pending[u] > 0:
                    continue
                # Vertex u is no longer needed
                if is_output[u]:
                    ops.append(['store', u])
                else:
                    ops.append(['delete', u])
                red[u] = False
                step += 1
                progress = True
            if not progress:
                break

    # Final pass: ensure all outputs end up in slow memory (blue)
    for o in outputs:
        if step >= max_ops or time.time() > _deadline:
            break
        if red[o]:
            ops.append(['store', o])
            red[o] = False
            step += 1

    if len(ops) > max_ops:
        ops = ops[:max_ops]

    return ops
# EVOLVE-BLOCK-END
