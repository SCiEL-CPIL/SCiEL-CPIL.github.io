def _get_vertices(dag):
    """Get list of vertex IDs from dag (list or dict)."""
    if isinstance(dag, dict):
        return sorted(dag.keys())
    else:
        return list(range(len(dag)))


def _topo_order(dag):
    """Return vertex list in topological order (inputs before outputs)."""
    vertices = _get_vertices(dag)
    n = len(vertices)
    
    # children: vertex -> list of vertices that depend on it
    children = {v: [] for v in vertices}
    indeg = {v: 0 for v in vertices}
    
    for v in vertices:
        parents = dag[v]
        for p in parents:
            if p in children:
                children[p].append(v)
                indeg[v] += 1

    from collections import deque
    q = deque([v for v in vertices if indeg[v] == 0])
    order = []
    
    while q:
        v = q.popleft()
        order.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
                
    if len(order) != n:
        order = vertices
        
    return order, children


def _naive_schedule(dag, budget):
    """Naive: load each input once, compute top-down, store each output.
    Returns (num_transfers, ops)."""
    vertices = _get_vertices(dag)
    order, children = _topo_order(dag)
    
    inputs = [v for v in vertices if len(dag[v]) == 0]
    outputs = [v for v in vertices if len(children[v]) == 0]
    
    ops = []
    for v in inputs:
        ops.append(["load", v])
        
    for v in order:
        if len(dag[v]) == 0:
            continue
        ops.append(["compute", v])
        
    for v in outputs:
        ops.append(["store", v])
        
    transfers = len(inputs) + len(outputs)
    return transfers, ops


def _simulate(dag, ops, budget):
    """Verify a schedule is valid; return True if valid, else False."""
    vertices = _get_vertices(dag)
    state = {v: 0 for v in vertices} # 0=blue, 1=red
    fast = set()
    
    for op, v in ops:
        if op == "load":
            if state[v] != 0:
                return False
            if len(fast) >= budget:
                return False
            state[v] = 1
            fast.add(v)
        elif op == "store":
            if state[v] != 1:
                return False
            state[v] = 0
            fast.discard(v)
        elif op == "compute":
            for p in dag[v]:
                if state[p] != 1:
                    return False
            if len(fast) >= budget:
                return False
            state[v] = 1
            fast.add(v)
        elif op == "delete":
            if state[v] != 1:
                return False
            state[v] = 0
            fast.discard(v)
        else:
            return False
            
    _, children = _topo_order(dag)
    outputs = [v for v in vertices if len(children[v]) == 0]
    for v in outputs:
        if state[v] != 0:
            return False
    return True


def _simple_greedy(dag, budget):
    """Cleaner greedy: red = in fast memory, blue = not in memory.
    Process topologically. Load inputs, compute, evict when full."""
    vertices = _get_vertices(dag)
    order, children = _topo_order(dag)
    
    inputs = [v for v in vertices if len(dag[v]) == 0]
    outputs = [v for v in vertices if len(children[v]) == 0]
    outputs_set = set(outputs)
    
    remaining = {v: 0 for v in vertices}
    for v in vertices:
        for p in dag[v]:
            remaining[p] += 1
    
    state = {v: 0 for v in vertices}
    fast = set()
    ops = []
    
    def make_room():
        if len(fast) < budget:
            return
        best_victim = None
        best_score = float('inf')
        for u in fast:
            score = remaining[u]
            if u in outputs_set and remaining[u] == 0:
                score = -1
            if score < best_score:
                best_score = score
                best_victim = u
        if best_victim is not None:
            if best_victim in outputs_set and remaining[best_victim] == 0:
                ops.append(["store", best_victim])
            else:
                ops.append(["delete", best_victim])
            state[best_victim] = 0
            fast.discard(best_victim)
            
    for v in order:
        if len(dag[v]) == 0:
            if state[v] == 0:
                make_room()
                ops.append(["load", v])
                state[v] = 1
                fast.add(v)
            remaining[v] -= 1
            continue
        
        for p in dag[v]:
            if state[p] == 0:
                make_room()
                ops.append(["load", p])
                state[p] = 1
                fast.add(p)
        
        make_room()
        ops.append(["compute", v])
        state[v] = 1
        fast.add(v)
        
        for p in dag[v]:
            remaining[p] -= 1
            
    for v in outputs:
        if state[v] == 1:
            ops.append(["store", v])
            state[v] = 0
            fast.discard(v)
            
    transfers = sum(1 for op, _ in ops if op in ("load", "store"))
    return transfers, ops


def pebble(dag, fast_memory_budget):
    """Main entry point. Returns list of [operation, vertex]."""
    naive_transfers, naive_ops = _naive_schedule(dag, fast_memory_budget)
    
    if _time_left() > 5.0:
        try:
            greedy_transfers, greedy_ops = _simple_greedy(dag, fast_memory_budget)
            if greedy_transfers < naive_transfers and _simulate(dag, greedy_ops, fast_memory_budget):
                return greedy_ops
        except Exception:
            pass
            
    return naive_ops
