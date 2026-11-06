def pebble(dag, fast_memory_budget):
    vertices = dag['vertices']
    edges = dag['edges']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    children = {v: [] for v in vertices}
    parents = {v: [] for v in vertices}
    in_degree = {v: 0 for v in vertices}
    for p, c in edges:
        children[p].append(c)
        parents[c].append(p)
        in_degree[c] += 1
    
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Next use position
    next_use = {v: -1 for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents[v]:
            if i > next_use[p]:
                next_use[p] = i
    
    fast_memory = set()
    operations = []
    
    def evict_one():
        """Evict one vertex from fast memory to make room."""
        if not fast_memory:
            return
        # Prefer evicting non-outputs, then the one with farthest next use
        candidates = [u for u in fast_memory if u not in outputs]
        if not candidates:
            candidates = list(fast_memory)
        victim = max(candidates, key=lambda u: (next_use[u] if next_use[u] >= 0 else float('inf')))
        fast_memory.remove(victim)
        operations.append(['store', victim])
    
    for v in topo_order:
        if v in inputs:
            if v not in fast_memory:
                while len(fast_memory) >= fast_memory_budget:
                    evict_one()
                fast_memory.add(v)
                operations.append(['load', v])
        else:
            # Ensure all parents are in fast memory
            for p in parents[v]:
                if p not in fast_memory:
                    while len(fast_memory) >= fast_memory_budget:
                        evict_one()
                    fast_memory.add(p)
                    operations.append(['load', p])
            # Compute v
            while len(fast_memory) >= fast_memory_budget:
                evict_one()
            fast_memory.add(v)
            operations.append(['compute', v])
    
    return operations
