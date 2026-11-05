import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # 50s limit for safety
    
    # Determine structure of dag
    if isinstance(dag, dict):
        vertices = list(dag.keys())
        parents = {v: dag[v] for v in vertices}
    elif isinstance(dag, list):
        vertices = range(len(dag))
        parents = {i: dag[i] for i in vertices}
    else:
        return []
        
    num_vertices = len(vertices)
    if num_vertices == 0:
        return []

    # Identify inputs (sources) and outputs (sinks)
    is_parent = set()
    for v in vertices:
        for p in parents[v]:
            is_parent.add(p)
            
    sources = [v for v in vertices if not parents[v]]
    sinks = [v for v in vertices if v not in is_parent]
    
    # Topological Sort
    in_degree = {v: len(parents[v]) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
            
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0) 
        topo_order.append(u)
        for c in children[u]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != num_vertices:
        return []

    index_map = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute next use
    next_use = {}
    for v in vertices:
        if children[v]:
            min_idx = min(index_map[c] for c in children[v])
            next_use[v] = min_idx
        else:
            next_use[v] = float('inf')

    fast_mem = set()
    operations = []
    
    def evict(fast_mem, exclude=None):
        if not fast_mem:
            return
        if exclude:
            candidates = [x for x in fast_mem if x not in exclude]
            if candidates:
                evict_v = max(candidates, key=lambda x: next_use[x])
                fast_mem.remove(evict_v)
                operations.append(['store', evict_v])
                return
        # If no candidates or exclude is empty, evict any
        evict_v = max(fast_mem, key=lambda x: next_use[x])
        fast_mem.remove(evict_v)
        operations.append(['store', evict_v])

    for v in topo_order:
        if time.time() > deadline:
            break
            
        if not parents[v]:
            if v not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evict(fast_mem)
                fast_mem.add(v)
                operations.append(['load', v])
        else:
            # Load parents
            for p in parents[v]:
                if p not in fast_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        evict(fast_mem, exclude=set(parents[v]))
                    fast_mem.add(p)
                    operations.append(['load', p])
            
            # Compute v
            if len(fast_mem) >= fast_memory_budget:
                evict(fast_mem, exclude=set(parents[v]))
            
            fast_mem.add(v)
            operations.append(['compute', v])
            
    # Ensure sinks are in fast_mem
    for s in sinks:
        if time.time() > deadline:
            break
        if s not in fast_mem:
            # Load parents if needed (they might have been evicted)
            for p in parents[s]:
                if p not in fast_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        evict(fast_mem, exclude=set(parents[s]))
                    fast_mem.add(p)
                    operations.append(['load', p])
            
            if len(fast_mem) >= fast_memory_budget:
                evict(fast_mem, exclude=set(parents[s]))
            
            fast_mem.add(s)
            # Since s was computed before and stored, we load it.
            operations.append(['load', s])

    return operations
# EVOLVE-BLOCK-END
