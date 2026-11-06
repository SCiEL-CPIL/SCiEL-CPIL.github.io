def pebble(dag, fast_memory_budget):
    from collections import deque
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', list(dag.get('parents', {}).keys())))
        parents = dict(dag.get('parents', {}))
        inputs = list(dag.get('inputs', []))
        outputs = list(dag.get('outputs', []))
        topo_order = list(dag.get('topological_order', vertices))
    else:
        vertices = list(getattr(dag, 'vertices', []))
        parents = dict(getattr(dag, 'parents', {}))
        inputs = list(getattr(dag, 'inputs', []))
        outputs = list(getattr(dag, 'outputs', []))
        topo_order = list(getattr(dag, 'topological_order', vertices))
    
    # Compute topological order if not provided
    if not topo_order:
        in_degree = {v: len(parents.get(v, [])) for v in vertices}
        children = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                children[p].append(v)
        queue = deque([v for v in vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
    
    # Identify inputs and outputs if not provided
    if not inputs:
        inputs = [v for v in vertices if not parents.get(v, [])]
    if not outputs:
        children = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                children[p].append(v)
        outputs = [v for v in vertices if not children[v]]
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Process in topological order with smart eviction
    fast_memory = set()
    slow_memory = set(inputs)
    operations = []
    
    # remaining_children[x] = number of children of x not yet computed
    remaining_children = {v: len(children[v]) for v in vertices}
    
    for v in topo_order:
        # Ensure all parents of v are in fast_memory
        for p in parents.get(v, []):
            if p not in fast_memory:
                if p in slow_memory:
                    operations.append(['load', p])
                    fast_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # Update remaining_children for parents of v
        for p in parents.get(v, []):
            remaining_children[p] -= 1
            # If p is no longer needed and is an intermediate, delete it
            if remaining_children[p] == 0 and p not in outputs:
                if p in fast_memory:
                    operations.append(['delete', p])
                    fast_memory.remove(p)
        
        # If over budget, evict the least useful value
        while len(fast_memory) > fast_memory_budget:
            # Find the best value to evict
            best_evict = None
            best_score = -1
            for x in fast_memory:
                if x in outputs:
                    continue
                if x in inputs:
                    next_need = len(topo_order)
                    for c in children[x]:
                        if remaining_children[x] > 0:
                            idx = topo_order.index(c)
                            if idx > topo_order.index(v):
                                next_need = min(next_need, idx)
                    score = next_need
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_evict = x
            
            if best_evict is None:
                break
            
            if best_evict in inputs:
                operations.append(['store', best_evict])
                fast_memory.remove(best_evict)
                slow_memory.add(best_evict)
            else:
                operations.append(['delete', best_evict])
                fast_memory.remove(best_evict)
        
        # Safety: limit operations
        if len(operations) > 20000:
            break
    
    # Store all outputs
    for v in outputs:
        if v in fast_memory:
            operations.append(['store', v])
            fast_memory.remove(v)
            slow_memory.add(v)
    
    return operations
