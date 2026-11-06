def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0

    # Parse DAG
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', range(dag.get('n', 0))))
        parents = dag.get('parents', {})
        # Ensure parents are lists
        for k in parents:
            if not isinstance(parents[k], list):
                parents[k] = list(parents[k])
        inputs = set(dag.get('inputs', dag.get('sources', [])))
        outputs = set(dag.get('outputs', dag.get('sinks', [])))
    else:
        # Fallback parsing
        vertices = list(range(len(dag)))
        parents = {}
        inputs = set()
        outputs = set()
        for i, item in enumerate(dag):
            if isinstance(item, tuple) and len(item) == 2:
                parents[i] = list(item[1])
            elif isinstance(item, dict):
                parents[i] = list(item.get('parents', []))
            if not parents.get(i):
                inputs.add(i)
            # Heuristic for outputs: no children? We need children map.
        
        # Build children for output detection if not provided
        children = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                if p in children:
                    children[p].append(v)
        for v in vertices:
            if not children.get(v):
                outputs.add(v)

    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
        
    # If inputs/outputs were empty in dict case, re-derive
    if not inputs:
        inputs = set(v for v in vertices if not parents.get(v))
    if not outputs:
        outputs = set(v for v in vertices if not children.get(v))

    # Topological order
    # Assume vertices are already in topological order or sort them
    # Simple topo sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        # Sort queue to ensure deterministic order?
        u = queue.pop(0)
        topo_order.append(u)
        for c in children.get(u, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(vertices):
        # Cycle or error, just use original order
        topo_order = list(vertices)

    # Remaining uses
    remaining_uses = {v: len(children.get(v, [])) + (1 if v in outputs else 0) for v in vertices}
    
    fast_mem = set()
    operations = []
    
    def evict(victim):
        if victim in fast_mem:
            fast_mem.remove(victim)
            operations.append(('store', victim))
            
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
            
        # 1. Ensure parents are in fast_mem
        for p in parents.get(v, []):
            if p not in fast_mem:
                # Need to load p
                # If full, evict
                while len(fast_mem) >= fast_memory_budget:
                    # Find victim
                    # Prefer victims that are not parents of v
                    victims = [u for u in fast_mem if u not in parents.get(v, [])]
                    if not victims:
                        victims = list(fast_mem)
                    # Among victims, pick one with lowest remaining uses
                    # If remaining uses is 0, it's ideal
                    victims.sort(key=lambda u: remaining_uses.get(u, 0))
                    evict_v = victims[0]
                    evict(evict_v)
                
                fast_mem.add(p)
                operations.append(('load', p))
                
        # 2. Compute v
        # Ensure space for v
        if len(fast_mem) >= fast_memory_budget:
             # Evict something
             # We just loaded parents, so they are in fast_mem.
             # We can evict any parent if we have to, but we just used them?
             # No, compute uses them. After compute, we don't need them for THIS step.
             # But we might need them for OTHER children.
             # So evicting a parent is valid, we'll reload if needed.
             
             # Prefer non-parents
             victims = [u for u in fast_mem if u not in parents.get(v, [])]
             if not victims:
                 victims = list(fast_mem)
             victims.sort(key=lambda u: remaining_uses.get(u, 0))
             evict_v = victims[0]
             evict(evict_v)
             
        fast_mem.add(v)
        operations.append(('compute', v))
        
        # 3. Update remaining uses
        for p in parents.get(v, []):
            if p in remaining_uses:
                remaining_uses[p] -= 1
                
    # 4. Handle outputs
    for v in outputs:
        if time.time() - start_time > DEADLINE:
            break
        if v in fast_mem:
            fast_mem.remove(v)
            operations.append(('store', v))
        # If not in fast_mem, it's already blue (stored earlier or never loaded? 
        # If it was computed, it was added to fast_mem. If evicted, it was stored. 
        # So it is in slow memory. Correct.)
        
    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations
