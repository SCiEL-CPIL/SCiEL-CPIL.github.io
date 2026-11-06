def pebble(dag, fast_memory_budget):
    # Extract DAG structure
    nodes = list(dag.nodes) if hasattr(dag, 'nodes') else list(dag)
    parents = dag.parents if hasattr(dag, 'parents') else (dag.get('parents', {}) if isinstance(dag, dict) else {})
    outputs = set(dag.outputs) if hasattr(dag, 'outputs') else (set(dag.get('outputs', [])) if isinstance(dag, dict) else set())
    topo = list(dag.topo_order) if hasattr(dag, 'topo_order') else (list(dag.get('topo_order', nodes)) if isinstance(dag, dict) else nodes)
    
    # Build children map
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            if p in children:
                children[p].append(n)
    
    # Compute layer (longest path from source)
    layer = {}
    for n in topo:
        ps = parents.get(n, [])
        if not ps:
            layer[n] = 0
        else:
            layer[n] = max(layer[p] for p in ps) + 1
    
    # Determine when each node is last needed
    last_needed = {}
    for n in reversed(topo):
        if not children[n]:
            last_needed[n] = topo.index(n)
        else:
            last_needed[n] = max(last_needed[c] for c in children[n])
    
    # Initial state
    inputs = [n for n in nodes if not parents.get(n, [])]
    fast_mem = set(inputs[:fast_memory_budget])
    computed = set(fast_mem)
    operations = []
    
    # Process in topological order
    for node in topo:
        if node in computed:
            continue
        
        # Ensure parents are in fast memory
        for p in parents.get(node, []):
            if p not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    # Evict: prefer nodes not needed soon
                    candidates = list(fast_mem)
                    # Score: higher last_needed = less urgent
                    victim = max(candidates, key=lambda x: last_needed.get(x, 0))
                    operations.append(['store', victim])
                    fast_mem.remove(victim)
                
                operations.append(['load', p])
                fast_mem.add(p)
        
        # Compute
        operations.append(['compute', node])
        fast_mem.add(node)
        computed.add(node)
        
        # Delete parents no longer needed
        for p in parents.get(node, []):
            if p in fast_mem:
                if all(c in computed for c in children[p]):
                    operations.append(['delete', p])
                    fast_mem.remove(p)
    
    # Ensure outputs in fast memory
    for out in outputs:
        if out not in fast_mem:
            if len(fast_mem) >= fast_memory_budget:
                candidates = [x for x in fast_mem if x not in outputs]
                if candidates:
                    victim = max(candidates, key=lambda x: last_needed.get(x, 0))
                    operations.append(['store', victim])
                    fast_mem.remove(victim)
            operations.append(['load', out])
            fast_mem.add(out)
    
    return operations
