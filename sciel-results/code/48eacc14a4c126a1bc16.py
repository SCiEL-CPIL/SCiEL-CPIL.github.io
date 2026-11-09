def pebble(dag, fast_memory_budget):
    # Extract DAG structure
    if isinstance(dag, dict):
        n = dag.get('n', None)
        parents = dag.get('parents', [])
        if n is None:
            n = len(parents)
        inputs = set(dag.get('inputs', dag.get('sources', [])))
        outputs = set(dag.get('outputs', dag.get('sinks', [])))
        if not inputs:
            inputs = set(i for i in range(n) if not parents[i])
        if not outputs:
            children = [set() for _ in range(n)]
            for v in range(n):
                for p in parents[v]:
                    children[p].add(v)
            outputs = set(i for i in range(n) if not children[i])
    else:
        parents = list(dag)
        n = len(parents)
        inputs = set(i for i in range(n) if not parents[i])
        children = [set() for _ in range(n)]
        for v in range(n):
            for p in parents[v]:
                children[p].add(v)
        outputs = set(i for i in range(n) if not children[i])
    
    # Build children
    children = [set() for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].add(v)
    
    # Fast memory: set of vertices currently held
    in_fast = set(inputs)
    computed = set(inputs)
    operations = []
    
    # Process in topological order
    for v in range(n):
        if v in inputs:
            continue
        
        # Load parents not in fast memory
        for p in parents[v]:
            if p not in in_fast and p in computed:
                operations.append(['load', p])
                in_fast.add(p)
        
        # Compute
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        
        # Prune: remove values no longer needed if over budget
        if len(in_fast) > fast_memory_budget:
            # Remove values that are no longer needed
            for p in list(in_fast):
                if p in inputs or p in outputs:
                    continue
                # Check if p is needed by any future vertex
                needed = False
                for c in children[p]:
                    if c > v:
                        needed = True
                        break
                if not needed:
                    operations.append(['delete', p])
                    in_fast.remove(p)
                    if len(in_fast) <= fast_memory_budget:
                        break
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in in_fast and v in computed:
            operations.append(['load', v])
            in_fast.add(v)
    
    return operations
