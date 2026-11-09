def pebble(dag, fast_memory_budget):
    # Parse dag (same as before)
    # ...
    
    in_fast = set()
    computed = set()
    operations = []
    remaining_children = {v: len(children[v]) for v in vertices}
    
    for v in topo_order:
        # Load parents if not in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                operations.append(['load', p])
                in_fast.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        
        # Update remaining children for parents
        for p in parents.get(v, []):
            remaining_children[p] -= 1
        
        # Store any vertex that is no longer needed and not an output
        # Do this proactively to free up fast memory
        for x in list(in_fast):
            if x not in outputs and remaining_children.get(x, 0) == 0:
                operations.append(['store', x])
                in_fast.discard(x)
        
        # If still over budget, store the least needed non-output
        while len(in_fast) > fast_memory_budget:
            # Find the vertex with the smallest remaining_children (but > 0)
            # or any non-output
            candidates = [x for x in in_fast if x not in outputs]
            if not candidates:
                break
            # Pick the one with smallest remaining_children
            to_store = min(candidates, key=lambda x: remaining_children.get(x, 0))
            operations.append(['store', to_store])
            in_fast.discard(to_store)
    
    # Store all remaining outputs
    for v in outputs:
        if v in in_fast:
            operations.append(['store', v])
            in_fast.discard(v)
    
    return operations
