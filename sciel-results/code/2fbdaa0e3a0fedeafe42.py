def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0
    
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    # Precompute children
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children_map[p].append(v)
            
    # remaining_children[v] = number of children of v that are NOT yet computed
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    in_memory = set()
    ops = []
    
    for v in order:
        if time.time() > deadline:
            # If we run out of time, we should probably return what we have or a fallback.
            # But the loop is fast.
            break

        # 1. Ensure parents are in memory
        for p in parents.get(v, []):
            if p not in in_memory:
                # p is in slow memory (input or stored output)
                ops.append(['load', p])
                in_memory.add(p)
                
        # 2. Compute v
        ops.append(['compute', v])
        in_memory.add(v)
        
        # 3. Update remaining_children for parents
        for p in parents.get(v, []):
            remaining_children[p] -= 1
            
        # 4. Handle outputs
        if v in outputs:
            # If v has no more uncomputed children, we can store it now
            if remaining_children[v] == 0:
                ops.append(['store', v])
                in_memory.remove(v)
            # Else, keep in memory. It will be stored later when its last child is computed.
            
        # 5. Memory Management
        while len(in_memory) > fast_memory_budget:
            # Try to delete a non-output vertex that is no longer needed
            candidate = None
            for u in in_memory:
                if u not in outputs and remaining_children[u] == 0:
                    candidate = u
                    break
            
            if candidate:
                ops.append(['delete', candidate])
                in_memory.remove(candidate)
            else:
                # Try to store an output that is no longer needed (should have been handled above, but just in case)
                candidate_out = None
                for u in in_memory:
                    if u in outputs and remaining_children[u] == 0:
                        candidate_out = u
                        break
                        
                if candidate_out:
                    ops.append(['store', candidate_out])
                    in_memory.remove(candidate_out)
                else:
                    # Must store an output that still has uncomputed children.
                    # This is expensive (will require reload later) but necessary to fit in memory.
                    # Choose the one with the fewest remaining children? Or just the first one.
                    candidate_out = None
                    min_rem = float('inf')
                    for u in in_memory:
                        if u in outputs:
                            if remaining_children[u] < min_rem:
                                min_rem = remaining_children[u]
                                candidate_out = u
                                
                    if candidate_out:
                        ops.append(['store', candidate_out])
                        in_memory.remove(candidate_out)
                    else:
                        # Critical failure: only non-outputs with remaining children > 0 are in memory.
                        # This means we cannot free space.
                        # We might have to break or raise an error.
                        # For now, break to avoid infinite loop.
                        break
                        
    # 6. Finalize: Store any remaining outputs, delete any remaining non-outputs
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops
