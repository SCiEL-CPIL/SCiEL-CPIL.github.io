while len(in_fast) > fast_memory_budget:
    candidate = None
    # 1. Find a non-output vertex in in_fast that has no unprocessed children
    for u in list(in_fast):
        if u in outputs: continue
        # Check if any child is unprocessed
        safe = True
        for c in children_map.get(u, []):
            if c not in processed:
                safe = False
                break
        if safe:
            candidate = u
            break
    
    if candidate is None:
        # 2. Pick any non-output vertex in in_fast
        for u in list(in_fast):
            if u not in outputs:
                candidate = u
                break
                
    if candidate is None:
        # 3. Pick any vertex (even output) - risky, but better than crash
        candidate = list(in_fast)[0]
        
    operations.append(['store', candidate])
    in_fast.discard(candidate)
    in_slow.add(candidate)
