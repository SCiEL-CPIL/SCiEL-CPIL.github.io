def local_search(S, deadline):
    current_S = set(S)
    current_size = len(current_S)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Add
        for x in range(1, N + 1):
            if x not in current_S and is_valid_add(x, current_S):
                current_S.add(x)
                current_size += 1
                improved = True
                break # Restart add phase
        
        if improved:
            continue

        # Phase 2: Swap 1 out, 2 in
        # Full pass over all x in S
        items = list(current_S)
        random.shuffle(items)
        
        for x in items:
            if time.time() > deadline:
                break
            
            current_S.remove(x)
            
            valid_adds = []
            for y in range(1, N + 1):
                if y not in current_S and is_valid_add(y, current_S):
                    valid_adds.append(y)
                    if len(valid_adds) >= 2:
                        break # We only need 2
            
            if len(valid_adds) >= 2:
                current_S.add(valid_adds[0])
                current_S.add(valid_adds[1])
                current_size += 1
                improved = True
                break # Restart loop
            else:
                current_S.add(x)
                
        if not improved:
            break
            
    return current_S
