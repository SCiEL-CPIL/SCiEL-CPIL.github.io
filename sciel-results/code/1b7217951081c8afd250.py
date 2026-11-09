def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Get initial cycle
    initial_cycle = find_initial_cycle(deadline)
    if initial_cycle is None:
        return list(range(N))
    
    current_cycle = initial_cycle
    current_score = score_cycle(current_cycle)
    best_cycle = current_cycle[:]
    best_score = current_score
    
    # 2. Iterated Local Search
    while time.time() < deadline:
        # Hill Climbing Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(current_cycle)
            curr_max = current_score
            
            # Try all 2-opt moves (reversal)
            for i in range(n):
                if time.time() > deadline:
                    break
                u1 = current_cycle[i]
                v1 = current_cycle[(i + 1) % n]
                
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    if (j == (i + 1) % n) or (i == (j + 1) % n):
                        continue
                    
                    u2 = current_cycle[j]
                    v2 = current_cycle[(j + 1) % n]
                    
                    # Check if new edges (u1, u2) and (v1, v2) are promising
                    e1 = pos_matrix[u1][u2]
                    e2 = pos_matrix[v1][v2]
                    
                    if e1 <= curr_max and e2 <= curr_max:
                        # Apply reversal of segment i+1 to j
                        seg = current_cycle[i+1 : j+1]
                        seg.reverse()
                        new_cycle = current_cycle[:i+1] + seg + current_cycle[j+1:]
                        
                        new_score = score_cycle(new_cycle)
                        if new_score < curr_max:
                            current_cycle = new_cycle
                            current_score = new_score
                            improved = True
                            if new_score < best_score:
                                best_score = new_score
                                best_cycle = current_cycle[:]
                            # Break to restart the pass with new cycle structure
                            break
                    # Note: We do NOT check the "cross" option (u1,v2) and (v1,u2) 
                    # because it splits the cycle into two components.
                    
                if improved:
                    break # Restart outer loop
            
            if not improved:
                # Perturb Phase
                if time.time() > deadline:
                    break
                
                # Perform a random 2-opt move
                n = len(current_cycle)
                # Pick random i, j
                # Ensure i < j and not adjacent
                while True:
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i > j:
                        i, j = j, i
                    if j - i == 1:
                        continue
                    if i == 0 and j == n-1:
                        continue # Adjacent in cycle
                    
                    # Apply reversal
                    seg = current_cycle[i+1 : j+1]
                    seg.reverse()
                    new_cycle = current_cycle[:i+1] + seg + current_cycle[j+1:]
                    
                    current_cycle = new_cycle
                    current_score = score_cycle(current_cycle)
                    if current_score < best_score:
                        best_score = current_score
                        best_cycle = current_cycle[:]
                    break

    return best_cycle
