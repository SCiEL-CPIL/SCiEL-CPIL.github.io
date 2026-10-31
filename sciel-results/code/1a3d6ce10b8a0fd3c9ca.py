    import numpy as np
    import random
    import time

    N = 50

    def construct():
        best_S = []
        best_size = 0
        
        # Time limit for search
        start_time = time.time()
        time_limit = 60.0 # Leave some buffer for 120s limit
        
        # Number of restarts
        max_restarts = 1000
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # 1. Greedy Construction
            S = []
            used_sums = set()
            candidates = list(range(N))
            random.shuffle(candidates)
            
            for i in candidates:
                new_sums = [i + i]
                for j in S:
                    new_sums.append(i + j)
                    
                if any(s in used_sums for s in new_sums):
                    continue
                    
                S.append(i)
                used_sums.update(new_sums)
                
            # 2. Local Search: Try to grow or swap to enable growth
            # We will iterate a few times
            improved = True
            while improved:
                improved = False
                
                # Check if we can add any element
                for y in range(N):
                    if y in S:
                        continue
                    
                    new_sums = [y + y]
                    conflict = False
                    for x in S:
                        new_sums.append(x + y)
                        if (x + y) in used_sums:
                            conflict = True
                            break
                        if (y + y) in used_sums: # Check self-sum once
                            conflict = True
                            break
                    
                    # Re-check properly
                    if not conflict:
                        # Verify all sums
                        valid = True
                        # Self sum
                        if (y+y) in used_sums:
                            valid = False
                        else:
                            for x in S:
                                if (x+y) in used_sums:
                                    valid = False
                                    break
                        
                        if valid:
                            S.append(y)
                            used_sums.add(y+y)
                            for x in S[:-1]:
                                used_sums.add(x+y)
                            improved = True
                            break # Restart inner loop to check others
                
                if not improved:
                    # Try swaps
                    # Find a swap (x in S, y not in S) such that S' is valid
                    # Heuristic: Prefer y that has fewer conflicts with S \ {x}
                    best_swap = None
                    min_conflicts = float('inf')
                    
                    S_set = set(S)
                    for x in S:
                        S_minus_x = S_set - {x}
                        # Rebuild used_sums for S_minus_x? Expensive.
                        # Instead, just check validity of S' directly.
                        for y in range(N):
                            if y in S_set:
                                continue
                            
                            S_prime = (S_set - {x}) | {y}
                            # Check if S_prime is Sidon
                            if is_sidon(S_prime):
                                # Count conflicts of y with S_minus_x to guide future
                                # Actually, just take the first valid swap?
                                # Or one that maximizes potential?
                                # Let's just take a random valid swap to escape local minimum
                                # But we need to ensure we don't cycle.
                                # Simple strategy: Take the swap that results in the smallest sum of indices?
                                # Or just break.
                                best_swap = (x, y)
                                break
                            # Optimization: If we find many, pick one
                            # But is_sidon is O(k^2). k~8. 64 ops.
                            # 50 * 8 * 50 = 20000 checks per local step.
                            # This is fast.
                        
                        if best_swap:
                            break
                        # If no swap found for this x, try next x?
                        # We should check all x.
                    
                    if best_swap:
                        x, y = best_swap
                        S.remove(x)
                        S.append(y)
                        # Rebuild used_sums
                        used_sums = set()
                        for i in S:
                            used_sums.add(i+i)
                            for j in S:
                                if i <= j:
                                    used_sums.add(i+j)
                        # Actually, just recompute is_sidon style sums
                        # And set improved = True to try growing again
                        improved = True
                        # Break out of while to restart growth check?
                        # Yes, swap changed the set, so growth might be possible now.
                        continue 
                    
                    # If no swap found, we are stuck
                    break

            if len(S) > best_size:
                best_size = len(S)
                best_S = S

        if not best_S:
            best_S = [0]
            
        h = np.zeros(N)
        for i in best_S:
            h[i] = 1.0 / len(best_S)
            
        return list(h)

    def is_sidon(S):
        sums = set()
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                s = S_list[i] + S_list[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    