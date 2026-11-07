import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    # Precompute gcd? Not needed for this approach.
    
    # We search for minimal W
    # W must be at least n-1 = 9
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    for W in range(9, 1000):
        if time.time() > deadline:
            break
            
        # Precompute position lists for this W
        # pos_lists[i][r] is a list of positions k in [0, W] such that k % divisors[i] == r
        pos_lists = []
        for i in range(n):
            d = divisors[i]
            # r ranges from 0 to d-1
            # But we only care about residues that appear in [0, W]
            # Actually, we can just use a list of size d, filled with empty lists
            buckets = [[] for _ in range(d)]
            for k in range(W + 1):
                buckets[k % d].append(k)
            pos_lists.append(buckets)
            
        # Now iterate S
        found = False
        best_S = -1
        best_assignment = None
        
        for S in range(LCM):
            if time.time() > deadline:
                found = True # Break outer? No, just break inner and return current best?
                # If we break, we might not have a solution.
                # We should ensure we return a valid solution.
                # If we run out of time, we fall back to the trivial solution.
                break
                
            # Get the valid positions for each divisor
            current_sets = []
            for i in range(n):
                d = divisors[i]
                residue = (-S) % d
                current_sets.append(pos_lists[i][residue])
                
            # Check if SDR exists
            # Sort indices by set size for MRV
            indices = list(range(n))
            indices.sort(key=lambda i: len(current_sets[i]))
            
            # If any set is empty, skip
            if any(len(current_sets[i]) == 0 for i in indices):
                continue
                
            # DFS
            used = set()
            assignment = [None] * n
            
            def dfs(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for pos in current_sets[i]:
                    if pos not in used:
                        used.add(pos)
                        assignment[i] = pos
                        if dfs(idx + 1):
                            return True
                        used.remove(pos)
                return False
            
            if dfs(0):
                best_S = S
                best_assignment = assignment[:]
                found = True
                break
        
        if found and best_S != -1:
            # Construct result
            # x_i = best_S + best_assignment[i]
            # Ensure positive
            vals = [best_S + best_assignment[i] for i in range(n)]
            min_val = min(vals)
            if min_val < 1:
                # Shift by LCM
                shift = (1 - min_val + LCM - 1) // LCM
                if shift < 0:
                    shift = 0
                else:
                    vals = [v + shift * LCM for v in vals]
            
            # Check distinctness and divisibility (sanity check)
            # They are distinct by construction (used set)
            # Divisibility: best_S + best_assignment[i] is divisible by divisors[i]
            # (best_S + best_assignment[i]) % divisors[i] == (best_S % d + best_assignment[i] % d) % d
            # best_assignment[i] % d == (-best_S) % d
            # Sum == 0. Correct.
            
            return [int(v) for v in vals]
            
    # Fallback
    reps = [int((i + 1) * LCM) for i in range(n)]
    return reps
