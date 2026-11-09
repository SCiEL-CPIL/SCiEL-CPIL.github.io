import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Candidate moduli: All even numbers from 2 to 128
    # This is the largest set of pairwise non-coprime integers in [2, 128]
    candidates = list(range(2, 129, 2))
    n = len(candidates)
    
    # Precompute gcds between all pairs
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(candidates[i], candidates[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Heuristic ordering: 
    # We want to assign residues to moduli in an order that prunes the search space early.
    # Moduli with smaller values have smaller domains, so they are more constrained.
    # Let's sort indices by modulus value.
    # indices = list(range(n))
    # indices.sort(key=lambda i: candidates[i])
    
    # Alternatively, we can try to solve it directly. 
    # Given the structure (all even), let's see if a simple greedy works with backtracking.
    
    assignment = [-1] * n  # assignment[i] = residue for candidates[i]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == n:
            return True
            
        m = candidates[idx]
        
        # Try residues 0 to m-1
        # Optimization: We can check validity against all previously assigned moduli (0 to idx-1)
        for r in range(m):
            valid = True
            for prev_idx in range(idx):
                prev_m = candidates[prev_idx]
                prev_r = assignment[prev_idx]
                g = gcd_table[idx][prev_idx]
                if (r - prev_r) % g == 0:
                    valid = False
                    break
            if valid:
                assignment[idx] = r
                if backtrack(idx + 1):
                    return True
                assignment[idx] = -1
                
        return False

    # Try the static order (increasing modulus)
    # This is a depth-first search.
    # If it fails, we might need randomization or a different order.
    # But for even numbers, there is likely a solution.
    # Let's add a small randomization to the residue order if the first attempt fails or takes too long?
    # Actually, the static order is deterministic. If it finds a solution, great.
    # If it doesn't, we might need to restart with a different order.
    
    # Let's try multiple random orders of the variables as well, just in case.
    best_set = []
    
    # Attempt 1: Static order (smallest to largest)
    if backtrack(0):
        best_set = [[assignment[i], candidates[i]] for i in range(n)]
        return best_set
    
    # If static order failed (unlikely but possible due to time or dead end), try random restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Shuffle the order of variables
        indices = list(range(n))
        random.shuffle(indices)
        
        # Reset assignment
        assignment = [-1] * n
        
        # We need to map the shuffled indices to the gcd table correctly.
        # The backtrack function assumes index i corresponds to candidates[i].
        # So we need a version that respects the permutation.
        
        def backtrack_perm(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
                
            orig_idx = indices[idx]
            m = candidates[orig_idx]
            
            for r in range(m):
                valid = True
                for prev_idx in range(idx):
                    prev_orig_idx = indices[prev_idx]
                    prev_m = candidates[prev_orig_idx]
                    prev_r = assignment[prev_orig_idx]
                    g = gcd_table[orig_idx][prev_orig_idx]
                    if (r - prev_r) % g == 0:
                        valid = False
                        break
                if valid:
                    assignment[orig_idx] = r
                    if backtrack_perm(idx + 1):
                        return True
                    assignment[orig_idx] = -1
                    
            return False
        
        if backtrack_perm(0):
            best_set = [[assignment[i], candidates[i]] for i in range(n)]
            return best_set
            
    return best_set
