"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    # Precompute multiples for each divisor
    multiples = []
    for d in DIVISORS:
        mults = list(range(d, MAX_VAL + 1, d))
        multiples.append(mults)
    
    # Sort indices by number of multiples (ascending) - most constrained first
    # But we want to assign in order of constraint; for backtracking, assign largest divisors first
    # Actually, let's assign in order of fewest candidates in a typical window
    # For now, use original order but we can reorder in backtracking
    
    n = len(DIVISORS)
    
    # Try window sizes in increasing order
    # For each window size, try candidate L positions
    # Candidate L positions: near multiples of divisors
    
    # Generate candidate L values: for each multiple m, L can be in [m - W + 1, m]
    # So L candidates are near multiples
    
    best_result = None
    best_range = float('inf')
    
    # First, try a greedy/heuristic approach to get a good initial solution
    # Then refine
    
    # Let's try window sizes from 1 up to some reasonable limit
    # The answer is likely small (maybe < 200)
    
    for W in range(1, 500):
        if time.time() > deadline:
            break
            
        # For this window size W, find candidate L values
        # L must be such that [L, L+W] contains at least one multiple of each divisor
        # Candidate L: for each divisor i, L must be in union of [k*d_i - W, k*d_i] for all k
        # Let's generate candidate L values from multiples
        
        # For efficiency, generate candidate L from multiples of the largest divisors
        # (they have fewer multiples, so fewer candidates)
        
        candidate_Ls = set()
        # Add L values near multiples of each divisor
        for i, d in enumerate(DIVISORS):
            for m in multiples[i]:
                # L can be from max(1, m - W) to min(m, MAX_VAL - W)
                # But we just need to sample some
                # Add m - W and m as candidates (the window [m-W, m] and [m, m+W])
                lo = max(1, m - W)
                hi = min(m, MAX_VAL - W)
                if lo <= hi:
                    # Add a few candidates
                    candidate_Ls.add(lo)
                    candidate_Ls.add(hi)
                    if W > 0:
                        candidate_Ls.add(m - W // 2)
        
        # Limit candidate_Ls to avoid too many
        candidate_Ls = sorted(candidate_Ls)
        
        # For each candidate L, check if window [L, L+W] is feasible
        for L in candidate_Ls:
            if time.time() > deadline:
                break
                
            # Check if each divisor has at least one multiple in [L, L+W]
            feasible = True
            for i, d in enumerate(DIVISORS):
                # Find if any multiple of d is in [L, L+W]
                # Smallest multiple >= L: ceil(L/d)*d
                k = (L + d - 1) // d
                m = k * d
                if m > L + W:
                    feasible = False
                    break
            if not feasible:
                continue
            
            # Now do backtracking to assign distinct values
            # For each divisor, get candidates in [L, L+W]
            candidates = []
            for i, d in enumerate(DIVISORS):
                k_start = (L + d - 1) // d
                k_end = (L + W) // d
                cands = [k * d for k in range(k_start, k_end + 1)]
                candidates.append(cands)
            
            # Sort by number of candidates (ascending) for backtracking efficiency
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # Backtracking
            assigned = [None] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assigned[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assigned[i] = None
                return False
            
            if backtrack(0):
                result = [int(assigned[i]) for i in range(n)]
                rng = max(result) - min(result)
                if rng < best_range:
                    best_range = rng
                    best_result = result
                    if rng == 1:  # minimum possible for distinct integers
                        return best_result
                # Don't break; try other L for same W to see if we can get smaller
                # But if we found a solution for this W, we can stop searching for this W
                # and move to next W (but we're iterating W in increasing order, so this is the best for this W)
                # Actually, we want the minimum W, so once we find a solution for W, we can break
                break
        
        # If we found a solution for this W, we can return it (since we're trying W in increasing order)
        if best_result is not None and best_range == W:
            return best_result
    
    # If we didn't find a small window, return the best we have
    if best_result is not None:
        return best_result
    
    # Fallback: return the divisors themselves (which is the seed solution)
    return DIVISORS[:]
# EVOLVE-BLOCK-END
