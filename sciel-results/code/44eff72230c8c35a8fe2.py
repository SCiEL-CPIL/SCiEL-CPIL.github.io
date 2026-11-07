"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import functools

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = functools.reduce(lcm, divisors, 1)  # 30030
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    best_result = None
    best_range = float('inf')
    
    # Precompute for each divisor, the residues mod period where multiples occur
    # For divisor d, multiples are at positions congruent to 0 mod d
    # In a window [L, L+R] (within one period), multiples of d are at L + offset
    # where (L + offset) % d == 0, i.e., offset % d == (-L) % d
    
    # For a given window [L, L+R] with L in [0, period-1]:
    # For divisor d, the multiples in the window are:
    #   k*d where L <= k*d <= L+R
    #   k ranges from ceil(L/d) to floor((L+R)/d)
    
    # Strategy: For each window size R (starting from minimum possible),
    # check all L positions using bitsets for fast SDR check.
    
    # Minimum possible range: at least 9 (10 distinct integers)
    # But also constrained by divisors. Let's start from R=9 and go up.
    
    # For the SDR check with bitsets:
    # Each candidate multiple m in [L, L+R] can be indexed by (m - L) in [0, R]
    # For each divisor i, we have a bitmask of which positions (0..R) have a valid multiple
    # We need to check if there's an assignment of distinct positions to divisors
    
    # Actually, let's use a different approach:
    # For each L, for each divisor d_i, compute the list of multiples in [L, L+R]
    # Then use a fast bipartite matching or Hall's theorem check
    
    # Better: use bitset representation where bit j is set if position j (0-indexed in window)
    # has a multiple of divisor i. Then we need to find a system of distinct representatives.
    # This is equivalent to checking if the bipartite graph has a perfect matching.
    
    # For small R (say < 200), we can use a simple backtracking with bitsets for speed.
    
    # Let's precompute: for each divisor index i and each possible L (0 to period-1),
    # and for a given R, the set of multiples. But R varies, so let's compute on the fly.
    
    # Optimize: for a fixed R, iterate over L from 0 to period-1.
    # For each L, for each divisor d, compute multiples in [L, L+R].
    # Use a fast feasibility check.
    
    # Precompute divisor info
    div_info = []
    for d in divisors:
        div_info.append(d)
    
    # Try window sizes from 9 upward
    for R in range(9, 500):
        if time.time() > deadline:
            break
        
        found = False
        
        for L in range(period):
            if time.time() > deadline:
                break
            
            # For each divisor, compute multiples in [L, L+R]
            candidates = []
            valid = True
            
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + R) // d
                
                if k_max < k_min:
                    valid = False
                    break
                
                # Store as list of (k, multiple) pairs
                cands = []
                for k in range(k_min, k_max + 1):
                    m = k * d
                    if m <= P['maximum']:
                        cands.append(m)
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Check SDR using backtracking with MRV and bitsets
            # Use bitset for used positions
            indices = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # For speed, if any candidate list is empty, skip
            if any(len(candidates[i]) == 0 for i in range(n)):
                continue
            
            # Backtracking with MRV
            used = set()
            assignment = [None] * n
            found_solution = False
            
            def backtrack(idx):
                nonlocal found_solution
                if found_solution:
                    return
                if idx == n:
                    found_solution = True
                    return
                d_idx = indices[idx]
                for m in candidates[d_idx]:
                    if m not in used:
                        used.add(m)
                        assignment[d_idx] = m
                        backtrack(idx + 1)
                        used.remove(m)
                        if found_solution:
                            return
                        assignment[d_idx] = None
            
            backtrack(0)
            
            if found_solution:
                actual_vals = [assignment[i] for i in range(n)]
                actual_range = max(actual_vals) - min(actual_vals)
                
                if actual_range < best_range:
                    best_range = actual_range
                    best_result = [int(v) for v in actual_vals]
                
                found = True
                break
        
        if found:
            return best_result
    
    if best_result is None:
        return [int(d) for d in divisors]
    
    return best_result

# EVOLVE-BLOCK-END
