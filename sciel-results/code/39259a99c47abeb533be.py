import time
import math
from itertools import product

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # LCM of divisors
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2*3*5*7*11*13 = 30030
    lcm_val = 30030
    
    # Precompute for each divisor, for each residue class mod lcm_val,
    # the multiples of that divisor in [0, lcm_val-1]
    # Actually, we need to think about this differently.
    
    # For a window [L, L+W], the multiples of d in this window are:
    # The first multiple of d >= L is m0 = ((L + d - 1) // d) * d
    # Then m0, m0+d, m0+2d, ... <= L+W
    
    # Key insight: The pattern of available multiples repeats with period lcm_val.
    # So we only need to check L in [0, lcm_val-1].
    
    # For each W (window width), we need to find if there exists L such that
    # we can assign distinct values to each divisor.
    
    # Strategy: For each W from 1 upward, for each L in [0, lcm_val-1]:
    #   - For each divisor d_i, compute the set of multiples in [L, L+W]
    #   - Check if there's a system of distinct representatives (SDR)
    #   - Use backtracking with constraint propagation for efficiency
    
    # Optimization: Pre-sort divisors by how "constrained" they are
    # (fewer candidates in a typical window = more constrained)
    
    # For a window of width W, divisor d has approximately W/d + 1 candidates
    # So larger divisors are more constrained
    
    # Let's try a smarter approach:
    # For each W, iterate over L values. For each L, build candidate lists.
    # Use Hall's theorem or backtracking to check for SDR.
    
    # Given the small size (10 divisors), backtracking should be fast.
    
    # Precompute: for each divisor index and each possible L mod lcm_val,
    # what are the candidate offsets? Actually, let's just compute on the fly
    # but with good pruning.
    
    best_val = float('inf')
    best_assign = None
    
    # Try W from 1 to some upper bound
    # Theoretical minimum is at least 9 (need 10 distinct integers, so span >= 9)
    # But due to divisibility constraints, it could be larger
    
    for W in range(9, 500):  # Start from theoretical minimum
        if time.time() > deadline:
            break
            
        for L in range(lcm_val):
            if time.time() > deadline:
                break
                
            # Build candidate lists for each divisor
            candidates = []
            valid = True
            for d in divisors:
                # First multiple of d >= L
                m = ((L + d - 1) // d) * d
                if m > L + W:
                    valid = False
                    break
                cand = []
                while m <= L + W:
                    cand.append(m)
                    m += d
                candidates.append(cand)
            
            if not valid:
                continue
            
            # Check for SDR using backtracking
            # Sort by number of candidates (most constrained first)
            # But we need to track original indices
            
            # Create list of (original_index, candidates)
            items = list(enumerate(candidates))
            items.sort(key=lambda x: len(x[1]))
            
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                orig_idx, cands = items[idx]
                for val in cands:
                    if val not in used:
                        used.add(val)
                        assignment[orig_idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[orig_idx] = None
                return False
            
            if backtrack(0):
                # Found a valid assignment
                # Verify
                if len(set(assignment)) != n:
                    continue
                ok = True
                for i in range(n):
                    if assignment[i] % divisors[i] != 0:
                        ok = False
                        break
                    if assignment[i] < 1 or assignment[i] > MAX_V:
                        ok = False
                        break
                if ok:
                    w_actual = max(assignment) - min(assignment)
                    if w_actual < best_val:
                        best_val = w_actual
                        best_assign = assignment[:]
                    # Since we're iterating W in increasing order,
                    # the first valid W gives us the minimum possible span
                    # (because any set with span R fits in a window of width R)
                    return [int(x) for x in best_assign]
    
    if best_assign is not None:
        return [int(x) for x in best_assign]
    
    # Fallback: return the divisors themselves
    return [int(x) for x in divisors]
