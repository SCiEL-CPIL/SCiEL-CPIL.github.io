"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    max_val = P['maximum']
    n = len(divisors)
    
    # Calculate LCM of divisors to know the period of the pattern
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    import functools
    period = functools.reduce(lcm, divisors, 1)
    
    deadline = time.time() + 55  # Safety margin
    
    best_result = None
    best_range = float('inf')
    
    # We search for increasing window sizes W
    # The optimal W is likely small. 
    # For each W, we only need to check L in [0, period-1]
    # If no solution is found for a W in one full period, no solution exists for that W.
    
    # Optimization: Precompute for each divisor d, the residues L mod d 
    # for which a multiple exists in [L, L+W].
    # Actually, it's easier to just check on the fly for small W.
    
    for W in range(1, 300):  # Try windows up to size 299
        if time.time() > deadline:
            break
            
        # For this W, iterate L over one period
        # We can break early if we find a solution, but we want the MINIMAL range.
        # Since we iterate W from 1 up, the first W for which we find ANY solution 
        # gives us a range <= W. 
        # However, the actual range of the found assignment might be less than W.
        # But if we find a solution for W, we can record it.
        # We should continue searching for smaller actual ranges?
        # No, if we find a solution with range R <= W, and we are at W, 
        # we know that no solution with range < W exists (because we checked all smaller W).
        # Wait, a solution with range R < W would have been found in a previous iteration 
        # where W' = R. 
        # So the first W for which we find a valid assignment gives us the optimal range?
        # Not exactly. The "range" of the assignment is max-min.
        # If we find an assignment in window [L, L+W], its range is <= W.
        # It is possible that for W=10, we find a solution with range 8.
        # But if there was a solution with range 7, it would have been found when W=7.
        # So the first W that yields a solution corresponds to the optimal range.
        
        found_solution = False
        
        # Iterate L. 
        # We don't necessarily need to check all L in [0, period) if we can prune.
        # But period is 30030, which is small enough.
        
        for L in range(period):
            if time.time() > deadline:
                break
                
            candidates = []
            valid = True
            
            for d in divisors:
                # Find multiples of d in [L, L+W]
                # k_min = ceil(L/d)
                # k_max = floor((L+W)/d)
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                
                if k_max < k_min:
                    valid = False
                    break
                
                # Generate candidates
                # To keep the list small, we just need the values
                # If k_max - k_min + 1 is large, the problem is easier, 
                # but for small W it is small.
                cands = [k * d for k in range(k_min, k_max + 1)]
                candidates.append(cands)
                
            if not valid:
                continue
                
            # Backtracking with MRV
            # Sort indices by number of candidates
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                d_idx = indices[idx]
                cands = candidates[d_idx]
                for m in cands:
                    if m not in used:
                        used.add(m)
                        assignment[d_idx] = m
                        if backtrack(idx + 1):
                            return True
                        used.remove(m)
                        assignment[d_idx] = None
                return False
            
            if backtrack(0):
                # Found a solution
                actual_vals = [assignment[i] for i in range(n)]
                actual_range = max(actual_vals) - min(actual_vals)
                
                # Since we are iterating W from 1 upwards, and we found a solution 
                # in window W, the actual_range must be <= W.
                # And since no solution was found for any W' < W, 
                # the optimal range is exactly actual_range?
                # Yes, because any solution with range R < W would have been 
                # contained in a window of size R, and thus found in an earlier iteration.
                
                best_range = actual_range
                best_result = [int(v) for v in actual_vals]
                found_solution = True
                break  # We found a solution for this W, so we can move to next W?
                # Actually, we can stop completely if we are confident.
                # But wait, is it possible that for the same W, there is a solution 
                # with a smaller range?
                # If we find a solution with range R, and we are at W, 
                # we know R <= W.
                # If R < W, it would have been found at W=R.
                # So if we are at W, and we found a solution, the range must be exactly W?
                # Not necessarily. Example: W=10. Solution found has range 5.
                # This implies there is a solution with range 5.
                # That solution should have been found when W=5.
                # So if we reached W=10, it means NO solution with range < 10 exists.
                # Therefore, the first solution found at W must have range exactly W.
                # (Or at least, we can stop searching for smaller ranges).
                
                # So we can return immediately?
                # Yes, because we are searching for the MINIMUM range.
                # The first W that yields a valid assignment contains the optimal range.
                return best_result
        
        if found_solution:
            return best_result
        
    if best_result is None:
        # Fallback
        return [int(d) for d in divisors]
    
    return best_result

# EVOLVE-BLOCK-END
