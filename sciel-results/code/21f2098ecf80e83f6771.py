import time
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # LCM of divisors: 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    LCM = 30030
    
    # Precompute LCM of each pair of divisors for conflict detection
    # Two offsets r_i and r_j (relative to L) conflict if they're equal
    # r_i ≡ -L (mod d_i) and r_j ≡ -L (mod d_j)
    # They're equal iff r ≡ -L (mod lcm(d_i, d_j))
    
    # For the backtracking, we need to check if two chosen offsets are distinct.
    # Since offsets are in [0, W] and W < LCM (for small W), distinctness is just numerical inequality.
    
    # Strategy: iterate W from 1 upward. For each W, iterate L from 0 to LCM-1.
    # For each (W, L), compute for each divisor the set of valid offsets.
    # Then use backtracking with MRV to assign distinct offsets.
    
    # Optimization: Precompute (-L) % d for all L and d? That's 30030 * 10 = 300300 values.
    # Actually, let's just compute on the fly but use a faster method.
    
    # Key optimization: For a given W, many L values will be invalid immediately.
    # An L is valid iff for every divisor d, (-L) % d <= W.
    # This is a strong filter.
    
    # Further optimization: We can precompute for each d, the set of residues r in [0, LCM)
    # such that r % d <= W. But W changes, so let's just filter on the fly.
    
    # Let's also try a different approach: instead of iterating all L, we can use
    # the fact that the pattern repeats every LCM. But we still need to check all L.
    
    # Another optimization: sort divisors by d (larger d means fewer candidates for small W)
    # and process them in that order in the backtracking.
    
    # Actually, the MRV heuristic (sort by number of candidates) is better.
    
    # Let's add a quick pre-check: for each divisor d, the minimum number of multiples
    # in any window of length W is floor(W/d) or ceil(W/d). If W < d-1, there might be
    # windows with no multiples. But we're checking each L individually, so this is handled.
    
    # Main loop
    for W in range(1, 100):
        if time.time() > deadline:
            # Fallback: return the divisors themselves (span = 39-6 = 33)
            return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
        
        # For this W, iterate over all possible L values
        for L in range(LCM):
            if time.time() > deadline:
                return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
            
            # Quick filter: check if each divisor has at least one multiple in [L, L+W]
            # This is equivalent to (-L) % d <= W for each d
            valid = True
            for d in divisors:
                if (-L) % d > W:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute offset sets for each divisor
            offset_sets = []
            for d in divisors:
                r0 = (-L) % d
                offsets = []
                r = r0
                while r <= W:
                    offsets.append(r)
                    r += d
                offset_sets.append(offsets)
            
            # Sort by number of offsets (MRV - Minimum Remaining Values)
            indices = list(range(n))
            indices.sort(key=lambda i: len(offset_sets[i]))
            
            # Backtracking with forward checking
            choices = [None] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                i = indices[idx]
                for off in offset_sets[i]:
                    if off not in used:
                        used.add(off)
                        choices[i] = off
                        if backtrack(idx + 1):
                            return True
                        choices[i] = None
                        used.remove(off)
                
                return False
            
            if backtrack(0):
                # Found a valid assignment
                result = []
                for i in range(n):
                    x = L + choices[i]
                    result.append(int(x))
                
                # Verify
                assert len(set(result)) == n, f"Result not distinct: {result}"
                assert all(result[i] % divisors[i] == 0 for i in range(n)), f"Divisibility check failed"
                assert all(0 < x <= 1000000 for x in result), f"Range check failed"
                actual_span = max(result) - min(result)
                assert actual_span <= W, f"Span check failed: {actual_span} > {W}"
                
                return result
    
    # Fallback
    return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
