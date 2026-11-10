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
    
    # Precompute GCDs to quickly check if two values can be equal (they can't if distinct,
    # but more useful: for a given offset, determine which residues are "compatible")
    
    # Key insight: two multiples k1*d1 and k2*d2 are equal iff k1*d1 ≡ k2*d2 (mod LCM)
    # and they are in the same window. Since all values are in [L, L+W] with W < LCM,
    # equality is just numerical equality.
    
    # For each divisor d_i, the multiples in [L, L+W] are at positions:
    #   L + r_i where r_i ≡ 0 (mod d_i) and 0 <= r_i <= W
    # i.e., r_i = m * d_i for m = 0, 1, ..., floor(W/d_i)
    # but only if (L + m*d_i) is divisible by d_i, i.e., L ≡ 0 (mod d_i) for m=0...
    # Wait, that's not right. Let me reconsider.
    
    # Actually, for a fixed L, the multiples of d in [L, L+W] are:
    #   x = k*d where L <= k*d <= L+W
    #   => ceil(L/d) <= k <= floor((L+W)/d)
    
    # The relative offset of x from L is x - L, which ranges from 0 to W.
    # x - L ≡ -L (mod d) for x to be a multiple of d.
    # So x - L = r where r ≡ -L (mod d) and 0 <= r <= W.
    
    # So for each divisor d_i, the set of relative offsets is:
    #   {r in [0, W] : r ≡ (-L) mod d_i}
    #   = {(-L) mod d_i + t * d_i : t >= 0, (-L) mod d_i + t*d_i <= W}
    
    # Two divisors d_i and d_j have a conflict at offset r if r is in both sets,
    # i.e., r ≡ -L (mod d_i) AND r ≡ -L (mod d_j), which means r ≡ -L (mod lcm(d_i, d_j)).
    
    # Strategy: iterate W from 1 upward. For each W, iterate L from 0 to LCM-1.
    # For each (W, L), compute for each divisor the set of valid offsets.
    # Then use backtracking to assign distinct offsets.
    
    # Optimization: precompute for each pair (i, j) the LCM of d_i and d_j.
    # If (-L) mod gcd(d_i, d_j) != (-L) mod gcd(d_i, d_j) [always true], no conflict.
    # Actually, the conflict condition is: there exists r in [0,W] such that
    # r ≡ -L (mod d_i) and r ≡ -L (mod d_j).
    # This is equivalent to: -L ≡ r (mod d_i) and -L ≡ r (mod d_j)
    # => r ≡ -L (mod lcm(d_i, d_j))
    # So a conflict exists iff there's a multiple of lcm(d_i, d_j) in the intersection
    # of the two offset sets.
    
    # Simpler approach: just compute the offset sets and check for overlaps.
    
    # Further optimization: for small W, each divisor has at most floor(W/d_i) + 1 offsets.
    # For W < 6, only divisor 6 can have 2 offsets, all others have at most 1.
    # For W < 10, divisors 6 and 10 can have 2 offsets.
    
    # Let's use a more efficient representation:
    # For each divisor d_i, the offsets are: r0_i, r0_i + d_i, r0_i + 2*d_i, ...
    # where r0_i = (-L) % d_i, and we keep those <= W.
    
    # Backtracking with MRV (Minimum Remaining Values) heuristic.
    
    for W in range(1, 100):
        if time.time() > deadline:
            return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
        
        for L in range(LCM):
            if time.time() > deadline:
                return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
            
            # Compute offset sets for each divisor
            offset_sets = []
            valid = True
            
            for d in divisors:
                r0 = (-L) % d
                offsets = []
                r = r0
                while r <= W:
                    offsets.append(r)
                    r += d
                if not offsets:
                    valid = False
                    break
                offset_sets.append(offsets)
            
            if not valid:
                continue
            
            # Sort by number of offsets (MRV)
            indices = list(range(n))
            indices.sort(key=lambda i: len(offset_sets[i]))
            
            # Backtracking
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
                assert max(result) - min(result) == W, f"Span check failed: {max(result)-min(result)} != {W}"
                
                return result
    
    # Fallback
    return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
