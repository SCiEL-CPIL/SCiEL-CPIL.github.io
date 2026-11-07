"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    all_lcm = reduce(lcm, divisors)  # 30030
    
    deadline = time.time() + 85  # 85s safety margin (90s limit)
    
    best_result = None
    best_L = float('inf')
    
    # Precompute: for each divisor d_i, the list of residues r in [0, d_i) 
    # that are "good" for a given span L.
    # Good if (d_i - r) % d_i <= L, i.e., r == 0 or r >= d_i - L (if L < d_i).
    # If L >= d_i, all residues are good.
    
    # Strategy:
    # 1. Iterate L from a lower bound upwards.
    # 2. For each L, determine the set of "bad" residues for each divisor.
    # 3. Use a bitmask or similar to quickly filter A values in [0, all_lcm) 
    #    that are valid for all divisors (i.e., A mod d_i is good for all i).
    # 4. For valid A, compute the multiples and check distinctness.
    # 5. Since all_lcm is small (30030), we can afford to check all A for small L.
    #    However, checking all A for every L is O(L_max * all_lcm * n).
    #    500 * 30030 * 10 = 150M ops. This might be tight in Python.
    
    # Optimization:
    # Instead of checking all A, we can generate candidate A's more efficiently.
    # Or, we can note that the "good" set for a divisor d is a union of intervals 
    # in the residue space.
    
    # Let's try a different structural approach:
    # Represent the problem as finding a window [A, A+L] such that for each d_i,
    # there is a multiple in the window, and all are distinct.
    
    # For a fixed L, the condition "A is valid for d" depends on A mod d.
    # Let's precompute a boolean array `valid_A` of size all_lcm, initially all True.
    # For each divisor d_i, mark as False all A in [0, all_lcm) such that A mod d_i is bad.
    # Then, for each remaining A, check distinctness.
    
    # This is O(all_lcm * n) per L, which is 30030 * 10 = 300k ops per L.
    # For 500 L's, that's 150M ops. Still tight.
    
    # Can we reduce the number of L's we check?
    # The lower bound for L is at least the maximum gap between consecutive "required" points.
    # But we don't know the points yet.
    
    # Let's try to be smarter about the lower bound.
    # The minimum span is at least the maximum over i of (d_i - 1) ? No.
    # Actually, if we have 10 distinct numbers, the span is at least 9.
    # But the divisors constrain the positions.
    
    # Let's just iterate L from 10 to 500, but with a fast check.
    
    # We can speed up the "mark bad A" step by using the fact that the pattern 
    # repeats every d_i.
    # For a fixed d, the bad residues are {1, 2, ..., d-L-1} if L < d.
    # So A is bad if A mod d in {1, ..., d-L-1}.
    # This is equivalent to A mod d not in {0} U {d-L, ..., d-1}.
    
    # Let's precompute the bad mask for each d and L? No, L changes.
    
    # Alternative: For each A in [0, all_lcm), compute A mod d_i for all i.
    # This is O(all_lcm * n) per L.
    
    # Let's try to optimize the inner loop.
    
    # Actually, let's try a different approach:
    # For each L, find all A in [0, all_lcm) such that for all i, A mod d_i is good.
    # This is a system of congruences. We can use the Chinese Remainder Theorem 
    # or a simple filter.
    
    # Since all_lcm is small, let's just do the filter.
    
    # To speed up, we can check the divisors in order of increasing "selectivity".
    # The most selective divisor is the one with the largest bad set, i.e., the largest d_i.
    # So sort divisors by d_i descending.
    
    sorted_divs = sorted(divisors, reverse=True)
    # But we need to keep track of the original index for the result.
    # Let's just use the original order for the result, but check in sorted order.
    
    # Actually, let's just iterate A and check all divisors.
    
    # Let's try L from 10 to 500.
    
    for L in range(10, 500):
        if time.time() > deadline - 5:
            break
        
        # Precompute good/bad for each divisor for this L
        # bad_residues[i] = set of bad residues for d_i
        # Or just a function to check if r is bad.
        
        found = False
        for A in range(all_lcm):
            if time.time() > deadline - 3:
                break
            
            valid = True
            multiples = []
            for d in divisors:
                r = A % d
                # Check if r is good
                # Good if r == 0 or (L < d and r >= d - L)
                # Bad if 1 <= r <= d - L - 1 (when L < d)
                if L < d:
                    if 1 <= r <= d - L - 1:
                        valid = False
                        break
                # If L >= d, all residues are good.
                
                # Compute multiple
                if r == 0:
                    m = A
                else:
                    m = A + (d - r)
                multiples.append(m)
            
            if not valid:
                continue
            
            # Check distinctness
            if len(set(multiples)) == n:
                # Check positive
                if any(m <= 0 for m in multiples):
                    continue
                
                # Check max_val
                if any(m > max_val for m in multiples):
                    continue
                
                # Found a solution!
                best_L = L
                best_result = [int(m) for m in multiples]
                found = True
                break
        
        if found:
            break
    
    if best_result is None:
        # Fallback: try larger L or use a different strategy
        # For now, return the divisors themselves (valid but large span)
        # Actually, let's try L up to 1000
        for L in range(500, 1000):
            if time.time() > deadline - 2:
                break
            
            found = False
            for A in range(all_lcm):
                if time.time() > deadline - 1:
                    break
                
                valid = True
                multiples = []
                for d in divisors:
                    r = A % d
                    if L < d:
                        if 1 <= r <= d - L - 1:
                            valid = False
                            break
                    if r == 0:
                        m = A
                    else:
                        m = A + (d - r)
                    multiples.append(m)
                
                if not valid:
                    continue
                
                if len(set(multiples)) == n:
                    if any(m <= 0 for m in multiples) or any(m > max_val for m in multiples):
                        continue
                    best_L = L
                    best_result = [int(m) for m in multiples]
                    found = True
                    break
            
            if found:
                break
    
    if best_result is None:
        # Final fallback: divisors themselves
        best_result = [int(d) for d in divisors]
    
    return best_result
# EVOLVE-BLOCK-END
