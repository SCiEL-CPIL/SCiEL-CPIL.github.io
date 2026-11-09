import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Calculate LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)
    
    # We search for the minimum span W.
    # The pattern repeats every lcm_all.
    # We iterate W from minimum possible (n-1=9) upwards.
    
    # Backtracking function using bitmasks for speed
    def find_assignment(candidates):
        # candidates is a list of lists of integers (absolute values)
        # We will work with relative indices in the window [L, L+W]
        # But candidates are passed as absolute values. 
        # It's better to pass relative indices or compute them inside.
        # Let's pass the window start L and width W, and the divisors.
        # And we compute candidates on the fly in the loop below.
        pass 

    # Helper to check if a solution is valid (for debugging/fallback)
    def is_valid(sol):
        if len(set(sol)) != n:
            return False
        for i, d in enumerate(divisors):
            if sol[i] % d != 0:
                return False
        if any(v < 1 or v > MAX_V for v in sol):
            return False
        return True

    # We will iterate W and L.
    # To speed up, we can precompute for each divisor d, the residues.
    # But simple iteration is fine for W < 100 and L < 270270.
    # 270270 * 100 = 27 million iterations. 
    # In each iteration, we do O(n) work to build candidates + backtracking.
    # Backtracking is very fast if W is small.
    # 27 million iterations in Python might be too slow (approx 10-20s per million simple ops).
    # We need to prune L.
    
    # Optimization: 
    # For a fixed W, we only need to check L such that the window [L, L+W] 
    # contains at least one multiple of each divisor.
    # Actually, we can just iterate L and check.
    
    # Let's try to be smarter.
    # The condition "contains at least one multiple of d" is equivalent to 
    # there exists k such that L <= k*d <= L+W.
    # This means L mod d is in [0, W] or [d - (W % d), d-1] if W >= d? 
    # No, it's simply that the distance from L to the nearest multiple of d 
    # (in either direction, but constrained to be within W forward) is small.
    # Specifically, let r = L % d.
    # The next multiple is at L + (d - r) % d.
    # We need (d - r) % d <= W.
    # So for each d_i, L must satisfy (d_i - L % d_i) % d_i <= W.
    
    # We can combine these constraints.
    # However, checking this for every L is still O(n) per L.
    # But we can precompute valid L ranges for each d_i?
    # The constraint is periodic with period d_i.
    
    # Let's just run the loop but with a tight inner loop.
    # If it's too slow, we can break early.
    
    best_sol = None
    best_W = float('inf')
    
    # Start W from 9
    for W in range(9, 200):
        if time.time() > deadline:
            break
            
        # Iterate L from 1 to lcm_all
        # We can step by 1.
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
                
            # Check bounds
            if L + W > MAX_V:
                # We could wrap around or just skip, but since MAX_V is huge 
                # and lcm is 270k, L+W is small.
                # But if L is close to MAX_V, we skip.
                # Actually, we can just start L from 1.
                pass

            # Build candidates and check feasibility
            candidates = []
            feasible = True
            
            for d in divisors:
                # First multiple >= L
                first_k = (L + d - 1) // d
                # Last multiple <= L + W
                last_k = (L + W) // d
                
                if first_k > last_k:
                    feasible = False
                    break
                    
                # Generate candidates
                # We store them as a list of ints
                cands = [k * d for k in range(first_k, last_k + 1)]
                candidates.append(cands)
                
            if not feasible:
                continue
                
            # Sort by number of candidates (most constrained first)
            # We need to keep track of original indices
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # Backtracking
            used = set()
            assignment = [0] * n
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                
                i = order[idx]
                # Try each candidate for this variable
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                # Found a solution
                sol = [int(v) for v in assignment]
                # Verify
                if is_valid(sol):
                    return sol
                    
    # Fallback
    # This should rarely be reached
    # Return the divisors themselves if they are distinct and valid
    # They are distinct and <= 1000000
    return [int(d) for d in divisors]
