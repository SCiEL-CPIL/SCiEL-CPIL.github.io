import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    start_time = time.time()
    deadline = start_time + 45.0  # 45s safety margin
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, divisors)
    
    # Precompute multiples in [0, L-1] for each divisor
    # multiples[i] will be a list of multiples of divisors[i] in [0, L-1]
    multiples = []
    for d in divisors:
        muls = []
        m = d
        while m < L:
            muls.append(m)
            m += d
        multiples.append(muls)
    
    n = len(divisors)
    
    # We search for minimal W
    # Theoretical min W is n-1 = 9
    best_solution = None
    
    for W in range(9, 1000): # Upper bound on W, should be small
        if time.time() > deadline:
            break
            
        # Iterate over possible start points M modulo L
        # We only need to check M in [0, L-1]
        for M in range(L):
            if time.time() > deadline:
                if best_solution:
                    break
                else:
                    # Fallback: return divisors themselves if no solution found in time (should not happen)
                    # But we need distinct. If W is large enough, it's always possible.
                    # If we hit deadline without a solution, we might need a heuristic.
                    # However, W=9..1000 should cover it.
                    pass
            
            # Collect candidates for each divisor
            # Candidate x must satisfy M <= x <= M+W and x % d_i == 0
            # Note: x can be >= L. But since we are looking for a window, 
            # we can just check values directly.
            # To be efficient, we check if there's a multiple in [M, M+W].
            # Let k be such that k*d_i is in [M, M+W].
            # k_min = ceil(M/d_i), k_max = floor((M+W)/d_i)
            
            candidates = []
            feasible = True
            for i, d in enumerate(divisors):
                cands = []
                k_min = (M + d - 1) // d
                k_max = (M + W) // d
                
                for k in range(k_min, k_max + 1):
                    x = k * d
                    # x is positive integer
                    if x <= max_val:
                        cands.append(x)
                
                if not cands:
                    feasible = False
                    break
                candidates.append(cands)
            
            if not feasible:
                continue
            
            # Try to find distinct representatives
            # Sort divisors by number of candidates (most constrained first)
            # indices sorted by len(candidates[i])
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # Backtracking
            assignment = [None] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for x in candidates[i]:
                    if x not in used:
                        used.add(x)
                        assignment[i] = x
                        if backtrack(idx + 1):
                            return True
                        used.remove(x)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                # Found a solution with width W
                # Construct result in original order
                result = [assignment[i] for i in range(n)]
                # Verify distinctness and divisibility
                if len(set(result)) == n and all(result[i] % divisors[i] == 0 for i in range(n)):
                    return result
                    # We return the first minimal W found
                else:
                    # Should not happen
                    pass
        
        # If no M worked for this W, try next W
        # We can add a small optimization: if W is large, M=0 is likely to work.
        # But we iterate all M.
        
    # If we exhausted W or time, return a valid (though maybe not optimal) solution
    # A trivial valid solution: x_i = i * L + d_i ? No, distinctness.
    # x_i = (i+1) * L. All distinct. Divisible? (i+1)*L is divisible by d_i since L is multiple of d_i.
    # Range: 10*L.
    fallback = [ (i+1) * L for i in range(n) ]
    return fallback
