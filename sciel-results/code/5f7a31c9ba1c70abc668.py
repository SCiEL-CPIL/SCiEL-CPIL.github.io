import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
    
    # Function to check if a valid assignment exists for a given window width W
    def check_width(W):
        # Sieve valid L's
        # valid_mask[i] is True if window [i, i+W] contains at least one multiple of every divisor
        valid_mask = [True] * lcm
        
        for d in divisors:
            # For a given d, L is invalid if (d - L % d) % d > W
            # i.e., L % d is in the "bad" set
            bad_residues = []
            for r in range(d):
                # Next multiple of d after L (inclusive) is at offset (d - r) % d
                # If offset is 0, multiple is L itself.
                # If offset > W, no multiple in [L, L+W]
                offset = (d - r) % d
                if offset > W:
                    bad_residues.append(r)
            
            if bad_residues:
                for r in bad_residues:
                    for L in range(r, lcm, d):
                        valid_mask[L] = False
        
        # Try to find a solution among valid L's
        for L in range(lcm):
            if not valid_mask[L]:
                continue
            
            # Build candidate sets for each divisor
            candidates = []
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                if k_min > k_max:
                    # This should not happen if valid_mask is True, but safety check
                    candidates.append([])
                else:
                    candidates.append([k * d for k in range(k_min, k_max + 1)])
            
            # Check for perfect matching
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            assigned = [None] * n
            used_values = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for v in candidates[i]:
                    if v not in used_values:
                        used_values.add(v)
                        assigned[i] = v
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(v)
                        assigned[i] = None
                return False
            
            if backtrack(0):
                return True, assigned
        
        return False, None

    # Iterate W from 0 upwards
    # Upper bound: The span cannot exceed LCM - 1 in the worst case?
    # Actually, we can always pick multiples that are far apart, but we want min span.
    # A safe upper bound is sum of divisors or something?
    # LCM is 30030.
    # Let's iterate up to LCM.
    for W in range(lcm + 1):
        found, sol = check_width(W)
        if found:
            return sol
            
    # Fallback
    return divisors[:]

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I need to wrap it in the specific format requested.
# The prompt asks to rewrite the program.
# The current program has P and construct.
# I will replace the content inside EVOLVE-BLOCK.

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

import math

def construct():
    divisors = P['divisors']
    n = len(divisors)
    
    # Calculate LCM
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
    
    # Function to check if a valid assignment exists for a given window width W
    def check_width(W):
        # Sieve valid L's
        valid_mask = [True] * lcm
        
        for d in divisors:
            bad_residues = []
            for r in range(d):
                offset = (d - r) % d
                if offset > W:
                    bad_residues.append(r)
            
            if bad_residues:
                for r in bad_residues:
                    for L in range(r, lcm, d):
                        valid_mask[L] = False
        
        for L in range(lcm):
            if not valid_mask[L]:
                continue
            
            candidates = []
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                if k_min > k_max:
                    candidates.append([])
                else:
                    candidates.append([k * d for k in range(k_min, k_max + 1)])
            
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            assigned = [None] * n
            used_values = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for v in candidates[i]:
                    if v not in used_values:
                        used_values.add(v)
                        assigned[i] = v
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(v)
                        assigned[i] = None
                return False
            
            if backtrack(0):
                return True, assigned
        
        return False, None

    for W in range(lcm + 1):
        found, sol = check_width(W)
        if found:
            return sol
            
    # Fallback
    return divisors[:]
# EVOLVE-BLOCK-END
