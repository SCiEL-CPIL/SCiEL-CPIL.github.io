import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM to determine the period of the pattern
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
    
    # Function to check if a valid assignment exists for a given window width W
    def check_width(W):
        # We check all possible starting positions L in one period
        for L in range(lcm):
            # Build candidate sets for each divisor
            candidates = []
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                if k_min > k_max:
                    candidates.append([])
                else:
                    # Values are k * d
                    candidates.append([k * d for k in range(k_min, k_max + 1)])
            
            # Check for perfect matching
            # Sort indices by number of candidates to prune early
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
                # Return the solution
                return True, assigned
        return False, None

    # Binary search for the minimum W
    lo = 0
    hi = 1000  # Upper bound estimate
    best_w = hi
    best_solution = None
    
    # We can also just iterate W from 0 upwards since it's likely small
    # But binary search is safer.
    # Actually, let's just iterate W from 0 to 1000. It's only 1000 * 30030 * matching_cost.
    # If matching_cost is small, this is fine.
    # Let's try iterating W directly to avoid binary search overhead if W is very small.
    
    for W in range(1001):
        found, sol = check_width(W)
        if found:
            best_w = W
            best_solution = sol
            break
            
    if best_solution is None:
        # Fallback, though highly unlikely
        return divisors[:]
        
    return best_solution

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
        for L in range(lcm):
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

    for W in range(1001):
        found, sol = check_width(W)
        if found:
            return sol
            
    # Fallback
    return divisors[:]
# EVOLVE-BLOCK-END
