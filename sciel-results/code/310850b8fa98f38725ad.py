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
            valid = True
            for d in divisors:
                # Ensure positive integers: k*d >= 1 => k >= 1
                # Also k*d >= L => k >= ceil(L/d)
                k_min = max(1, (L + d - 1) // d)
                k_max = (L + W) // d
                if k_min > k_max:
                    valid = False
                    break
                else:
                    # Values are k * d
                    candidates.append([k * d for k in range(k_min, k_max + 1)])
            
            if not valid:
                continue
                
            # Check for perfect matching
            # Sort indices by number of candidates to prune early
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            # If any divisor has 0 candidates, fail (handled by valid flag mostly, but check just in case)
            if any(len(candidates[i]) == 0 for i in indices):
                continue

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
    best_solution = None
    
    # Binary search on W
    # The predicate "exists valid matching for width W" is monotonic
    while lo < hi:
        mid = (lo + hi) // 2
        found, sol = check_width(mid)
        if found:
            best_solution = sol
            hi = mid
        else:
            lo = mid + 1
            
    # Check the final candidate
    if best_solution is None:
        found, sol = check_width(lo)
        if found:
            best_solution = sol

    if best_solution is None:
        # Fallback, though highly unlikely
        return divisors[:]
        
    return best_solution
