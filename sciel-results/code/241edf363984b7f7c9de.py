import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    # We search for the minimum range W
    # The solution must exist within one period L
    # We iterate W from 1 upwards
    
    for W in range(1, 10000):  # Upper bound for W
        if time.time() > deadline:
            # Fallback: return the divisors themselves (valid but not minimal)
            # Or a known good solution if we found one earlier
            # Since we iterate W increasing, if we time out, we might not have found the optimal.
            # But W is expected to be small.
            break
            
        # For each starting position A in [1, L]
        for A in range(1, L + 1):
            if time.time() > deadline:
                break
                
            # Check if we can pick distinct multiples
            # S_i is the set of multiples of divisors[i] in [A, A+W]
            sets = []
            valid = True
            sizes = []
            for i in range(n):
                d = divisors[i]
                # Multiples of d in [A, A+W]
                # k*d >= A => k >= ceil(A/d)
                # k*d <= A+W => k <= floor((A+W)/d)
                k_min = (A + d - 1) // d
                k_max = (A + W) // d
                
                if k_min > k_max:
                    valid = False
                    break
                    
                # Generate the actual values
                # Since W is small, the number of multiples is small
                vals = [k * d for k in range(k_min, k_max + 1)]
                sets.append(vals)
                sizes.append(len(vals))
                
            if not valid:
                continue
                
            # Optimization: Sort indices by the size of their candidate sets
            # This prunes the backtracking search space significantly
            indices = list(range(n))
            indices.sort(key=lambda i: sizes[i])
            
            used = set()
            assignment = [None] * n
            
            # Backtrack on sorted indices
            def backtrack(idx):
                if idx == n:
                    return True
                if time.time() > deadline:
                    return False
                orig_idx = indices[idx]
                for val in sets[orig_idx]:
                    if val not in used:
                        used.add(val)
                        assignment[orig_idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[orig_idx] = None
                return False

            if backtrack(0):
                # We found a valid assignment
                # Ensure values are positive and <= 1000000
                # A is in [1, L], W is small, so max val is ~30000 + 10000 < 1000000
                # Min val is >= 1
                result = [int(assignment[i]) for i in range(n)]
                # Verify distinctness and divisibility just in case
                if len(set(result)) == n:
                    for i in range(n):
                        if result[i] % divisors[i] != 0:
                            raise ValueError("Logic error")
                        if not (1 <= result[i] <= 1000000):
                            raise ValueError("Out of bounds")
                return result
    
    # Fallback if no small W found (should not happen)
    # Return the divisors themselves (which are distinct and valid, though range is large)
    return [int(d) for d in divisors]
