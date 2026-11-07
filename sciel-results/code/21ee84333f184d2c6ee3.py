import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    n = len(divisors)
    
    # Helper to check if a set of candidate lists has a distinct assignment
    def find_distinct_assignment(candidates):
        # candidates is a list of lists of ints
        # returns a list of chosen ints or None
        # Sort indices by length of candidate list to prune
        indices = sorted(range(n), key=lambda i: len(candidates[i]))
        chosen = [None] * n
        used = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            i = indices[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    chosen[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    chosen[i] = None
            return False
            
        if backtrack(0):
            return chosen
        return None

    # Search for minimal span S
    # Minimum span for 10 distinct integers is 9
    for S in range(9, 1000):
        if time.time() > deadline:
            break
            
        # Iterate over possible window start positions modulo L
        # We only need to check m in [0, L-1]
        for m in range(L):
            if time.time() > deadline:
                break
                
            candidates = []
            valid = True
            for d in divisors:
                min_k = (m + d - 1) // d # ceil(m/d)
                max_k = (m + S) // d      # floor((m+S)/d)
                if min_k > max_k:
                    valid = False
                    break
                # Generate candidates
                # k * d
                cands = [k * d for k in range(min_k, max_k + 1)]
                candidates.append(cands)
                
            if not valid:
                continue
                
            # Try to find distinct assignment
            assignment = find_distinct_assignment(candidates)
            if assignment is None:
                continue
                
            # Check if we can shift by k*L to fit in [1, 1000000]
            min_x = min(assignment)
            max_x = max(assignment)
            
            # We need 1 <= min_x + k*L and max_x + k*L <= 1000000
            # k >= (1 - min_x) / L
            # k <= (1000000 - max_x) / L
            
            k_min = math.ceil((1 - min_x) / L)
            k_max = math.floor((1000000 - max_x) / L)
            
            if k_min <= k_max:
                # Pick a valid k, e.g., k_min
                k = k_min
                result = [x + k * L for x in assignment]
                # Verify constraints
                # Distinct? Yes, by construction.
                # Divisible? Yes.
                # Range? Yes.
                # Positive? Yes.
                return result
                
    # Fallback: return the divisors themselves if no better found (should not happen)
    # But we need distinct multiples. Divisors are distinct?
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39 are distinct.
    # Span is 39-6 = 33.
    # This is a valid solution.
    return divisors[:]
