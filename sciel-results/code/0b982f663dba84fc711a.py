import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # We search for the minimum range R
    # Start with R=1 and increase
    # For each R, we search for a starting value M
    # We limit M to a reasonable range to ensure speed.
    # Given the small divisors, a tight cluster should appear early.
    
    max_R = 200
    max_M = 5000
    
    for R in range(1, max_R + 1):
        for M in range(1, max_M + 1):
            # Check if each divisor has at least one multiple in [M, M+R]
            # And collect them
            candidates = []
            valid = True
            for d in divisors:
                # Find multiples of d in [M, M+R]
                # First multiple >= M is ceil(M/d) * d
                k_start = (M + d - 1) // d
                k_end = (M + R) // d
                
                mults = []
                for k in range(k_start, k_end + 1):
                    val = k * d
                    if M <= val <= M + R:
                        mults.append(val)
                
                if not mults:
                    valid = False
                    break
                candidates.append(mults)
            
            if not valid:
                continue
            
            # Now we need to select one value from each candidates[i]
            # such that all selected values are distinct.
            # This is a bipartite matching problem or simple backtracking.
            # Since R is small, the number of candidates per divisor is small.
            
            # Optimization: Order variables by number of candidates (MRV heuristic)
            # This helps prune the search space more effectively.
            # We create a list of indices sorted by the length of candidates[i]
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            assignment = [None] * n
            used = set()
            
            def backtrack(pos):
                if pos == n:
                    return True
                i = indices[pos]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(pos + 1):
                            return True
                        used.remove(val)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                # Verify distinctness and constraints just in case
                if len(set(assignment)) == n:
                    # Check bounds
                    if all(1 <= x <= 1000000 for x in assignment):
                        return assignment[:]
            
    # Fallback if no small range found (should not happen)
    # Just return the divisors themselves (which are distinct and valid, but range might be large)
    # Or a simple greedy
    return divisors[:]
