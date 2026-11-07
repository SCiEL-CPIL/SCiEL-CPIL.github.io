import time
import math

# EVOLVE-BLOCK-START

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of all divisors
    L = divisors[0]
    for i in range(1, n):
        L = _lcm(L, divisors[i])
        
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Fallback solution: just use the divisors themselves? No, they must be distinct.
    # They are distinct. But span is max(d)-min(d) = 39-6=33.
    # Is it valid? x_i = d_i. Distinct? Yes.
    # So we have a valid solution with span 33.
    best_span = 33
    best_solution = divisors[:]
    
    # We can try to improve upon 33.
    # Iterate W from 1 upwards.
    # If we find a solution with span W, it's better than best_span.
    
    # To speed up, we can skip W values that are larger than best_span.
    # But we want to minimize, so we start from 1.
    
    # Optimization: Precompute nothing complex, just loop.
    
    for W in range(1, best_span + 1):
        if time.time() > deadline:
            break
            
        for Y in range(L):
            if time.time() > deadline:
                break
                
            # Check if window [Y, Y+W] has a multiple for each d_i
            # And if we can assign distinct values.
            
            # For each divisor, find the multiples in [Y, Y+W]
            # candidates[i] will be a list of values
            candidates = [[] for _ in range(n)]
            valid_window = True
            
            for i, d in enumerate(divisors):
                # First multiple >= Y
                # k = ceil(Y/d)
                k = (Y + d - 1) // d
                val = k * d
                while val <= Y + W:
                    candidates[i].append(val)
                    val += d
                if not candidates[i]:
                    valid_window = False
                    break
            
            if not valid_window:
                continue
                
            # Check if distinct assignment is possible
            # We have n lists of candidates.
            # We need to pick one from each list such that all picked are distinct.
            # This is a bipartite matching problem.
            # Left side: divisors 0..n-1
            # Right side: values in the window
            
            # Since n is small, we can use simple backtracking.
            # Or, since the values are integers, we can just try to map.
            
            # Let's use a simple DFS matching.
            # To make it faster, we can sort candidates or just use the lists as is.
            
            used_values = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                for v in candidates[idx]:
                    if v not in used_values:
                        used_values.add(v)
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(v)
                return False
            
            if backtrack(0):
                # Found a valid assignment for span W
                # Extract the solution
                # Note: backtrack modifies used_values but we need the actual values.
                # Let's re-do the assignment to get the values.
                # Since it's small, we can just store the path.
                
                # Re-implement to store path
                used_values = set()
                path = [None] * n
                
                def backtrack_store(idx):
                    if idx == n:
                        return True
                    for v in candidates[idx]:
                        if v not in used_values:
                            used_values.add(v)
                            path[idx] = v
                            if backtrack_store(idx + 1):
                                return True
                            used_values.remove(v)
                            path[idx] = None
                    return False
                
                if backtrack_store(0):
                    # path contains the values
                    # Check if all distinct (should be)
                    if len(set(path)) == n:
                        # Update best
                        # The span is W.
                        # But wait, the actual span of the selected values might be less than W?
                        # No, the window is [Y, Y+W]. The values are in this window.
                        # The span is max(path) - min(path).
                        # This is <= W.
                        # But we are iterating W. If we find a solution in window W,
                        # the actual span is max(path)-min(path).
                        # It's possible that max(path)-min(path) < W.
                        # In that case, we would have found it in a smaller W?
                        # Not necessarily, because the window [Y, Y+W'] might not align.
                        # However, if the span is S, then the values fit in a window of size S.
                        # So if we iterate W from 1, we will find the minimal span first.
                        
                        actual_span = max(path) - min(path)
                        if actual_span < best_span:
                            best_span = actual_span
                            best_solution = path[:]
                        # Since we are iterating W in increasing order, 
                        # and we found a solution with span <= W,
                        # and we haven't found any with span < W (because we checked smaller W),
                        # this must be the optimal solution?
                        # Yes, because if there was a solution with span S < W,
                        # it would have been found in the iteration for W=S (or smaller).
                        # Wait, the window Y might be different.
                        # But we check all Y for each W.
                        # So if a solution with span S exists, it fits in some window of size S.
                        # So we would have found it when W=S.
                        # Thus, the first W for which we find ANY solution gives the minimal span.
                        
                        # So we can return immediately.
                        return [int(x) for x in path]
                        
    # If loop finishes without finding better, return best
    return [int(x) for x in best_solution]

# EVOLVE-BLOCK-END
