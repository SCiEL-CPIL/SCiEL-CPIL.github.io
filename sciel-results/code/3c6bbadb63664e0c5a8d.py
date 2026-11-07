import time
import math

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
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    # Fallback solution: use the divisors themselves
    best_span = max(divisors) - min(divisors)
    best_solution = divisors[:]
    
    # Iterate W from 1 upward. First valid solution found is optimal.
    for W in range(1, best_span + 1):
        if time.time() > deadline:
            break
            
        for Y in range(0, L):
            if time.time() > deadline:
                break
                
            # For each divisor, find multiples in [Y, Y+W]
            candidates = [[] for _ in range(n)]
            valid_window = True
            
            for i, d in enumerate(divisors):
                # First multiple >= Y
                k = (Y + d - 1) // d
                val = k * d
                if val > Y + W:
                    valid_window = False
                    break
                # Add all multiples in [Y, Y+W]
                while val <= Y + W:
                    candidates[i].append(val)
                    val += d
            
            if not valid_window:
                continue
                
            # Check if we can assign distinct values using backtracking
            # Sort by number of candidates (most constrained first)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used_values = set()
            path = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    if v not in used_values:
                        used_values.add(v)
                        path[i] = v
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(v)
                        path[i] = None
                return False
            
            if backtrack(0):
                # Found valid solution with span <= W
                actual_span = max(path) - min(path)
                if actual_span < best_span:
                    best_span = actual_span
                    best_solution = path[:]
                return [int(x) for x in path]
    
    return [int(x) for x in best_solution]
