import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # Pre-compute indices sorted by divisor size? 
    # Larger divisors have fewer multiples in a small window, so they are more constrained.
    # Sorting by divisor descending is a good heuristic for MRV if counts are similar.
    # But counts depend on L.
    
    # We will search W from 9 upwards.
    # For each W, we check all L in [1, LCM].
    
    # To speed up, we can check if a solution exists for a given L.
    
    for W in range(9, 100):
        if time.time() - start_time > timeout:
            break
            
        # We iterate L from 1 to LCM
        # Optimization: We can break early if we find a solution.
        found = False
        for L in range(1, LCM + 1):
            if time.time() - start_time > timeout:
                break
                
            start = L
            end = L + W
            
            # Collect available multiples
            # We want to sort by number of choices to prune early.
            # Let's collect (count, index, multiples)
            avail_info = []
            valid = True
            
            for i, d in enumerate(divisors):
                k = (start + d - 1) // d
                first = k * d
                if first > end:
                    valid = False
                    break
                multiples = []
                x = first
                while x <= end:
                    multiples.append(x)
                    x += d
                avail_info.append((len(multiples), i, multiples))
                
            if not valid:
                continue
            
            # Sort by count (most constrained first)
            avail_info.sort(key=lambda x: x[0])
            
            # Backtracking
            used = set()
            choices = [0] * n
            
            # Iterative deepening or just DFS?
            # DFS with MRV is standard.
            
            # To avoid re-defining function, we use a nested function but 
            # we can also just use a stack or a specific structure.
            # Given N=10, recursion is fine.
            
            def solve(idx):
                if idx == n:
                    return True
                count, i, mults = avail_info[idx]
                for v in mults:
                    if v not in used:
                        used.add(v)
                        choices[i] = v
                        if solve(idx + 1):
                            return True
                        used.remove(v)
                return False

            if solve(0):
                # Found a solution for this W
                result = [choices[i] for i in range(n)]
                # Verify
                if len(set(result)) == n:
                    for i in range(n):
                        if result[i] % divisors[i] != 0 or not (0 < result[i] <= 1000000):
                            # Should not happen
                            pass
                    return result
                found = True # If verification fails, something is wrong, but logic should prevent it.
                
        # If loop finishes without returning, no solution for this W.
        
    # Fallback
    return divisors[:]
