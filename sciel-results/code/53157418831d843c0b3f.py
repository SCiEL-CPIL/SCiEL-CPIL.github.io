import time
from math import gcd

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
LCM_ALL = 30030

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    
    # We search for the minimum W
    # Based on previous scores, W is likely small (< 100)
    # We scan W from 1 upwards
    
    for W in range(1, 500):
        if time.time() - start_time > 80.0:
            # Fallback if time runs out (should not happen for small W)
            # Just return a valid solution, e.g., the divisors themselves scaled
            # But we need distinct multiples.
            # Simple fallback: x_i = i * d_i * 1000
            return [i * d * 1000 for i, d in enumerate(DIVISORS)]
        
        # For this W, check all possible starting positions L in one LCM period
        # L must be such that [L, L+W] is within [1, MAX_VAL]
        # Since W is small, L can be 1..LCM_ALL
        for L in range(1, LCM_ALL + 1):
            if L + W > MAX_VAL:
                break
            
            # Check if a valid assignment exists for this L and W
            # Candidates for divisor i are multiples of d_i in [L, L+W]
            candidates = []
            valid = True
            for i in range(N):
                d = DIVISORS[i]
                # First multiple >= L
                first = ((L + d - 1) // d) * d
                cand_list = []
                v = first
                while v <= L + W:
                    cand_list.append(v)
                    v += d
                if not cand_list:
                    valid = False
                    break
                candidates.append(cand_list)
            
            if not valid:
                continue
            
            # Solve the assignment problem using backtracking
            # We assign a candidate value to each divisor
            # To optimize, sort divisors by number of candidates (MRV)
            # But we need to return values in original order
            
            # Let's use a simple recursive DFS
            # State: index of divisor being assigned, set of used values
            # Since N=10, recursion depth is 10
            
            # Optimization: Pre-sort indices by candidate count
            indices = list(range(N))
            indices.sort(key=lambda i: len(candidates[i]))
            
            assignment = [None] * N
            used_vals = set()
            
            def dfs(idx):
                if idx == N:
                    return True
                
                i = indices[idx]
                for v in candidates[i]:
                    if v not in used_vals:
                        used_vals.add(v)
                        assignment[i] = v
                        if dfs(idx + 1):
                            return True
                        used_vals.remove(v)
                        assignment[i] = None
                return False
            
            if dfs(0):
                # Check distinctness and bounds (already ensured by construction)
                # Return the assignment in the original order
                return [int(assignment[i]) for i in range(N)]
    
    # If no solution found (unlikely), return a default valid set
    # e.g., x_i = d_i * (i+1) * 100
    return [int(d * (i + 1) * 100) for i, d in enumerate(DIVISORS)]
