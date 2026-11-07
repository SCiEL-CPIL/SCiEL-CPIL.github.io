import time
import sys

# EVOLVE-BLOCK-START
divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

# Calculate LCM of divisors to bound the search space for L
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a // gcd(a, b) * b

L = 1
for d in divisors:
    L = lcm(L, d)

def check_feasibility(R):
    """
    Returns (is_feasible, solution_list) where solution_list is None if not feasible.
    """
    # Precompute nothing, just iterate L
    for start in range(L):
        # For each divisor, find the set of multiples in [start, start+R]
        sets = []
        valid = True
        for d in divisors:
            # Find first multiple >= start
            # k = ceil(start / d)
            if start % d == 0:
                k = start // d
            else:
                k = start // d + 1
            
            first_mult = k * d
            if first_mult > start + R:
                valid = False
                break
            
            # Collect all multiples in range
            s = []
            curr = first_mult
            while curr <= start + R:
                s.append(curr)
                curr += d
            sets.append(s)
        
        if not valid:
            continue
        
        # Now check if we can pick distinct elements from each set
        # Backtracking
        n = len(sets)
        # Sort indices by set size to optimize backtracking
        indices = list(range(n))
        indices.sort(key=lambda i: len(sets[i]))
        
        used = set()
        assignment = [None] * n
        
        def backtrack(idx):
            if idx == len(indices):
                return True
            i = indices[idx]
            for val in sets[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assignment[i] = None
            return False
        
        if backtrack(0):
            # Construct result in original order
            result = [0] * n
            for i in range(n):
                result[i] = assignment[i]
            return True, result
    
    return False, None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Binary search for minimum R
    lo = 9
    hi = 1000000  # Upper bound
    best_R = None
    best_sol = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        is_feas, sol = check_feasibility(mid)
        if is_feas:
            best_R = mid
            best_sol = sol
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_sol is None:
        # Fallback: should not happen, but return something valid
        # Just return divisors themselves if they are distinct and <= 1000000
        # They are distinct and small.
        return divisors[:]
    
    return best_sol
# EVOLVE-BLOCK-END
