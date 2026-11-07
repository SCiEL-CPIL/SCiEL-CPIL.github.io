import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of divisors
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    # We search for the minimum window size W
    # The window is [S, S+W]
    # We iterate W from 1 upwards
    # For each W, we iterate S from 1 to L
    # If a valid configuration is found, we return it
    
    # Deadline guard
    start_time = time.time()
    timeout = 45.0 # 45 seconds to be safe within 60s limit
    
    # Helper to check if a window [S, S+W] works
    def find_assignment(S, W):
        # For each divisor i, find multiples in [S, S+W]
        candidates = []
        for i in range(n):
            d = divisors[i]
            # Smallest multiple >= S
            m = ((S + d - 1) // d) * d
            if m > S + W:
                return None
            # Collect all multiples in [S, S+W]
            cands = []
            while m <= S + W:
                cands.append(m)
                m += d
            candidates.append(cands)
            
        # Backtracking to find distinct values
        # Assign values to divisors 0..n-1
        assigned = [False] * (S + W + 1) # Boolean array for used numbers in window
        # Note: S+W can be up to L+W ~ 30030. Array size is fine.
        # Actually, we can just use a set of used numbers.
        used = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            for val in candidates[idx]:
                if val not in used:
                    used.add(val)
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False
            
        if backtrack(0):
            # Reconstruct the solution
            # We need to map back to the original indices
            # The 'used' set contains the values, but we need to know which value went to which divisor.
            # The backtrack function above doesn't store the assignment.
            # Let's modify it to store the assignment.
            pass
            
        return None

    # Better backtracking that stores assignment
    def solve_window(S, W):
        candidates = []
        for i in range(n):
            d = divisors[i]
            m = ((S + d - 1) // d) * d
            if m > S + W:
                return None
            cands = []
            while m <= S + W:
                cands.append(m)
                m += d
            candidates.append(cands)
            
        assignment = [0] * n
        used = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            for val in candidates[idx]:
                if val not in used:
                    used.add(val)
                    assignment[idx] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False
            
        if backtrack(0):
            return assignment
        return None

    for W in range(1, L + 1):
        if time.time() - start_time > timeout:
            # Fallback: just return divisors (which is valid, just not optimal)
            # But we need distinct. Divisors are distinct.
            # Check if divisors are distinct: 6,10,14,15,21,22,26,33,35,39 are distinct.
            # Max is 39, Min is 6. Range 33.
            # This is a valid solution.
            return divisors[:]
            
        for S in range(1, L + 1):
            res = solve_window(S, W)
            if res is not None:
                # Verify distinctness and bounds
                if len(set(res)) != n:
                    continue
                if any(x < 1 or x > 1000000 for x in res):
                    continue
                return res
    # Should not reach here
    return divisors[:]
