import time
from math import factorial
from collections import deque

_START = time.time()
_DEADLINE = _START + 85

def construct():
    target = factorial(12)
    
    # Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = []
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.append(val)
                            
    divisors.sort()
    divisors_set = set(divisors)
    
    # Precompute exponents for each divisor
    def get_exponents(val):
        e2 = 0
        while val % 2 == 0:
            e2 += 1
            val //= 2
        e3 = 0
        while val % 3 == 0:
            e3 += 1
            val //= 3
        e5 = 0
        while val % 5 == 0:
            e5 += 1
            val //= 5
        e7 = 0
        while val % 7 == 0:
            e7 += 1
            val //= 7
        e11 = 0
        while val % 11 == 0:
            e11 += 1
            val //= 11
        return (e2, e3, e5, e7, e11)

    target_exp = get_exponents(target)
    
    # Map divisor value to its exponent tuple
    # Note: Multiple divisors can have the same value? No, divisors are unique.
    # But we need to map index in candidate list to exponent.
    
    # Iterate M
    # Lower bound for M
    # Geometric mean of 12 factors: (479001600)^(1/12) ~ 23.3
    # So M >= 24
    
    for M in range(24, 1000): # Upper bound heuristic
        if time.time() > _DEADLINE:
            break
            
        # Get candidates <= M
        candidates = [d for d in divisors if d <= M]
        K = len(candidates)
        
        if K == 0:
            continue
            
        # Check if product of all candidates can reach target? 
        # Not a necessary condition, but if sum of exponents < target, impossible.
        # Actually, we just need to check if a subset exists.
        
        if K <= 16:
            # Bitmask DP
            # State: (e2, e3, e5, e7, e11, mask)
            # Start: (10, 5, 2, 1, 1, 0)
            # Target: (0, 0, 0, 0, 0, any_mask)
            
            start_state = (target_exp[0], target_exp[1], target_exp[2], target_exp[3], target_exp[4], 0)
            visited = {start_state}
            queue = deque([start_state])
            parent = {start_state: None}
            
            found = None
            while queue:
                if time.time() > _DEADLINE:
                    break
                state = queue.popleft()
                e2, e3, e5, e7, e11, mask = state
                
                if e2 == 0 and e3 == 0 and e5 == 0 and e7 == 0 and e11 == 0:
                    found = state
                    break
                    
                # Try adding each candidate not in mask
                for i in range(K):
                    if mask & (1 << i):
                        continue
                    
                    d = candidates[i]
                    # Check if d divides current remaining product
                    # i.e., exponents of d <= current exponents
                    # We need exponents of d
                    # Precompute exponents for candidates?
                    # candidates are from divisors, so we can compute on fly or precompute for this M
                    # Let's precompute for this M
                    # Actually, we can precompute a list of exponents for candidates
                    # But candidates change with M.
                    # Let's compute inside loop for simplicity, or precompute for all divisors once.
                    
                    # Precompute exponents for all divisors
                    # This is better
                    # Let's do it outside the M loop
                    
                    pass # Placeholder
                    
            # If found, backtrack
            if found:
                # Backtrack to find the set of indices
                path = []
                curr = found
                while parent[curr] is not None:
                    prev_state, idx = parent[curr]
                    path.append(idx)
                    curr = prev_state
                # The last step in backtracking is the first factor added?
                # parent[curr] = (prev_state, idx) means curr was reached from prev_state by adding idx.
                # So the factors are the indices in path.
                factors = [candidates[i] for i in path]
                return factors
        else:
            # K > 16, use DFS
            # This is a fallback, might be slow
            pass
            
    # Fallback
    return [target]
