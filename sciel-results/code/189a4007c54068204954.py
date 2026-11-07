import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    best_w = float('inf')
    best_L = 0
    best_assignment = None
    
    # Search L from 1 to a reasonable limit
    # The optimal L is likely small; search up to 500000
    L_limit = 500000
    
    for L in range(1, L_limit + 1):
        if time.time() > deadline:
            break
        
        # Find minimal W for this L
        # Start with W=0, increase
        # For efficiency, start with a lower bound
        # The minimal W must be at least max(ceil(L/d_i)*d_i - L) over i, but we need distinctness
        
        # Get the first multiple >= L for each divisor
        first_mults = []
        for d in divisors:
            if L % d == 0:
                first_mults.append(L)
            else:
                first_mults.append((L // d + 1) * d)
        
        # Lower bound on W: max(first_mults) - L
        W_lower = max(first_mults) - L
        
        # Try W from W_lower upward
        # But we need distinctness, so might need more
        W = W_lower
        max_W = 200  # cap search per L
        
        found = False
        while W <= max_W and not found:
            if time.time() > deadline:
                break
            
            # Build candidate lists
            candidates = []
            valid = True
            for i, d in enumerate(divisors):
                # First multiple >= L
                if L % d == 0:
                    m = L
                else:
                    m = (L // d + 1) * d
                # All multiples in [L, L+W]
                cands = []
                while m <= L + W:
                    if m <= max_val:
                        cands.append(m)
                    m += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
            
            if not valid:
                W += 1
                continue
            
            # Try to find distinct assignment
            # Sort by number of candidates (fewest first)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            assigned = [None] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for c in candidates[i]:
                    if c not in used:
                        used.add(c)
                        assigned[i] = c
                        if backtrack(idx + 1):
                            return True
                        used.remove(c)
                        assigned[i] = None
                return False
            
            if backtrack(0):
                found = True
                current_w = W
                if current_w < best_w:
                    best_w = current_w
                    best_L = L
                    best_assignment = assigned[:]
            
            W += 1
        
        if best_w == 0:  # impossible but check
            break
    
    # If we found a solution, return it
    if best_assignment is not None:
        # Verify
        result = []
        for i in range(n):
            val = best_assignment[i]
            assert isinstance(val, int)
            assert val > 0
            assert val <= max_val
            assert val % divisors[i] == 0
            result.append(val)
        # Check distinct
        assert len(set(result)) == n
        return result
    
    # Fallback: use the divisors themselves (they are distinct and valid)
    return divisors[:]
