import time
import math

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    divisors = P['divisors']
    max_val = P['maximum']
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Compute LCM of all divisors
    lcm_val = 1
    for d in divisors:
        lcm_val = _lcm(lcm_val, d)
    
    n = len(divisors)
    
    # For a given W, we need to find if there exists an interval [L, L+W]
    # such that each divisor d_i has at least one multiple in [L, L+W],
    # and we can assign distinct multiples to each divisor.
    
    # Key insight: due to periodicity with period lcm_val, we only need to
    # check L values in [1, lcm_val]. If a solution exists for some L,
    # then L + k*lcm_val also works (as long as within bounds).
    
    # For small W, the number of multiples per divisor in any window of width W
    # is at most floor(W/d) + 1, which is small.
    
    # Strategy: iterate W from 1 upward. For each W, check all critical L positions.
    # Critical L positions: for each divisor d, multiples of d are at positions
    # m*d. The window [L, L+W] contains a multiple of d iff L is in
    # [m*d - W, m*d] for some m. So critical L values are m*d and m*d - W.
    
    # But we only need to check L in [1, lcm_val] due to periodicity.
    # And L must be such that L+W <= max_val (or we can use L in [1, lcm_val]
    # and the actual position is L + k*lcm_val for appropriate k).
    
    # Actually, simpler: just check L in [1, lcm_val] and verify that
    # there exists some shift k such that all assigned values are <= max_val.
    
    best_solution = None
    
    # Upper bound on W: worst case, we can always pick multiples that are
    # far apart. But a reasonable upper bound: sum of gaps.
    # For safety, let's try W up to 200.
    max_W_to_try = 200
    
    for W in range(1, max_W_to_try + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Generate candidate L values in [1, lcm_val]
        # For each divisor d, critical positions are m*d and m*d - W
        # where m*d is in [1, lcm_val] (or slightly beyond for the -W case)
        
        candidates = set()
        for d in divisors:
            # Multiples of d in [1, lcm_val + W] (to handle m*d - W >= 1)
            m_start = max(1, 1)  # m*d >= 1
            m_end = (lcm_val + W) // d
            for m in range(m_start, m_end + 1):
                pos = m * d
                candidates.add(pos)
                candidates.add(pos - W)
        
        # Filter to valid L range [1, lcm_val]
        candidates = sorted([L for L in candidates if 1 <= L <= lcm_val])
        
        # For each candidate L, check if we can assign distinct multiples
        # in [L, L+W] (mod lcm_val, but we'll handle the actual values later)
        
        found = False
        for L in candidates:
            if time.time() > deadline - 2.0:
                break
            
            # Get available multiples for each divisor in [L, L+W]
            # within one period [1, lcm_val]
            # A multiple of d in [L, L+W] modulo lcm_val:
            # We need k*d such that L <= k*d <= L+W (in the cyclic sense)
            # Or more simply: k*d mod lcm_val in [L, L+W] (cyclic)
            
            # Actually, let's think differently. The pattern repeats every lcm_val.
            # So a multiple of d at position p (1-indexed) appears at p, p+lcm_val, p+2*lcm_val, ...
            # In window [L, L+W], the multiples of d are those p where
            # there exists integer k such that L <= p + k*lcm_val <= L+W.
            # This means p mod lcm_val is in [L, L+W] (cyclic) or
            # p mod lcm_val is in [L, lcm_val] U [1, L+W-lcm_val] if W >= lcm_val.
            
            # Since W < lcm_val (we hope), it's simpler:
            # p mod lcm_val in [L, L+W] (non-cyclic, since W < lcm_val)
            # But we need to handle the wrap-around case.
            
            # Let's just enumerate: for divisor d, the multiples in one period are
            # d, 2d, 3d, ..., (lcm_val/d)*d
            # For window [L, L+W], a multiple m*d is in the window if:
            #   L <= m*d <= L+W  (direct)
            #   or m*d > lcm_val and m*d - lcm_val in [L, L+W] (wrap)
            #   or m*d < L and m*d + lcm_val in [L, L+W] (wrap other way)
            
            # Since W < lcm_val, at most one wrap is possible.
            
            available = []
            valid = True
            for d in divisors:
                avail = []
                # Direct: m*d in [L, L+W]
                m_lo = max(1, (L + d - 1) // d)
                m_hi = min(lcm_val // d, (L + W) // d)
                for m in range(m_lo, m_hi + 1):
                    avail.append(m * d)
                
                # Wrap-around: m*d in [L, lcm_val] and we consider m*d - lcm_val? No.
                # The multiples in the infinite line are ..., -lcm_val+d, d, d+lcm_val, ...
                # In window [L, L+W] with L >= 1, W < lcm_val:
                # A multiple m*d (where m*d can be any multiple, positive or negative)
                # is in [L, L+W] iff L <= m*d <= L+W.
                # Since L >= 1 and L+W < lcm_val + 1 (if W < lcm_val),
                # m*d must be in [1, lcm_val] essentially, or possibly 0 or negative?
                # No, we need positive integers.
                
                # Actually, the simplest: enumerate all multiples of d in [1, lcm_val]
                # and check if any fall in [L, L+W] OR if (m*d - lcm_val) falls in [L, L+W]
                # (which means m*d in [L+lcm_val, L+W+lcm_val], but m*d <= lcm_val, so no)
                # OR if (m*d + lcm_val) falls in [L, L+W] (m*d + lcm_val >= lcm_val + 1 > L+W, so no)
                
                # So only direct multiples in [L, L+W] within [1, lcm_val] matter.
                
                if not avail:
                    valid = False
                    break
                available.append(avail)
            
            if not valid:
                continue
            
            # Now try to assign distinct values
            # Use backtracking with MRV
            order = sorted(range(n), key=lambda i: len(available[i]))
            assignment = [None] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                var_idx = order[idx]
                for val in available[var_idx]:
                    if val not in used:
                        used.add(val)
                        assignment[var_idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[var_idx] = None
                return False
            
            if backtrack(0):
                # We found a solution in the pattern [L, L+W] within one period.
                # Now we need to find actual values <= max_val.
                # The values are assignment[i] (in [1, lcm_val]).
                # We can use these directly if all <= max_val.
                # Since lcm_val = 510510 < 1000000, all values are <= max_val.
                
                result = [int(assignment[i]) for i in range(n)]
                
                # Verify distinctness and divisibility
                if len(set(result)) != n:
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                    break
                else:
                    # Check bounds
                    if all(1 <= x <= max_val for x in result):
                        return result
        
        # If no solution found for this W, continue
    
    # Fallback: return the divisors themselves (they are distinct and valid)
    return [int(d) for d in divisors]
