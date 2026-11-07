import math
import time

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    divisors = P['divisors']
    max_val = P['maximum']
    n = len(divisors)
    
    # Calculate LCM of all divisors to define the period
    D = 1
    for d in divisors:
        D = _lcm(D, d)
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0

    # We search for the minimum range R
    # The minimal spread S* will be found at R=S*
    for R in range(1, 10000):
        if time.time() > deadline:
            break
            
        # Generate critical points for L in [0, D)
        # A change in candidates for d_i happens when L crosses a multiple of d_i
        # or L+R crosses a multiple of d_i.
        # i.e., L = k*d_i or L = k*d_i - R.
        critical_points = set()
        for d in divisors:
            # Multiples of d in [0, D + R]
            # k from 0 to (D + R) // d
            k_max = (D + R) // d
            for k in range(k_max + 1):
                m = k * d
                # Points: m mod D and (m - R) mod D
                critical_points.add(m % D)
                critical_points.add((m - R) % D)
                
        # Sort critical points
        sorted_points = sorted(critical_points)
        
        # Iterate over intervals defined by critical points
        # The candidate sets are constant in (p_i, p_{i+1})
        # We check L = p_i for each interval [p_i, p_{i+1})
        # Note: The interval wraps around D.
        # We check L = 0 as well? 
        # The set of critical points partitions [0, D).
        # For each p in sorted_points, the interval is [p, next_p).
        # We should check a representative L from each interval.
        # p itself is a boundary. The candidates for L=p might be different from L=p+1.
        # Actually, at L=p, a multiple enters or leaves.
        # So the set of candidates for L=p is the set for the interval ending at p?
        # Let's just check L = p for all p in sorted_points.
        # This covers all distinct candidate sets.
        
        for L in sorted_points:
            if time.time() > deadline:
                break
                
            # Check if a valid distinct assignment exists in [L, L+R]
            # Candidates for each divisor d_i are multiples of d_i in [L, L+R]
            candidates = []
            valid_window = True
            for i in range(n):
                d = divisors[i]
                # Smallest multiple of d >= L
                # Note: L is in [0, D). The window is [L, L+R].
                # We are effectively working on the circle of length D.
                # But since R is small, and we just need distinct values,
                # we can treat the window as a linear segment [L, L+R].
                # The values will be in [L, L+R].
                # Since L < D and R is small, max val < D + R < 1000000.
                
                # k_min = ceil(L/d)
                if L % d == 0:
                    k_min = L // d
                else:
                    k_min = L // d + 1
                    
                # k_max = floor((L+R)/d)
                k_max = (L + R) // d
                
                if k_min > k_max:
                    valid_window = False
                    break
                    
                cand_list = []
                for k in range(k_min, k_max + 1):
                    val = k * d
                    cand_list.append(val)
                candidates.append(cand_list)
                
            if not valid_window:
                continue
                
            # Backtracking to find distinct assignment
            assigned = [None] * n
            used_values = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                if time.time() > deadline:
                    return False
                    
                for val in candidates[idx]:
                    if val not in used_values:
                        assigned[idx] = val
                        used_values.add(val)
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(val)
                        assigned[idx] = None
                return False
            
            if backtrack(0):
                # Found a valid solution
                # The actual spread is max(assigned) - min(assigned)
                # Since we iterate R from 1, this is the minimal possible spread.
                return [int(x) for x in assigned]

    # Fallback: Return divisors themselves (or a known good solution)
    # The divisors themselves are distinct and valid.
    # Spread = max(divisors) - min(divisors) = 39 - 6 = 33.
    # This is a valid solution.
    return [int(x) for x in divisors]
# EVOLVE-BLOCK-END
