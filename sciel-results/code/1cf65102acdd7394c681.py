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
        
    # We search for the minimum range R
    # Start with a baseline solution: the divisors themselves
    # This ensures we always have a valid fallback
    baseline = divisors[:]
    best_range = max(baseline) - min(baseline)
    best_solution = baseline
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin for 60s limit
    
    # Iterate over possible range sizes R
    # The minimum range must be at least 1 (distinct integers)
    # Upper bound on R? If we don't find a solution, R will grow.
    # However, since D is ~30k, and divisors are small, R should be small.
    # Let's iterate R from 1 up to, say, 1000.
    
    for R in range(1, 2000):
        if time.time() > deadline:
            break
            
        # For a fixed R, check all possible window starts L in [0, D-1]
        for L in range(D):
            if time.time() > deadline:
                break
                
            # Check if a valid distinct assignment exists in [L, L+R]
            # Candidates for each divisor d_i are multiples of d_i in [L, L+R]
            candidates = []
            valid_window = True
            for i in range(n):
                d = divisors[i]
                # Smallest multiple of d >= L
                # k_min = ceil(L/d)
                if L % d == 0:
                    k_min = L // d
                else:
                    k_min = L // d + 1
                    
                # Largest multiple of d <= L+R
                k_max = (L + R) // d
                
                if k_min > k_max:
                    valid_window = False
                    break
                    
                # Generate list of candidates
                # To optimize backtracking, sort candidates? 
                # Actually, just list them.
                cand_list = []
                for k in range(k_min, k_max + 1):
                    val = k * d
                    cand_list.append(val)
                candidates.append(cand_list)
                
            if not valid_window:
                continue
                
            # Backtracking to find distinct assignment
            # We want to assign x_i from candidates[i] such that all x_i are distinct
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
                # Found a valid solution with range R
                # The actual range is max(assigned) - min(assigned) which is <= R
                # But we are iterating R, so this is the minimal R for this L?
                # Yes, because we iterate R increasing.
                # However, for a fixed R, we might find a solution that uses a sub-interval.
                # The score is max - min.
                # Let's compute the actual score
                actual_min = min(assigned)
                actual_max = max(assigned)
                actual_score = actual_max - actual_min
                
                # Update best
                if actual_score < best_range:
                    best_range = actual_score
                    best_solution = assigned[:]
                    
                # Since we are iterating R from 1 upwards, and for a fixed R we found a solution,
                # can we stop?
                # Not necessarily. A larger R might allow a smaller actual min-max spread?
                # No, if a solution fits in [L, L+R], its spread is <= R.
                # If we found a solution with spread S <= R, and we are at R, 
                # any future R' > R will have spread >= S? 
                # Wait. If we found a solution with spread 5 at R=10, 
                # could there be a solution with spread 4 at R=11?
                # Yes, if the window [L', L'+11] contains numbers with spread 4.
                # But we would have found spread 4 at R=4 if it existed in any window?
                # No, the window start L matters.
                # However, if a set of numbers has spread S, it fits in a window of size S.
                # So if there exists a solution with spread S, it will be found when R=S.
                # Therefore, the first R for which *any* solution is found corresponds to the minimal possible spread.
                # Proof: Let S* be the minimal spread. There exists a valid assignment with spread S*.
                # This assignment fits in a window of size S*.
                # So when R=S*, the loop will check the L corresponding to min(assignment) 
                # (mod D) and find a valid assignment.
                # Thus, the first R that yields a valid assignment is S*.
                
                # We can return immediately?
                # We must ensure the returned values are <= 1000000.
                # L < D = 30030. R is small. So max val < 30030 + 2000 < 1000000.
                # So the constraint is satisfied.
                
                # Return the solution
                # Ensure types are int
                return [int(x) for x in assigned]

    # If no small R found, return baseline
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
