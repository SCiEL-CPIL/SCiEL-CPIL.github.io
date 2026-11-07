"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = 10
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s
    
    divisors = DIVISORS[:]
    n = len(divisors)
    
    # Compute LCM of all divisors
    lcm_val = 1
    for d in divisors:
        lcm_val = _lcm(lcm_val, d)
    
    # Fallback solution: the divisors themselves
    trivial = divisors[:]
    best_R = max(trivial) - min(trivial)
    best_result = trivial[:]
    
    # Search for smaller R
    # For each R from 9 upward, check all S in 0..LCM-1
    for R in range(9, 50):
        if time.time() > deadline - 10.0:
            break
        
        found_any = False
        best_span_for_R = R + 1  # worst case is R+1 (shouldn't happen)
        best_assign_for_R = None
        
        for S in range(lcm_val):
            if time.time() > deadline - 10.0:
                break
            
            # Check if window [S, S+R] is feasible
            # For each divisor, get candidate multiples in [S, S+R]
            candidates = []
            feasible = True
            for d in divisors:
                # Multiples of d in [S, S+R]
                first_k = (S + d - 1) // d  # ceil(S/d)
                last_k = (S + R) // d  # floor((S+R)/d)
                if first_k > last_k:
                    feasible = False
                    break
                cand = [first_k * d + i * d for i in range(last_k - first_k + 1)]
                candidates.append(cand)
            
            if not feasible:
                continue
            
            # Now check if we can assign distinct values
            # Use backtracking with MRV heuristic
            # Order divisors by number of candidates (ascending)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                # Extract the solution
                result = [int(assignment[i]) for i in range(n)]
                # Verify
                assert len(set(result)) == n
                for i, d in enumerate(divisors):
                    assert result[i] % d == 0
                    assert result[i] >= 1
                    assert result[i] <= MAX_VAL
                actual_R = max(result) - min(result)
                if actual_R < best_span_for_R:
                    best_span_for_R = actual_R
                    best_assign_for_R = result[:]
                found_any = True
        
        if found_any and best_span_for_R < best_R:
            best_R = best_span_for_R
            best_result = best_assign_for_R
            # Since we're iterating R from 9 upward, and we found a solution,
            # the minimum possible span is at most best_R.
            # But a larger R window might still give a smaller actual span? No,
            # because actual span <= R, and we're going from small R to large R.
            # Actually, we should continue to see if smaller actual spans exist
            # with the same or larger R. But since actual_span <= R, and we
            # started from R=9, if we find span=9, that's optimal (minimum
            # possible for 10 distinct integers is 9).
            if best_R == 9:
                break
        elif found_any:
            # We found a solution with this R but it's not better than current best
            # Continue searching
            pass
    
    # Ensure types are correct
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
