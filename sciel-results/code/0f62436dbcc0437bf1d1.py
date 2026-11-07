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
    deadline = start_time + 85.0  # safety margin before 90s
    
    divisors = DIVISORS[:]
    n = len(divisors)
    
    # Compute LCM of all divisors
    lcm_val = 1
    for d in divisors:
        lcm_val = _lcm(lcm_val, d)
    
    # The theoretical minimum window size is n-1 = 9 (since we need n distinct integers)
    # But actually the window must be at least large enough to fit n distinct multiples.
    # Let's search from R = 9 upward.
    
    # For a given window [S, S+R] (inclusive), we need to pick one multiple of each divisor
    # from within that window, all distinct.
    # 
    # Key insight: the problem is periodic modulo LCM. So we can enumerate window starts S
    # modulo LCM, and for each, check if there exists a valid assignment.
    #
    # For efficiency, for a fixed R and fixed S mod LCM, the set of available multiples
    # for divisor d in [S, S+R] is determined. We need to check if we can assign distinct
    # values.
    
    # Strategy: iterate R from 9 upward. For each R, iterate S from 0 to LCM-1.
    # For each (R, S), check feasibility using backtracking with bitsets.
    
    # Precompute for each divisor d, the residues of multiples of d mod LCM
    # Actually, let's think differently. For a window starting at S with length R (so values S..S+R),
    # a multiple of d is in the window iff there exists k such that S <= k*d <= S+R.
    # The multiples of d in [S, S+R] are: ceil(S/d)*d, ceil(S/d)*d + d, ..., up to S+R.
    
    # For each divisor d and each possible S (0..LCM-1), we can precompute the set of offsets
    # (relative to S) where multiples of d fall within [S, S+R]. But R varies.
    
    # Better approach: for each S in 0..LCM-1, and for each divisor d, compute the list of
    # actual multiples of d in [S, S+R] for a given R. But this depends on R.
    
    # Let's use a different formulation. For a window [S, S+R], the multiples of d in it are
    # {k*d : S <= k*d <= S+R}. The count is roughly R/d + 1.
    
    # For feasibility check with backtracking:
    # For each divisor d_i, the candidate values are multiples of d_i in [S, S+R].
    # We need to assign one to each, all distinct.
    
    # Since R is small (starting from 9), the number of candidates per divisor is small.
    
    best_result = None
    best_R = float('inf')
    
    # Start with the trivial solution as fallback
    trivial = divisors[:]
    best_R = max(trivial) - min(trivial)
    best_result = trivial
    
    # Search for smaller R
    for R in range(9, 100):  # search up to R=99
        if time.time() > deadline - 5.0:
            break
        
        found = False
        for S in range(lcm_val):
            if time.time() > deadline - 5.0:
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
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                # Extract the solution
                # We need to reconstruct the assignment
                # Let's redo with tracking
                used2 = set()
                assignment = [None] * n
                
                def backtrack2(idx):
                    if idx == n:
                        return True
                    i = order[idx]
                    for val in candidates[i]:
                        if val not in used2:
                            used2.add(val)
                            assignment[i] = val
                            if backtrack2(idx + 1):
                                return True
                            used2.remove(val)
                            assignment[i] = None
                    return False
                
                if backtrack2(0):
                    result = [int(assignment[i]) for i in range(n)]
                    # Verify
                    assert len(set(result)) == n
                    for i, d in enumerate(divisors):
                        assert result[i] % d == 0
                        assert result[i] >= 1
                        assert result[i] <= MAX_VAL
                    actual_R = max(result) - min(result)
                    if actual_R < best_R:
                        best_R = actual_R
                        best_result = result
                        found = True
                    # We found a valid R, but maybe a smaller R exists? No, R is fixed here.
                    # Actually we want to minimize max-min, which equals R in this window.
                    # But the actual max-min might be less than R if the min and max aren't at the boundaries.
                    # Wait, the window is [S, S+R], but the actual min and max of chosen values
                    # could be strictly inside. So actual_R <= R.
                    # We should continue searching for smaller R.
                    break
        
        if found:
            # We found a solution with this R. Check if we can do better with smaller R?
            # No, we're iterating R from 9 upward, so this is the first R that works.
            # But wait, we break on first S that works for a given R. There might be a different
            # S for the same R that gives a tighter actual interval. But actual_R <= R, and
            # we're minimizing actual_R. Let's continue checking all S for this R to find the
            # minimum actual_R.
            # Actually, let's not break on first S. Let's check all S for this R.
            # But for time efficiency, let's just take the first valid one.
            # Actually, since we're iterating R from 9 upward, the first R that has any valid S
            # gives us a solution with max-min <= R. But maybe R=9 has a solution with max-min=9,
            # and R=10 has a solution with max-min=10. We want the minimum possible max-min.
            # So we should find the minimum R such that there exists S with a valid assignment,
            # and then among all valid (R,S) pairs, find the one with minimum actual max-min.
            
            # For now, let's just return the first valid solution found.
            # To improve, we could continue searching for smaller actual_R.
            break
    
    if best_result is None:
        best_result = divisors[:]
    
    # Ensure types are correct
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
