import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # 1. Precompute APs
    # We need to minimize the max |sum| over all APs.
    # An AP is defined by start s (0-indexed), step d, and length L.
    # The terms are s, s+d, ..., s+(L-1)d.
    # Constraint: s + (L-1)d < N.
    # We only care about APs with L >= 2? 
    # The prompt says "any start, positive step and length including one".
    # Length 1 APs have sum +/- 1, so max is at least 1.
    # We want to minimize the max over ALL APs.
    
    ap_indices = []
    for d in range(1, N + 1):
        for s in range(0, N):
            terms = []
            curr = s
            while curr < N:
                terms.append(curr)
                curr += d
            if len(terms) >= 2:
                ap_indices.append(terms)
            # Length 1 is trivial, sum is always +/-1.
            # If we include length 1, the max is at least 1.
            # If we exclude length 1, we are minimizing the max over L>=2.
            # The prompt says "including one". So the global max includes 1.
            # Since 1 is fixed, minimizing the max over L>=2 is equivalent to minimizing the global max (as long as max(L>=2) >= 1).
            # Actually, if max(L>=2) < 1, it's impossible since sums are integers.
            # So min global max = max(1, min max(L>=2)).
            # So we just need to minimize the max over L>=2.
            
    num_aps = len(ap_indices)
    
    # Build the constraint matrix A for MILP.
    # We want to find x in {-1, 1}^N such that |sum_{i in AP} x_i| <= D for all APs.
    # This is equivalent to:
    # -D <= sum_{i in AP} x_i <= D for all APs.
    
    # To use scipy.milp, we convert x in {-1, 1} to y in {0, 1} where x = 2y - 1.
    # sum x_i = sum (2y_i - 1) = 2 sum y_i - len(AP).
    # So |2 sum y_i - len(AP)| <= D.
    # -D <= 2 sum y_i - len(AP) <= D
    # len(AP) - D <= 2 sum y_i <= len(AP) + D
    # (len(AP) - D)/2 <= sum y_i <= (len(AP) + D)/2
    
    # We can just keep x as variables in {-1, 1} if scipy supports it?
    # scipy.milp requires integer variables. We can set integrality=1 for all.
    # But bounds for x are [-1, 1].
    # Let's stick to x in {-1, 1}.
    
    # A is a sparse matrix of size (2 * num_aps) x N.
    # Row 2j: +1 for indices in AP j. Constraint: -D <= A[2j] @ x <= D.
    # Row 2j+1: -1 for indices in AP j. Constraint: -D <= A[2j+1] @ x <= D.
    # Actually, just using A @ x <= D and -A @ x <= D is sufficient.
    # So rows 0..num_aps-1 are +APs, rows num_aps..2*num_aps-1 are -APs.
    
    # Building a dense matrix of size 2*num_aps x N might be memory heavy.
    # num_aps is roughly N^2/2 / 2? No.
    # For d=1, N APs.
    # For d=N/2, 2 APs.
    # Total APs ~ N log N.
    # N=256. N log N ~ 256 * 8 = 2048.
    # Let's count exactly.
    # d=1: 255 APs (len>=2)
    # d=2: 127 APs
    # ...
    # Sum_{d=1}^{N/2} (N-d-1) approx N^2/4.
    # 256^2 / 4 = 16384.
    # So ~16k APs.
    # 2 * 16k * 256 floats = 8M floats = 64 MB. Manageable.
    
    # Let's build the matrix.
    A_rows = []
    for terms in ap_indices:
        row_pos = np.zeros(N)
        for idx in terms:
            row_pos[idx] = 1.0
        A_rows.append(row_pos)
        row_neg = -row_pos
        A_rows.append(row_neg)
        
    A = np.array(A_rows)
    num_constraints = A.shape[0]
    
    # Objective: 0 (feasibility problem)
    c = np.zeros(N)
    
    # Bounds: x in {-1, 1}
    # We can use integrality=1 and bounds (-1, 1).
    # But scipy.milp might not handle non-integer bounds for integer vars well?
    # It should.
    # Alternatively, transform to y in {0, 1}.
    # Let's stick to x in {-1, 1} for simplicity, but check scipy docs.
    # "integrality: 1 for integer, 0 for continuous".
    # "bounds: Bounds(lb, ub)".
    # If integrality=1, variables are integers.
    # So x will be -1 or 1.
    
    integrality = np.ones(N)
    bounds = Bounds(lb=-1, ub=1)
    
    def check_feasibility(D):
        # Constraints: -D <= A @ x <= D
        # This can be written as:
        # A @ x <= D
        # -A @ x <= D
        # Our A already contains both +AP and -AP rows.
        # So we just need A @ x <= D for all rows.
        # Wait, my A construction:
        # Row 2j: +AP_j. Constraint: +AP_j @ x <= D.
        # Row 2j+1: -AP_j. Constraint: -AP_j @ x <= D.
        # This is correct.
        
        # LinearConstraint(A, -np.inf, D) ?
        # No, we need A @ x <= D.
        # So lower bound -inf, upper bound D.
        
        try:
            cons = LinearConstraint(A, -np.inf, D)
            res = milp(c=c, integrality=integrality, bounds=bounds, constraints=cons, options={"time_limit": 5})
            if res.status == 0:
                # Feasible
                return True, res.x
            else:
                return False, None
        except Exception as e:
            return False, None

    # Bisection search for minimum D
    # Lower bound: 1 (since singleton sums are 1, but we excluded them. Min sum for L>=2 can be 0? No, sum of two +/-1 is -2, 0, 2. So min max is at least 0? No, we have to cover ALL APs.
    # Actually, for N=2, APs: {0,1}. Sum is -2, 0, 2. Max abs is 2.
    # For N=4, APs: {0,1}, {1,2}, {2,3}, {0,2}, {1,3}, {0,1,2,3}, {0,2,4}...
    # It is known that the discrepancy is small.
    # Let's start with D=1, 2, 3...
    
    best_D = None
    best_x = None
    
    # Try small D values first
    # Theoretical lower bound for N=256?
    # For step 1, it's a random walk. Max deviation ~ sqrt(N) ~ 16.
    # But we can balance it.
    # Thue-Morse sequence has low discrepancy.
    # For N=256, Thue-Morse discrepancy is bounded by log2(N) ~ 8?
    # Actually, Thue-Morse has discrepancy O(log N).
    # Let's try D from 1 upwards.
    
    # To speed up, we can use a known good solution as a starting point?
    # MILP doesn't take warm starts easily in scipy.
    
    # Let's try D=1 to 10.
    # If D=1 is infeasible, try D=2, etc.
    
    # Optimization: If D is infeasible, we don't need to try smaller D.
    # We want the MINIMUM D.
    # So we check D=1, 2, 3... until feasible.
    
    # However, MILP might be slow.
    # Let's set a time limit per MILP call.
    
    for D in range(1, 20):
        if time.time() > deadline - 5:
            break
            
        feasible, x = check_feasibility(D)
        if feasible:
            best_D = D
            best_x = x
            break
            
    if best_x is not None:
        # Convert to list of ints
        return [int(round(xi)) for xi in best_x]
        
    # Fallback: Use Thue-Morse or Random if MILP fails/times out
    # Thue-Morse
    signs = []
    for i in range(N):
        # Count number of 1s in binary representation of i
        # Thue-Morse: t(n) = parity of number of 1s in binary expansion of n
        # Sign = 1 if even, -1 if odd? Or vice versa.
        # Let's use 1 for even, -1 for odd.
        count = bin(i).count('1')
        if count % 2 == 0:
            signs.append(1)
        else:
            signs.append(-1)
            
    return signs

# Test locally
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Result: {res[:10]}")
    
    # Verify
    N = 256
    max_sum = 0
    for d in range(1, N + 1):
        for s in range(0, N):
            curr = s
            s_sum = 0
            while curr < N:
                s_sum += res[curr]
                curr += d
            if abs(s_sum) > max_sum:
                max_sum = abs(s_sum)
    print(f"Max AP Sum: {max_sum}")
