import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start = time.time()
    
    # Strategy: Use scipy.optimize.milp to find a set of integers x_1 < ... < x_k
    # such that for all c in 1..k-1:
    # sum of largest c elements < sum of smallest c+1 elements
    # This ensures that the range of sums for cardinality c is strictly less 
    # than the range of sums for cardinality c+1, and by transitivity, all 
    # cardinality sum ranges are disjoint.
    
    # The variables are x_1, ..., x_k.
    # Constraints:
    # 1. 1 <= x_1 < x_2 < ... < x_k <= 256
    #    x_i are integers.
    #    x_1 >= 1
    #    x_k <= 256
    #    x_i + 1 <= x_{i+1}  =>  x_{i+1} - x_i >= 1
    # 2. Separation constraints:
    #    For each c from 1 to k-1:
    #    sum(x_{k-c+1}...x_k) < sum(x_1...x_{c+1})
    #    Let L_c = sum of largest c, M_c = sum of smallest c+1.
    #    L_c - M_c <= -1  (since integers)
    
    # We want to maximize k. We try k from 15 down to 1.
    # Why 15? The current best is 1.5 (seed size 10? or 15?). 
    # Wait, score 1.5 means value/seed = 1.5. 
    # If seed is 10, value is 15. If seed is 8, value is 12.
    # The prompt says "Maximize set size". 
    # The current program returns a list of ints. The score is the length.
    # If the current program scores 1.5, and the seed program scores 1.0, 
    # and the seed program likely returns 8 (powers of 2), then 1.5 * 8 = 12.
    # So the current best is likely 12 or 13.
    # Let's try to find a set of size 14, 15, 16...
    
    max_k = 18
    best_set = []
    
    for k in range(max_k, 0, -1):
        if time.time() - start > 80:
            break
            
        # Variables: x_0, ..., x_{k-1}
        # Objective: Maximize 0 (feasibility problem)
        # Or minimize 0.
        c_obj = np.zeros(k)
        
        # Bounds: 1 <= x_i <= 256
        lb = np.ones(k)
        ub = np.full(k, 256)
        bounds = Bounds(lb, ub)
        
        # Linear Constraints
        # 1. Strictly increasing: x_{i+1} - x_i >= 1
        #    A_inc * x >= b_inc
        #    Number of constraints: k-1
        A_inc = np.zeros((k-1, k))
        b_inc = np.ones(k-1)
        for i in range(k-1):
            A_inc[i, i] = -1
            A_inc[i, i+1] = 1
            # -x_i + x_{i+1} >= 1
            
        # 2. Separation constraints:
        #    For c in 1..k-1:
        #    sum(x_{k-c}...x_{k-1}) - sum(x_0...x_c) <= -1
        #    Note: indices are 0-based.
        #    Largest c: indices k-c to k-1.
        #    Smallest c+1: indices 0 to c.
        
        A_sep = []
        b_sep = []
        
        for c in range(1, k):
            row = np.zeros(k)
            # Sum of largest c
            for i in range(k-c, k):
                row[i] = 1
            # Sum of smallest c+1
            for i in range(0, c+1):
                row[i] = -1
            A_sep.append(row)
            b_sep.append(-1)
            
        if A_sep:
            A_sep = np.array(A_sep)
            b_sep = np.array(b_sep)
        else:
            A_sep = np.zeros((0, k))
            b_sep = np.zeros(0)
        
        # Combine constraints
        # We need LinearConstraint(A, lb, ub)
        # A_inc * x >= b_inc  =>  -A_inc * x <= -b_inc
        # A_sep * x <= b_sep
        
        # Let's just use two separate LinearConstraint objects or combine.
        # scipy.optimize.milp takes a single LinearConstraint object or a list?
        # It takes `constraints` which can be a LinearConstraint or a list of them.
        
        # Constraint 1: -A_inc * x <= -b_inc
        # A1 = -A_inc, lb1 = -inf, ub1 = -b_inc
        A1 = -A_inc
        ub1 = -b_inc
        lb1 = -np.inf * np.ones(len(ub1))
        
        # Constraint 2: A_sep * x <= b_sep
        # A2 = A_sep, lb2 = -inf, ub2 = b_sep
        A2 = A_sep
        ub2 = b_sep
        lb2 = -np.inf * np.ones(len(ub2))
        
        constraints = [
            LinearConstraint(A1, lb1, ub1),
            LinearConstraint(A2, lb2, ub2)
        ]
        
        # integrality: all variables are integers
        integrality = np.ones(k)
        
        try:
            res = milp(
                c=c_obj,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": 5, "presolve": True}
            )
            
            if res.status == 0: # Optimal
                x = res.x
                # Convert to list of ints
                sol = [int(round(v)) for v in x]
                # Verify strictly increasing and bounds
                if all(1 <= sol[i] <= 256 for i in range(k)) and all(sol[i] < sol[i+1] for i in range(k-1)):
                    # Double check the separation logic just in case
                    # The ILP enforced the sufficient condition.
                    # Let's do a quick DP check to be absolutely sure it's valid
                    # (The sufficient condition implies validity, but let's be safe)
                    if _is_valid_fast(sol):
                        best_set = sol
                        # We found a valid set of size k. 
                        # Since we iterate k downwards, this is the largest we found so far.
                        # Can we try larger? No, we started from max_k and went down.
                        # So the first valid k is the max k in our range.
                        return [int(v) for v in best_set]
                        
        except Exception:
            pass
            
    # If no set found via ILP (e.g., solver failed or no solution in range),
    # fall back to a simple greedy or return the best found (empty)
    # But we should probably have a fallback.
    # Let's try a simple greedy if ILP fails for all k?
    # Or just return the best_set (which might be empty).
    # To be safe, let's try a very small k if ILP didn't work.
    
    if not best_set:
        # Fallback: Powers of 2
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
        
    return [int(x) for x in best_set]

def _is_valid_fast(s):
    n = len(s)
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    for x in s:
        for c in range(n - 1, -1, -1):
            if dp[c]:
                new_sums = {v + x for v in dp[c]}
                dp[c + 1].update(new_sums)
    seen = set()
    for c in range(n + 1):
        if not seen.isdisjoint(dp[c]):
            return False
        seen.update(dp[c])
    return True
