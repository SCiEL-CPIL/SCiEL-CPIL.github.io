import numpy as np
from scipy.optimize import linprog
import sys

def prime_number_theorem_partial_function():
    # We search for the best small K using linear programming
    best_score = 0.0
    best_domain = np.array([1], dtype=np.int64)
    best_values = np.array([0.0], dtype=np.float64)
    
    max_K = 10 # Try small K first, can increase if time permits
    
    for K in range(1, max_K + 1):
        # Variables are v_1, ..., v_K
        # But v_1 is determined by v_2...v_K if we fix the structure?
        # No, v_1 is a variable too, but f(1) = v_1 - S.
        # The evaluator uses f(1) = v_1 - S.
        # The metric uses f(k) which is the adjusted value.
        # Let's work with the adjusted values f(k) directly?
        # No, we return v(k). The evaluator adjusts.
        # It is easier to optimize v(k).
        
        # Objective: maximize a = - sum_{k=2}^K f(k) ln(k)/k
        # f(k) for k>1 is v(k).
        # f(1) is v(1) - S.
        # a = - ( (v(1)-S) * 0 + sum_{k=2}^K v(k) ln(k)/k )
        # a = - sum_{k=2}^K v(k) ln(k)/k
        # So we only care about v(k) for k>1 in the objective.
        # v(1) does not appear in the objective!
        # This means we can set v(1) to any value that helps feasibility?
        # No, S depends on v(1).
        # S = v(1) + sum_{k=2}^K v(k)/k.
        # f(1) = v(1) - S = - sum_{k=2}^K v(k)/k.
        # So f(1) is determined by v(2)..v(K).
        # Thus, the only free variables are v(2)..v(K).
        
        n_vars = K - 1
        if n_vars == 0:
            # K=1, f(1)=0, a=0
            continue
            
        # Variables: x_i = v(i+1) for i in 0..n_vars-1
        # Objective coeff for linprog (minimize -a):
        # minimize sum v(k) ln(k)/k
        c = np.zeros(n_vars)
        for i in range(n_vars):
            k = i + 2
            c[i] = np.log(k) / k
            
        # Constraints: M(x) <= 1 for x in 1..10K
        # M(x) = f(1) x + sum_{j=2}^K v(j) floor(x/j)
        # f(1) = - sum_{j=2}^K v(j)/j
        # M(x) = x * (- sum_{j=2}^K v(j)/j) + sum_{j=2}^K v(j) floor(x/j)
        # M(x) = sum_{j=2}^K v(j) ( floor(x/j) - x/j )
        
        A_ub = []
        b_ub = []
        
        for x in range(1, 10 * K + 1):
            row = np.zeros(n_vars)
            for i in range(n_vars):
                j = i + 2
                row[i] = np.floor(x / j) - x / j
            A_ub.append(row)
            b_ub.append(1.0) # M(x) <= 1
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for v(j)
        # The evaluator clips values to [-10, 10].
        # So v(j) must be in [-10, 10] to avoid clipping affecting the intended optimization?
        # If we set v(j) outside, it gets clipped, changing the function.
        # We should restrict search to [-10, 10].
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                v = res.x
                # Calculate score
                score = -np.sum(v * c) # since c is ln(k)/k, and we minimized sum v*c
                # Wait, linprog minimizes c^T x. We set c = ln(k)/k.
                # So res.fun = sum v(k) ln(k)/k.
                # a = - sum v(k) ln(k)/k = -res.fun.
                if score > best_score:
                    best_score = score
                    # Construct full domain and values
                    # v(1) is not in res.x, but we can set it arbitrarily?
                    # f(1) is determined. The returned value v(1) just needs to be consistent.
                    # Let's set v(1) = 0 for simplicity, or 1.
                    # S = v(1) + sum v(k)/k.
                    # f(1) = v(1) - S.
                    # The evaluator uses f(k).
                    # Let's return v(1)=1 (like mu(1)) and v(k)=res.x for k>1.
                    domain = np.array([1] + list(range(2, K + 1)), dtype=np.int64)
                    values = np.zeros(K, dtype=np.float64)
                    values[0] = 1.0
                    values[1:] = v
                    best_domain = domain
                    best_values = values
        except Exception:
            pass
            
    # Fallback if no solution found or score is low
    if best_score < 0.6:
        # Use the K=2 analytic solution: v=[1, -2]
        # Score ln(2) ~ 0.693
        best_domain = np.array([1, 2], dtype=np.int64)
        best_values = np.array([1.0, -2.0], dtype=np.float64)
        best_score = np.log(2)
        
    return best_domain, best_values
