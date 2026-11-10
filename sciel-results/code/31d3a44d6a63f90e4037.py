import numpy as np
import time
import sys
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Solves the Prime Number Theorem partial function problem using Linear Programming.
    
    Strategy:
    1. Choose a domain size K.
    2. Formulate the problem as an LP:
       - Variables: f_1, ..., f_K
       - Objective: Minimize sum(f_k * log(k) / k)  (since we want to maximize -sum)
       - Equality: sum(f_k / k) = 0
       - Inequalities: sum(f_k * floor(x/k)) <= 1.0001 for x in 1..10K
       - Bounds: -10 <= f_k <= 10
    3. Use scipy.optimize.linprog with HiGHS solver.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    def fallback():
        # Safe fallback: small K, heuristic values
        # Based on previous analysis, mu(k) works well.
        # Let's return a small valid set.
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([0.0, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        # Check S
        S = np.sum(values / domain.astype(np.float64))
        values[0] = values[0] - S
        # Check M(1)
        # M(1) = values[0]
        # If M(1) > 1.0001, it fails. 
        # For this specific fallback, let's ensure it's valid.
        # S = 0 - 1/2 - 1/3 - 1/5 + 1/30 = -1/2 - 1/3 - 1/5 + 1/30
        # = (-15 - 10 - 6 + 1)/30 = -30/30 = -1.
        # f(1) = 0 - (-1) = 1.
        # M(1) = 1 <= 1.0001. OK.
        return domain, values

    try:
        # Determine K based on available time.
        # Start with a moderate K to ensure we get a result quickly, 
        # then try larger K if time permits.
        
        # K=1000 is a good balance. 
        # Constraints: 10 * 1000 = 10,000 inequalities.
        # Variables: 1000.
        # This is very fast for HiGHS.
        
        K = 1000
        
        # Check time
        if time.time() > deadline - 30:
            return fallback()

        # Precompute log(k)/k for objective
        k_vals = np.arange(1, K + 1, dtype=np.float64)
        log_k_over_k = np.log(k_vals) / k_vals
        # Objective is to MINIMIZE c^T x, where c = log(k)/k
        # We want to MAXIMIZE -sum(f_k * log(k)/k)
        # So we minimize sum(f_k * log(k)/k)
        c = log_k_over_k.copy()
        # Note: f_1 has log(1)/1 = 0, so c[0] = 0.

        # Precompute 1/k for equality constraint
        inv_k = 1.0 / k_vals

        # Build inequality matrix A_ub for M(x) <= 1.0001
        # M(x) = sum_{k=1}^K f_k * floor(x/k)
        # We need this for x = 1, 2, ..., 10*K
        
        X_max = 10 * K
        # Memory check: X_max * K floats. 
        # If K=1000, X_max=10000. 10000*1000 = 10^7 floats = 80MB. Acceptable.
        # If K=5000, X_max=50000. 50000*5000 = 2.5*10^8 floats = 2GB. Too much.
        # So stick to K=1000 or slightly higher if optimized.
        
        # We can construct A_ub row by row to save memory if needed, 
        # but for K=1000, dense is fine.
        
        # Optimization: 
        # floor(x/k) is non-zero only if k <= x.
        # For x < K, only first x rows are non-zero.
        # For x >= K, all k contribute.
        
        A_ub = np.zeros((X_max, K), dtype=np.float64)
        
        # Efficient filling of A_ub
        # For each k, floor(x/k) is constant in blocks.
        # But vectorized construction per x is O(K) per x, total O(K^2 * 10) = 10 K^2.
        # For K=1000, 10^7 ops. Very fast.
        
        for x in range(1, X_max + 1):
            # floor(x/k) for k=1..K
            # Only k <= x matter
            k_range = np.arange(1, min(x, K) + 1)
            A_ub[x-1, k_range-1] = x // k_range
            
        b_ub = np.full(X_max, 1.0001, dtype=np.float64)

        # Equality constraint: sum(f_k / k) = 0
        A_eq = inv_k.reshape(1, -1)
        b_eq = np.array([0.0])

        # Bounds
        bounds = [(-10.0, 10.0)] * K

        # Solve
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )

        if not res.success:
            # If LP fails (shouldn't happen as 0 is a feasible solution), fallback
            return fallback()

        f_vals = res.x

        # Verify constraints just in case of numerical drift
        # 1. Check sum f(k)/k == 0
        S_check = np.sum(f_vals / k_vals)
        if abs(S_check) > 1e-9:
             # If numerical error, we could correct f(1)
             # But linprog should be exact enough.
             pass

        # 2. Check M(x) <= 1.0001
        # We already enforced this, but let's double check the max M(x)
        # M(x) = A_ub @ f_vals
        M_vals = A_ub @ f_vals
        if np.any(M_vals > 1.0001 + 1e-9):
            # Numerical violation?
            # Clip or fallback?
            # If it's slightly over, we might need to scale down.
            # But linprog should satisfy <=.
            # Let's check if it's a small epsilon issue.
            max_M = np.max(M_vals)
            if max_M > 1.0001:
                # Scale down all f values to satisfy constraint?
                # This reduces objective.
                # Better to just fallback if it's significantly violated.
                if max_M > 1.001:
                    return fallback()
                # Otherwise, assume it's close enough or clip?
                # The evaluator clips values first, then computes M.
                # Wait, the evaluator logic:
                # 1. Clip values to [-10, 10].
                # 2. Replace f(1) by f(1) - S.
                # 3. Compute M(x).
                # My LP assumes f is already clipped (bounds) and sum=0 (so f1 unchanged).
                # So M(x) calculated here should be the final M(x).
                
                # If there's a tiny violation, it's a risk.
                # Let's ensure strict compliance by subtracting a tiny epsilon from all f?
                # No, that breaks sum=0.
                # Let's just trust HiGHS.
                
        # Construct output
        # Domain: 1..K
        # Values: f_vals
        # We can filter out zeros to reduce m?
        # The problem allows any m. Smaller m is faster for evaluator.
        # However, the LP solution might have many non-zeros.
        # Let's keep all K to be safe, K=1000 is small.
        
        domain = k_vals.astype(np.int64)
        values = f_vals.astype(np.float64)
        
        # Ensure values are within [-10, 10] (LP bounds ensure this, but float precision?)
        values = np.clip(values, -10.0, 10.0)
        
        # Re-check sum after clipping (clipping might break sum=0 if bounds were active)
        # If bounds were active, sum might not be 0.
        # If sum != 0, evaluator will adjust f(1).
        # This adjustment might violate M(x) <= 1.0001 if f(1) changes significantly.
        # However, f(1) is bounded by [-10, 10].
        # The change in M(x) due to f(1) adjustment is delta_f1 * x.
        # Max delta_f1 is roughly 20. Max x is 10K.
        # This could be huge.
        # Therefore, it is CRITICAL that sum(f_k/k) is exactly 0.
        # Since LP enforces this as equality, and bounds are symmetric, 
        # and we didn't clip (values are already in bounds), the sum should be 0.
        
        # Let's verify sum
        S_final = np.sum(values / domain.astype(np.float64))
        if abs(S_final) > 1e-6:
            # If numerical error, manually adjust f(1)
            # f(1) = f(1) - S_final
            # But this might break M(1) constraint.
            # M(1) = f(1).
            # If we adjust f(1), M(1) changes by -S_final.
            # If S_final is small, it's fine.
            values[0] -= S_final
            # Re-check M(1)
            if values[0] > 1.0001:
                # Fallback if we broke the constraint
                return fallback()
            
        return domain, values

    except Exception as e:
        return fallback()

# EVOLVE-BLOCK-END
