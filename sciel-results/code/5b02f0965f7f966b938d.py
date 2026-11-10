# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    DEADLINE = 580.0  # safety margin before 630s kill

    # Start with a strong baseline
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0

    # Try to improve via LP over a range of k values
    # The constraint is M(x) = sum_k f(k)*floor(x/k) <= 1 for x in [1, 10*K_max]
    # Objective: maximize -sum_k f(k)*log(k)/k for k >= 2 (f(1) is adjusted by evaluator)

    # We'll use scipy's linprog if available, otherwise fall back to baseline
    try:
        from scipy.optimize import linprog
        has_scipy = True
    except ImportError:
        has_scipy = False

    if not has_scipy:
        return best_domain, best_values

    # Try increasing K_max values
    # Based on inspiration, K=1000 gave 1.075. Let's try to go higher.
    # The constraint matrix B is (10*K) x K.
    # For K=5000, size is 50000 x 5000 = 250M elements. ~2GB memory. Risky but let's try if time permits.
    # Actually, let's stick to K=1000 first as a safe bet, then try 2000, 5000 if time allows.
    K_candidates = [1000, 2000, 5000]

    for K in K_candidates:
        if time.time() - start_time > DEADLINE - 60:
            break

        # Construct the constraint matrix B
        # B[x-1, j-1] = floor(x/j) - x/j for x in 1..10K, j in 1..K
        # We want to minimize c^T f subject to B f <= 1
        # where c[0]=0, c[j-1] = log(j)/j for j>=2
        
        # Vectorized construction
        x_vals = np.arange(1, 10 * K + 1)[:, np.newaxis]  # shape (10K, 1)
        j_vals = np.arange(1, K + 1)[np.newaxis, :]      # shape (1, K)
        
        # Calculate floor(x/j) - x/j
        # floor(x/j) is integer. x/j is float.
        # The result is in [-1, 0].
        # Note: for x < j, floor(x/j)=0, so term is -x/j.
        # For x >= j, term is {x/j} - 1 (if not integer) or 0 (if integer).
        # Wait, floor(x/j) - x/j is always in [-1, 0).
        # If x/j is integer, it is 0.
        # Otherwise it is -{x/j}.
        
        # Let's compute B directly.
        # To save memory, we could compute row by row if K is very large,
        # but for K=5000, 10K=50000. 50000*5000 = 250,000,000 floats.
        # 250M * 8 bytes = 2 GB. This might be too much for standard memory limits.
        # Let's be conservative and try K=1000 (10M floats, 80MB) and K=2000 (40M floats, 320MB).
        # If K=5000 is too slow/memory heavy, it will just fail or take too long.
        
        # Actually, the prompt says "Keep the function names...".
        # Let's try to optimize memory usage if possible, or just try K=1000 and 2000.
        
        # Let's stick to K=1000 and 2000 to be safe on memory/time.
        if K > 2000:
            # Check memory/time estimate
            # If we have enough time, we can try 5000.
            pass

        try:
            # Construct B
            # Using broadcasting
            # x_vals / j_vals creates a large intermediate array.
            # For K=2000, 20000 x 2000 = 40M divisions. This is fast in numpy.
            
            # However, creating the full B matrix for K=5000 might be slow or memory intensive.
            # Let's limit K to 2000 for safety in the dense construction phase.
            if K > 2000:
                 # For larger K, we might need a sparse approach or just skip.
                 # Given the 600s limit, let's see if K=2000 solves fast enough.
                 pass

            B = np.floor(x_vals / j_vals) - (x_vals / j_vals)
            
            # The constraints are B @ f <= 1
            # We also need bounds -10 <= f <= 10
            bounds = [(-10.0, 10.0)] * K
            
            # Objective: Maximize sum_{k=2}^K (-log(k)/k) * f_k
            # So minimize sum_{k=2}^K (log(k)/k) * f_k
            c = np.zeros(K)
            k_vals = np.arange(1, K + 1)
            # k=1 term is 0
            c[1:] = np.log(k_vals[1:]) / k_vals[1:]
            
            # Solve LP
            # Use HiGHS if available
            # Method 'highs' is the default in recent scipy versions and is fast.
            # For large sparse problems, 'highs' handles it well.
            # But B is dense here.
            
            res = linprog(c, A_ub=B, b_ub=np.ones(10 * K), bounds=bounds, method='highs')
            
            if time.time() - start_time > DEADLINE - 10:
                break

            if res.success:
                f_vals = res.x
                # Clip to [-10, 10] as the evaluator does (though bounds should ensure this)
                f_vals = np.clip(f_vals, -10.0, 10.0)
                
                # Compute the objective to verify
                # a = -sum_{k=2}^K f(k)*log(k)/k
                # Note: f(1) is included in the sum for the evaluator's adjustment,
                # but log(1)=0, so it doesn't contribute to 'a'.
                # The evaluator calculates a = -sum_k f(k) log(k) / k.
                # Since f(1) is adjusted to make sum f(k)/k = 0, does that affect 'a'?
                # No, 'a' is calculated on the final f values.
                # f(1) * log(1) / 1 = 0. So 'a' only depends on k>=2.
                
                a = -np.sum(f_vals[1:] * np.log(k_vals[1:]) / k_vals[1:])
                
                # Check feasibility more strictly?
                # The LP ensures M(x) <= 1 for the continuous relaxation?
                # No, the LP constraints are M(x) <= 1 for x=1..10K.
                # The evaluator checks M(x) for integer x in [1, 10K].
                # So this should be feasible.
                
                # Add a small epsilon to account for numerical errors in LP solution
                # The evaluator fails if M(x) > 1.0001.
                # LP solution might be slightly above 1 due to precision.
                # We can scale down the solution slightly if max_M is close to 1.
                
                # Let's verify max M(x)
                # M(x) = sum_k f(k) floor(x/k)
                # We can compute this efficiently.
                # But for now, trust the LP with a small margin.
                # If we want to be safe, we can multiply f_vals by 0.9999.
                # This reduces the objective slightly but ensures feasibility.
                
                # Let's check the max M(x) quickly for a few points or all?
                # Computing M(x) for all x is O(K * 10K) = 10K^2.
                # For K=2000, 40M ops. Fast.
                
                # Let's verify.
                # M = np.zeros(10*K)
                # for k in range(1, K+1):
                #    M += f_vals[k-1] * np.floor(np.arange(1, 10*K+1) / k)
                # max_M = np.max(M)
                
                # If max_M > 1.0001, scale down.
                # To save time in the loop, let's just apply a safety factor.
                # The LP solution is on the boundary. Scaling by 0.999 is safe.
                # But it reduces the score.
                # Let's try to verify first.
                
                # Optimization: We can compute M(x) using the same B matrix logic?
                # No, B was floor(x/k) - x/k.
                # M(x) = sum f(k) floor(x/k).
                # We didn't store floor(x/k) separately.
                
                # Let's just assume the LP solution is valid within numerical precision.
                # HiGHS is usually very accurate.
                # If we get a score of 0, it means infeasible.
                # Let's add a check.
                
                # Efficient check for max M(x):
                # M(x) is a step function.
                # We can compute it for all x in [1, 10K].
                # This is O(10K * K).
                # For K=2000, 40M operations. < 1s.
                
                # Let's do it.
                x_check = np.arange(1, 10 * K + 1)
                # We need sum_k f(k) floor(x/k)
                # This is a matrix multiplication if we build the floor matrix.
                # Floor matrix F is 10K x K.
                # F = np.floor(x_check[:, None] / np.arange(1, K+1)[None, :])
                # M = F @ f_vals
                # max_M = np.max(M)
                
                # Memory for F: 10K * K * 8 bytes.
                # For K=2000, 320MB. Acceptable.
                
                # But we already have B.
                # B = F - X/K matrix.
                # So F = B + X/K matrix.
                # X/K matrix is x_vals / j_vals.
                # We can reconstruct F?
                # x_vals / j_vals was an intermediate.
                # Let's just recompute F if needed.
                
                # Actually, let's just use the B matrix we have?
                # No, B is floor - frac.
                # Let's just recompute M for verification.
                
                # To save memory, we can compute M in chunks of x.
                # But for K=2000, full matrix is fine.
                
                F = np.floor(x_check[:, None] / np.arange(1, K + 1)[None, :])
                M_vals = F @ f_vals
                max_M = np.max(M_vals)
                
                if max_M > 1.0001:
                    # Scale down to ensure feasibility
                    # We need M(x) <= 1.
                    # Scale factor s = 1.0 / max_M * 0.9999
                    scale = 0.9999 / max_M
                    f_vals = f_vals * scale
                    # Recalculate a
                    a = -np.sum(f_vals[1:] * np.log(k_vals[1:]) / k_vals[1:])
                
                if a > best_score:
                    best_score = a
                    best_domain = np.arange(1, K + 1, dtype=np.int64)
                    best_values = f_vals.astype(np.float64)
                    
        except Exception as e:
            # If K is too large or memory error, break and use best found so far
            break

    return best_domain, best_values

# EVOLVE-BLOCK-END
