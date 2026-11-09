import numpy as np
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 595  # Safety margin for 600s limit
    
    m = 12
    alpha = 1
    
    # 1. Precompute the matrix K such that ||p||^2 = p^T K p
    # p are the monomial coefficients of the polynomial P(x) = prod (x - r_i)
    # P(x) = sum_{j=0}^m p_j x^j
    # The norm is defined in the Laguerre basis.
    # L_k^(alpha)(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+alpha, k-j) / j!
    
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + alpha, k - j) / math.factorial(j)
            
    d = np.arange(m + 1) + 1 # Norm weights for Laguerre basis
    
    # We want c such that p = A^T c => c = (A^T)^-1 p
    # Norm^2 = c^T diag(d) c = p^T (A^T)^-T diag(d) (A^T)^-1 p
    # K = A^-1 diag(d) (A^T)^-1
    
    try:
        A_inv = np.linalg.inv(A)
        K = A_inv @ np.diag(d) @ A_inv.T
    except np.linalg.LinAlgError:
        A_pinv = np.linalg.pinv(A)
        K = A_pinv @ np.diag(d) @ A_pinv.T

    def get_roots_from_coeffs(coeffs_asc):
        # coeffs_asc: [p_0, p_1, ..., p_m]
        # np.roots expects descending [p_m, ..., p_0]
        coeffs_desc = coeffs_asc[::-1]
        try:
            roots = np.roots(coeffs_desc)
        except Exception:
            return None
        
        # Check if all roots are real and positive
        if not all(np.isreal(roots)):
            return None
        roots_real = roots.real
        if not all(roots_real > 0):
            return None
        return roots_real

    def compute_score_from_roots(roots):
        # roots: array of 12 positive numbers
        # Compute monomial coefficients p (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        
        norm_sq = float(p @ K @ p)
        if norm_sq < 1e-10:
            return 1e10 # Invalid or zero norm
        return 1.0 / norm_sq

    def evaluate_state(coeffs_asc):
        # coeffs_asc: [p_0, p_1, ..., p_m]
        # We need to ensure the polynomial has 12 positive real roots.
        # This is the hard constraint.
        roots = get_roots_from_coeffs(coeffs_asc)
        if roots is None:
            return 1e10
        return compute_score_from_roots(roots)

    # Strategy: Simulated Annealing on the coefficients?
    # No, the constraint "12 positive real roots" is very restrictive for random coefficients.
    # It's better to stick to the root parameterization but use a different optimizer.
    # Let's try **Basin Hopping** or **Powell** or **Nelder-Mead** with many restarts?
    # The current program uses DE + L-BFGS-B.
    # Let's try a hybrid: 
    # 1. Generate many diverse initial guesses (Laguerre roots, geometric, random).
    # 2. Use **Nelder-Mead** (derivative-free, robust) for local optimization.
    # 3. Use **Basin Hopping** (scipy.optimize.basinhopping) which combines
    #    local minimization with global stochastic kicks. This is structurally
    #    different from DE (population-based) and L-BFGS-B (gradient-based).
    
    # Let's implement a custom Basin Hopping loop to control the time budget strictly.
    
    def cost_func_log_roots(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        return compute_score_from_roots(roots)

    # Initial candidates
    candidates = []
    
    # 1. Laguerre Roots of L_12^(1)
    lag_coeffs_desc = np.zeros(13)
    for k in range(13):
        if k <= 12:
            lag_coeffs_desc[12 - k] = ((-1)**k) * math.comb(13, 12 - k) / math.factorial(k)
    try:
        lag_roots = np.roots(lag_coeffs_desc)
        lag_roots_real = lag_roots[np.abs(lag_roots.imag) < 1e-9].real
        lag_roots_pos = lag_roots_real[lag_roots_real > 0]
        if len(lag_roots_pos) == 12:
            candidates.append(np.log(lag_roots_pos))
    except Exception:
        pass

    # 2. Geometric progressions
    for start, end, base in [(1.0, 10.0, None), (0.5, 5.0, None), (0.1, 20.0, None)]:
        r = start * ((end/start) ** (np.arange(12) / 11.0))
        candidates.append(np.log(r))

    # 3. Random points in log space
    random.seed(42)
    for _ in range(5):
        r = np.exp(np.random.uniform(-2, 3, 12))
        candidates.append(np.log(r))

    best_val = 1e10
    best_log_roots = None
    
    # Custom Basin Hopping
    # We will iterate through candidates and perform a basin hopping search from each.
    # Basin Hopping: 
    # 1. Minimize (Nelder-Mead or L-BFGS-B)
    # 2. Kick (random perturbation)
    # 3. Minimize
    # 4. Accept if lower energy (Metropolis)
    
    from scipy.optimize import minimize, basinhopping
    
    # We use scipy's basinhopping but we need to manage the time carefully.
    # Since basinhopping can run indefinitely, we wrap it in a time check.
    # However, scipy's basinhopping doesn't have a built-in time limit.
    # So we implement a simple manual basin hopping loop.
    
    def local_minimize(x0, method='Nelder-Mead'):
        # Nelder-Mead is derivative-free and often good for non-smooth or constrained-ish problems
        # L-BFGS-B uses gradients. Let's stick to Nelder-Mead for robustness in the "kick" phase.
        try:
            res = minimize(cost_func_log_roots, x0, method=method, options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            return res.x, res.fun
        except Exception:
            return x0, cost_func_log_roots(x0)
    
    temperature = 1.0
    kick_scale = 0.5
    
    for seed_log in candidates:
        if time.time() > deadline - 10:
            break
            
        current_x = seed_log.copy()
        current_f = cost_func_log_roots(current_x)
        
        # If the seed is already bad, skip
        if current_f > 1e5:
            continue
            
        if current_f < best_val:
            best_val = current_f
            best_log_roots = current_x.copy()
            
        # Run Basin Hopping for a fixed number of steps or until time limit
        # Let's run 200 kicks for each seed, but check time.
        for i in range(200):
            if time.time() > deadline - 10:
                break
                
            # 1. Local Minimize
            min_x, min_f = local_minimize(current_x, method='Nelder-Mead')
            
            if min_f < current_f:
                current_x = min_x
                current_f = min_f
            
            if current_f < best_val:
                best_val = current_f
                best_log_roots = current_x.copy()
                
            # 2. Kick
            # Random perturbation in log space
            step = np.random.normal(0, kick_scale, 12)
            kicked_x = current_x + step
            
            # 3. Evaluate Kicked Point (optional, usually we minimize from kicked point)
            # Let's just minimize from the kicked point in the next iteration?
            # Standard Basin Hopping: 
            #   x_new = x_current + kick
            #   Minimize x_new -> x_min
            #   Accept x_min if E(x_min) < E(x_current) or with probability exp(-(E_new-E_old)/T)
            
            # Let's follow the standard protocol strictly.
            # The loop above minimized current_x. Now we kick from current_x.
            
            # Actually, let's structure the loop as:
            # 1. Kick current_x to get x_proposed
            # 2. Minimize x_proposed to get x_min
            # 3. Compare E(x_min) with E(current_x)
            
            # Let's re-implement the loop body correctly.
            
            # (The previous loop body was slightly mixed. Let's restart the inner loop logic conceptually)
            # To save time, I will just use scipy.optimize.basinhopping with a maxcall limit.
            # But basinhopping is a black box. Let's try running it for a short duration.
            
            # Alternative: Just run Powell method (derivative free, efficient) from multiple starts.
            # Powell is often better than Nelder-Mead for high dimensions.
            
            break # Break out of manual loop to use scipy.basinhopping below
            
    # Use scipy.optimize.basinhopping for the most promising seeds
    # It handles the kick and minimize internally.
    
    def objective(x):
        return cost_func_log_roots(x)
    
    for seed_log in candidates:
        if time.time() > deadline - 20:
            break
            
        # Filter out bad seeds
        f0 = objective(seed_log)
        if f0 > 1e5:
            continue
            
        # Run basinhopping
        # We limit the number of calls to ensure it finishes
        # maxcall is the total number of function evaluations
        # Let's allow 5000 evaluations per seed. 
        # 5000 evals * ~1ms/eval = 5s. 
        # We have ~10 seeds. 50s total. Safe.
        
        try:
            res = basinhopping(
                objective, 
                seed_log, 
                stepsize=0.5, 
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6}},
                maxcall=5000,
                seed=42
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
        except Exception:
            pass

    # Final polish with L-BFGS-B if we have time
    if best_log_roots is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                cost_func_log_roots, best_log_roots, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
        except Exception:
            pass

    if best_log_roots is None:
        # Fallback
        return [4.0 * (2.0 ** i) for i in range(12)]
        
    best_roots = np.exp(best_log_roots)
    return list(np.maximum(best_roots, 1e-6))

