import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1  # alpha = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        # Sort roots to break symmetry and help local optimizers
        roots_sorted = np.sort(roots)
        
        norm_sq = compute_norm_sq(roots_sorted)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # Multi-start local optimization using Nelder-Mead and Powell.
    # We use log-space to ensure positivity.
    
    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Generate a set of initial guesses
    initial_guesses = []
    
    # 1. The known seed: geometric progression a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    initial_guesses.append(seed_log_roots)
    
    # 2. Variations of geometric progressions
    for a in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        for q in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
            guess = np.log(a) + np.arange(12) * np.log(q)
            initial_guesses.append(guess)
            
    # 3. Random log-uniform guesses
    rng = np.random.default_rng(42)
    for _ in range(50):
        # Log-uniform between 0.1 and 1000
        guess = rng.uniform(np.log(0.1), np.log(1000.0), size=12)
        initial_guesses.append(guess)
        
    # 4. "Clustered" guesses (roots close together)
    for center in [1.0, 5.0, 10.0, 50.0]:
        for width in [0.1, 0.5, 1.0, 2.0]:
            guess = np.log(center) + rng.uniform(-width, width, size=12)
            initial_guesses.append(guess)

    # Sort initial guesses by their cost to prioritize good starts
    # This allows us to stop early if we find a very good solution
    scored_guesses = []
    for guess in initial_guesses:
        try:
            val = cost_func(guess)
            scored_guesses.append((val, guess))
        except:
            continue
            
    scored_guesses.sort(key=lambda x: x[0])
    
    # Run local optimization on the top N initial guesses
    # We have ~590s. Nelder-Mead is fast. Let's try 30 best starts.
    max_starts = 30
    
    for i, (init_val, init_guess) in enumerate(scored_guesses[:max_starts]):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        # If the initial guess is already better than our best, update
        if init_val < best_val:
            best_val = init_val
            best_log_roots = init_guess
            best_roots = np.exp(init_guess)
            
        # Run Nelder-Mead
        try:
            result = minimize(
                cost_func, 
                init_guess, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(result.x)
        except Exception as e:
            pass
            
        # Run Powell (often more robust for box-constrained-like problems in log space)
        if time.time() - start_time < deadline - 10:
            try:
                result_powell = minimize(
                    cost_func, 
                    init_guess, 
                    method='Powell',
                    options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-15}
                )
                if result_powell.fun < best_val:
                    best_val = result_powell.fun
                    best_log_roots = result_powell.x
                    best_roots = np.exp(result_powell.x)
            except Exception as e:
                pass

    # Final polish with L-BFGS-B if we have a good candidate
    if best_log_roots is not None and time.time() - start_time < deadline - 5:
        try:
            polish_result = minimize(
                cost_func, 
                best_log_roots, 
                method='L-BFGS-B',
                bounds=[(-10.0, 15.0)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(polish_result.x)
        except Exception:
            pass

    # Ensure roots are positive and sorted
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
