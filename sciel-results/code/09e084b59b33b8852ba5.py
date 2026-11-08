from scipy.optimize import differential_evolution
import math
from scipy.optimize import minimize
import numpy as np
import time
def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    max_deg = 2 * m
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((max_deg + 1, max_deg + 1))
    for k in range(max_deg + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial P(x) = prod (x - r_i)^2 
        in the Laguerre basis L_k^{(1)}.
        roots: array of m positive numbers.
        """
        # Build polynomial Q(x) = prod (x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_Q_desc = np.poly(roots)
        
        # P(x) = Q(x)^2
        # Convolve coeffs_Q_desc with itself to get P_coeffs_desc
        coeffs_P_desc = np.convolve(coeffs_Q_desc, coeffs_Q_desc)
        
        # Convert to ascending order for the matrix multiplication
        p = coeffs_P_desc[::-1]
        
        # Ensure p has length max_deg + 1
        if len(p) < max_deg + 1:
            p = np.pad(p, (0, max_deg + 1 - len(p)))
        else:
            p = p[:max_deg + 1]
            
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(max_deg + 1) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Generate initial guesses based on Laguerre polynomial zeros
    # The optimal roots are often close to the zeros of L_m^{(alpha)}(x)
    # We try several alpha values to cover the landscape
    
    best_val = 1e10
    best_log_roots = None
    
    # Candidate initializations:
    # 1. Zeros of L_m^{(0)}(x)
    # 2. Zeros of L_m^{(1)}(x)
    # 3. Zeros of L_m^{(2)}(x)
    # 4. Geometric sequence (fallback/seed)
    
    candidates = []
    for alpha in [0, 1, 2, 3]:
        try:
            zeros = spec.roots_genlaguerre(m, alpha)
            # Ensure positive and real
            zeros = zeros[np.isreal(zeros)].real
            zeros = zeros[zeros > 0]
            if len(zeros) == m:
                candidates.append(np.log(zeros))
        except Exception:
            pass
            
    # Add geometric seed
    seed_roots = 4.0 * (2.0 ** np.arange(m))
    candidates.append(np.log(seed_roots))
    
    # Also add a linear seed
    lin_roots = np.linspace(1, 100, m)
    candidates.append(np.log(lin_roots))

    # Local optimization from each candidate
    for init_log_roots in candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
            
        # Perturb the initial guess slightly to explore neighborhood
        # We do a few random perturbations for each candidate
        for pert_iter in range(5):
            if time.time() - start_time > deadline - 20:
                break
                
            if pert_iter == 0:
                current_init = init_log_roots.copy()
            else:
                # Add small noise
                noise = np.random.normal(0, 0.1, m)
                current_init = init_log_roots + noise
                
            try:
                res = minimize(
                    cost_func, 
                    current_init, 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_log_roots = res.x.copy()
            except Exception:
                pass

    # If local optimization failed or didn't find anything better, 
    # try a short Differential Evolution run as a last resort
    if best_log_roots is None or best_val > 1e5:
        bounds = [(-4.0, 12.0)] * m
        try:
            res = differential_evolution(
                cost_func, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15, 
                tol=1e-8, 
                polish=True
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
        except Exception:
            pass

    if best_log_roots is None:
        # Absolute fallback
        return [4.0 * (2.0 ** i) for i in range(m)]
        
    best_roots = np.exp(best_log_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    return list(best_roots)
