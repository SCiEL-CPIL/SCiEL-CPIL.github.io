"""Improved program for the refined uncertainty inequality task."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import differential_evolution, minimize, newton

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

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

    def _poly_from_roots(roots):
        """
        Build monic polynomial coefficients in ascending order from roots.
        Returns ascending coefficients [c0, c1, ..., c12].
        """
        poly = np.array([1.0])
        for r in roots:
            factor = np.array([-r, 1.0])
            poly = np.polymul(poly, factor)
        return poly

    def compute_norm_sq_and_grad(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis
        and its gradient with respect to the roots.
        """
        p = _poly_from_roots(roots)
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        # Compute gradient via finite differences (fallback)
        # For efficiency, we'll use analytical approach
        grad = np.zeros(12)
        eps = 1e-7
        for i in range(12):
            roots_plus = roots.copy()
            roots_plus[i] += eps
            p_plus = _poly_from_roots(roots_plus)
            c_plus = A_T_inv @ p_plus
            norm_sq_plus = np.sum(c_plus**2 * (np.arange(13) + 1))
            grad[i] = (norm_sq_plus - norm_sq) / eps
        
        return norm_sq, grad

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4 = 1/||P||^2.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq_and_grad(roots)[0]
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def cost_func_with_grad(log_roots):
        """
        Cost function with gradient for L-BFGS-B.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.ones(12)
            
        norm_sq, grad_norm = compute_norm_sq_and_grad(roots)
        
        if norm_sq <= 1e-10:
            return 1e10, np.ones(12)
        
        # d(1/norm_sq)/d(log_r_i) = -1/norm_sq^2 * d(norm_sq)/d(log_r_i)
        # d(norm_sq)/d(log_r_i) = d(norm_sq)/d(r_i) * r_i
        grad_cost = -grad_norm * roots / (norm_sq ** 2)
        
        return 1.0 / norm_sq, grad_cost

    # Strategy:
    # 1. Generate diverse initial points using the Dyson gas equilibrium
    # 2. Run local optimization from each point with analytical gradients
    # 3. Use DE for global exploration if time permits
    # 4. Final high-precision polish

    # Generate initial points based on Dyson gas equilibrium
    # For linear confining potential, roots satisfy sum 2/(r_i - r_j) = 1
    # Use a simple iterative solver to find approximate equilibrium
    def solve_dyson_equilibrium():
        """Solve the Dyson gas equilibrium condition for 12 roots."""
        # Start with geometric sequence as initial guess
        roots = 4.0 * (2.0 ** np.arange(12))
        
        # Iterate to converge
        for _ in range(1000):
            # Compute the force on each root
            forces = np.zeros(12)
            for i in range(12):
                for j in range(12):
                    if i != j:
                        forces[i] += 2.0 / (roots[i] - roots[j])
            
            # Update: roots should satisfy forces = 1
            # Use simple relaxation
            new_roots = roots.copy()
            for i in range(12):
                # If force > 1, root should move left (decrease)
                # If force < 1, root should move right (increase)
                # Simple proportional update
                new_roots[i] = roots[i] * (1.0 + 0.1 * (1.0 - forces[i]))
            
            # Ensure positivity and monotonicity
            new_roots = np.maximum(new_roots, 0.1)
            new_roots = np.sort(new_roots)
            
            if np.max(np.abs(new_roots - roots)) < 1e-10:
                roots = new_roots
                break
            roots = new_roots
        
        return roots

    dyson_roots = solve_dyson_equilibrium()
    
    # Create multiple diverse initial points
    initial_points = []
    
    # 1. Dyson equilibrium
    initial_points.append(np.log(dyson_roots))
    
    # 2. Geometric sequence (original seed)
    initial_points.append(np.log(4.0 * (2.0 ** np.arange(12))))
    
    # 3. Arithmetic sequence
    arith_roots = np.linspace(1.0, 50.0, 12)
    initial_points.append(np.log(arith_roots))
    
    # 4. Clustering at small values
    cluster_roots = np.concatenate([np.linspace(0.5, 2.0, 6), np.linspace(10.0, 30.0, 6)])
    initial_points.append(np.log(cluster_roots))
    
    # 5. Wide spread
    wide_roots = np.logspace(0.0, 3.0, 12)
    initial_points.append(np.log(wide_roots))
    
    # 6. Tight cluster
    tight_roots = np.linspace(1.0, 3.0, 12)
    initial_points.append(np.log(tight_roots))

    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Local optimization from each initial point
    bounds = [(-3.0, 12.0)] * 12
    
    for i, init in enumerate(initial_points):
        if not time_ok():
            break
        
        try:
            result = minimize(
                cost_func_with_grad,
                init,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # Global search with DE if time permits
    if time_ok():
        seeds = [42, 123, 456, 789, 101, 202, 303, 404]
        for seed in seeds:
            if not time_ok():
                break
            
            try:
                result = differential_evolution(
                    cost_func, 
                    bounds, 
                    seed=seed, 
                    maxiter=100, 
                    popsize=15, 
                    tol=1e-10, 
                    polish=True,
                    init='sobol'
                )
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_log_roots = result.x
                    best_roots = np.exp(best_log_roots)
                    
            except Exception as e:
                pass

    # Final high-precision polish
    if best_log_roots is not None and time_ok():
        try:
            polish_result = minimize(
                cost_func_with_grad,
                best_log_roots,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'disp': False}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive and sorted
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
