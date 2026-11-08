import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    # Precompute the weight vector for norm calculation
    weight_vec = np.arange(m + 1) + 1  # (k+1) for k=0..m

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.dot(c**2, weight_vec)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq + penalty.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        # Enforce distinctness: penalize if any two roots are too close
        # Sort roots to check adjacent differences
        sorted_roots = np.sort(roots)
        diffs = np.diff(sorted_roots)
        min_diff = np.min(diffs)
        
        # If min_diff is very small, add a large penalty
        # We want roots to be distinct. A relative penalty is better.
        # Let's use a penalty that grows as roots get closer.
        # penalty = sum(1/max(diff, eps)) ? No, that's unstable.
        # Simple approach: if min_diff < threshold, add penalty.
        # Threshold: relative to the scale of roots.
        scale = np.max(roots)
        threshold = 1e-4 * scale
        
        penalty = 0.0
        if min_diff < threshold:
            # Strong penalty to push them apart
            penalty = 1e10 * (1.0 - min_diff / threshold)
        
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10 + penalty
            
        return 1.0 / norm_sq + penalty

    # Seed: geometric sequence a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Tighter bounds based on seed range
    # Seed: 4, 8, 16, ..., 8192 -> log range [1.39, 9.01]
    # Allow some flexibility: [0.5, 10.0]
    bounds = [(0.5, 10.0)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Strategy: 
    # 1. Run a few robust DE iterations to find good basins.
    # 2. Local polish on best candidates.
    # 3. Use Sobol initialization for better coverage.
    
    # Use fewer, more focused DE runs with better settings
    seeds = [42, 123, 456]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s for final polish and fallback
            break
        
        try:
            # Use smaller popsize and more iterations for better convergence
            # popsize=12 is 12*12=144 initial points
            # maxiter=300 allows more exploration
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=300, 
                popsize=12, 
                tol=1e-10, 
                polish=False,  # We'll do our own polish
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback: seed initialization
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Final high-precision polish pass with multiple methods
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            # First try L-BFGS-B
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
        
        # Try Nelder-Mead for additional refinement
        if time.time() - start_time < deadline - 3 and best_log_roots is not None:
            try:
                nm_result = minimize(
                    cost_func, best_log_roots, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15}
                )
                if nm_result.fun < best_val:
                    best_val = nm_result.fun
                    best_log_roots = nm_result.x
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # Ensure roots are positive, distinct, and within reasonable bounds
    if best_roots is not None:
        best_roots = np.clip(best_roots, 1e-6, 1e10)
        
        # Enforce distinctness explicitly if optimization failed to separate them enough
        # Sort and check
        sorted_indices = np.argsort(best_roots)
        sorted_roots = best_roots[sorted_indices]
        
        # If any two are too close, push them apart slightly
        # We do this in the original order by mapping back
        # But it's safer to just ensure the set is distinct.
        # Let's iterate and enforce a minimum gap
        min_gap = 1e-6 * np.max(best_roots)
        for i in range(len(sorted_roots) - 1):
            if sorted_roots[i+1] - sorted_roots[i] < min_gap:
                # Push the second one up
                sorted_roots[i+1] = sorted_roots[i] + min_gap
                # This might cascade, so we do a simple forward pass
                # A more robust way is to just return the seed if this happens often,
                # but the penalty should have prevented this.
        
        # Map back to original order? No, the problem just asks for a list of locations.
        # The order doesn't matter for the product, but we should return a valid list.
        # Let's just return the sorted roots to be safe, or the original roots if they were distinct.
        # Actually, the evaluator likely just checks the set.
        # If we modified sorted_roots, we need to unsort them to return the "best_roots" structure?
        # No, `construct()` returns a list. The order of roots in the product doesn't matter.
        # However, if we return sorted_roots, it's a valid set of roots.
        
        # Let's check if the original best_roots were distinct enough.
        orig_sorted = np.sort(best_roots)
        orig_diffs = np.diff(orig_sorted)
        if np.min(orig_diffs) > 1e-6 * np.max(best_roots):
            return list(best_roots)
        else:
            # Use the adjusted sorted roots
            return list(sorted_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
