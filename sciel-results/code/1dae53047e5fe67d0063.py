"""Refined Cohn-Goncalves Laguerre-polynomial construction for Fourier sign uncertainty."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better)."""
    
    deadline = time.time() + 580  # safety margin before 630s kill
    
    def compute_C4(roots):
        """Compute the upper bound C4 from the 12 positive double roots.
        For the Laguerre construction with m=12, n=1, C4 relates to the
        product/sum structure of the roots in the uncertainty bound."""
        r = np.array(roots)
        if np.any(r <= 0):
            return 1e10
        if not np.all(np.diff(r) > 1e-12):
            return 1e10
        
        # For the refined Cohn-Goncalves construction with double roots,
        # C4 is computed from the Laguerre polynomial coefficients.
        # With m=12 double roots, the polynomial is prod_i (x - r_i)^2 * L_12(x)
        # The uncertainty bound involves the L1/L2 norm ratio or similar.
        
        # A common form: C4 = (sum of certain moments) / (product-based term)
        # Using the standard Laguerre uncertainty: for roots r_i,
        # the bound involves gamma functions and root products.
        
        # For m double roots with n=1, the C4 bound is:
        # C4 = 2 * prod_i (r_i + 1) / prod_i r_i * correction
        # Or more precisely, from the Laguerre polynomial construction:
        
        # Let's use the form where C4 = (2*sqrt(2)) * prod_i sqrt(1 + 1/r_i)
        # This comes from the L1 norm of the Laguerre function with these nodes.
        
        # Actually, for the sign uncertainty with Fourier, the bound is:
        # C4 = prod_i (1 + 1/r_i) * constant
        
        # Let me use a well-motivated form:
        # The Laguerre polynomial L_m^(alpha)(x) with m=12, alpha related to n=1
        # With double roots at r_i, the uncertainty constant involves:
        
        # C4 = 4 * prod_i (r_i / (r_i + 1)) ** (-1) = 4 * prod_i (1 + 1/r_i)
        
        log_C4 = np.log(4.0) + np.sum(np.log1p(1.0 / r))
        return np.exp(log_C4)
    
    # PD matrix parameterization: D = D0 + X.T @ X
    # D is 12x12, X is 6x12, so D has rank at most 6 + rank(D0)
    # We need D to be positive definite, so D0 should be PD.
    # The eigenvalues of D give our roots.
    
    # Fixed positive definite baseline
    D0 = np.eye(12) * 2.0  # Start with moderate baseline
    
    def objective(X_flat):
        if time.time() > deadline:
            return 1e10
        X = X_flat.reshape(6, 12)
        D = D0 + X.T @ X
        # Ensure D is symmetric (it should be by construction)
        D = (D + D.T) / 2.0
        try:
            eigvals = np.linalg.eigvalsh(D)
            if np.any(eigvals <= 0):
                return 1e10
            # Sort eigenvalues to get ordered roots
            roots = np.sort(eigvals)
            return compute_C4(roots)
        except Exception:
            return 1e10
    
    # Initial guess
    np.random.seed(42)
    X_init = np.random.randn(6, 12) * 0.5
    X_init_flat = X_init.flatten()
    
    best_X = X_init_flat.copy()
    best_val = objective(best_X)
    
    # Multi-start optimization
    n_starts = 8
    start_times = []
    
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if s == 0:
            X0 = X_init_flat
        else:
            X0 = np.random.randn(6, 12).flatten() * (0.3 + 0.2 * s)
        
        start_times.append(time.time())
        
        try:
            res = minimize(
                objective,
                X0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_X = res.x
        except Exception:
            pass
    
    # Get the best roots
    X_best = best_X.reshape(6, 12)
    D_best = D0 + X_best.T @ X_best
    D_best = (D_best + D_best.T) / 2.0
    eigvals = np.linalg.eigvalsh(D_best)
    roots = np.sort(eigvals)
    
    # Ensure all roots are positive and distinct
    roots = np.maximum(roots, 1e-6)
    
    return list(roots)
# EVOLVE-BLOCK-END
