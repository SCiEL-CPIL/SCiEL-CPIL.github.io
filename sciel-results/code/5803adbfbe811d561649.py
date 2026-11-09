# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import roots_laguerre
import time

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses the Jacobi matrix method to robustly find roots of L_12^(alpha)(x) for alpha < 0,
    minimizing the largest root (upper bound on C4).
    """
    deadline = time.time() + 570
    
    n = 12

    def get_laguerre_roots_stable(alpha):
        """
        Compute roots of L_n^(alpha)(x) using the eigenvalues of the Jacobi matrix.
        This is numerically stable for alpha > -1.
        """
        # Build the tridiagonal Jacobi matrix
        # J[i,i] = 2*i + alpha + 1
        # J[i,i+1] = sqrt((i+1)*(i+1+alpha))
        J = np.zeros((n, n))
        for i in range(n):
            J[i, i] = 2 * i + alpha + 1
            if i < n - 1:
                val = np.sqrt((i + 1) * (i + 1 + alpha))
                J[i, i + 1] = val
                J[i + 1, i] = val
        
        # Eigenvalues of a real symmetric matrix are real
        roots = np.linalg.eigvalsh(J)
        roots.sort()
        return roots

    def objective(alpha):
        try:
            if alpha <= -0.999:
                return 1e10
            roots = get_laguerre_roots_stable(alpha)
            # The bound is the largest root
            return np.max(roots)
        except Exception:
            return 1e10

    # Search for optimal alpha
    # Roots shift left as alpha decreases.
    # We want to minimize the largest root, so we push alpha as low as possible.
    # However, we must ensure numerical stability and validity.
    # Let's search in [-0.99, 0.5]
    
    # First, check the baseline (alpha=0)
    base_roots = get_laguerre_roots_stable(0.0)
    base_val = np.max(base_roots)
    
    best_alpha = 0.0
    best_val = base_val
    
    # Use bounded minimization
    res = minimize_scalar(objective, bounds=(-0.99, 0.5), method='bounded', options={'xatol': 1e-8})
    
    if res.success:
        # Verify the result is better than baseline
        val_at_res = res.fun
        if val_at_res < best_val:
            best_alpha = res.x
            best_val = val_at_res
            
    # If optimization failed or didn't improve, stick to alpha=0 or try a few manual points
    # Given the monotonicity, alpha=-0.99 should be best.
    # Let's explicitly check alpha = -0.99
    try:
        roots_test = get_laguerre_roots_stable(-0.99)
        val_test = np.max(roots_test)
        if val_test < best_val:
            best_alpha = -0.99
            best_val = val_test
    except Exception:
        pass

    # Get the final roots
    best_roots = get_laguerre_roots_stable(best_alpha)
    
    result = list(best_roots)
    
    # Verify
    assert len(result) == 12
    assert all(r > 0 for r in result)
    assert all(result[i] < result[i+1] for i in range(11))
    
    return result
# EVOLVE-BLOCK-END
