"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using Radial Basis Functions (RBFs).
    
    Strategy:
    Instead of a fixed set of global basis functions (like x, x^2, sin(x)), 
    we use a set of localized Gaussian kernels centered at specific points 
    in the input domain [-3, 3].
    
    This approach is particularly powerful for symbolic regression because:
    1. Localized bases can approximate sharp features, discontinuities (in the 
       limit), or complex oscillatory behaviors better than low-degree polynomials.
    2. The parameters are linear coefficients, so the optimization landscape 
       is convex (quadratic) with respect to params, which BFGS handles very well.
    3. We include a constant term to handle the mean offset.
    
    We choose 9 centers evenly spaced in [-2.5, 2.5] to cover the domain [-3, 3] 
    with some margin, plus 1 constant term. Total 10 params.
    
    Centers: -2.5, -1.875, -1.25, -0.625, 0.0, 0.625, 1.25, 1.875, 2.5
    Width (sigma): 0.5. This provides a balance between local flexibility and 
    smoothness. A smaller sigma makes the basis more local (higher frequency 
    content), while a larger sigma makes it more global. 0.5 is a reasonable 
    default for a domain of width 6.
    
    The basis matrix is computed on the fly. Since x is fixed during the 
    optimization loop (the evaluator calls equation(x, params) repeatedly with 
    the same x but different params), we could cache the basis, but for 
    simplicity and to avoid statefulness issues across calls (if x changes 
    between test/train or if the evaluator resets state), we recompute it. 
    However, recomputation is O(N * K) which is fast for N=200, K=10.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure p has length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Define centers and width
    # 9 centers + 1 constant = 10 params
    centers = np.linspace(-2.5, 2.5, 9)
    sigma = 0.5
    
    # Compute RBF basis
    # shape: (N, 9)
    # diff: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    # Add constant term as the first column
    # We map params[0] to constant, params[1:10] to RBFs
    # Or we can just stack. Let's say:
    # param[0] * 1
    # param[1] * rbf[:,0]
    # ...
    # param[9] * rbf[:,8]
    
    # Construct basis matrix (N, 10)
    basis = np.column_stack([np.ones_like(x), rbf])
    
    pred = basis @ p
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
