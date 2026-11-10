# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation).
    Uses direct weight optimization via L-BFGS-B on softmax parameters.
    """
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Precompute indices for each bin
    # Projection X - Y: range -9 to 9
    proj_neg1 = X - Y
    # Projection X: range 0 to 9
    proj_0 = X
    # Projection X + Y: range 0 to 18
    proj_1 = X + Y
    # Projection X + 2Y: range 0 to 27
    proj_2 = X + 2 * Y

    # Flatten projections for vectorized binning
    p_neg1 = proj_neg1.flatten() + 9 # shift to 0..18
    p_0 = proj_0.flatten()
    p_1 = proj_1.flatten()
    p_2 = proj_2.flatten()

    def compute_entropies_from_weights(W_flat):
        """
        W_flat: 100-length array of weights
        Returns D, M
        """
        # Reshape W_flat to 10x10
        W = W_flat.reshape(10, 10)
        
        # Compute bin probabilities
        # Using np.bincount is efficient
        probs_neg1 = np.bincount(p_neg1, weights=W.flatten(), minlength=19)
        probs_0 = np.bincount(p_0, weights=W.flatten(), minlength=10)
        probs_1 = np.bincount(p_1, weights=W.flatten(), minlength=19)
        probs_2 = np.bincount(p_2, weights=W.flatten(), minlength=28)
        
        def entropy(probs):
            # Ignore zeros
            probs = probs[probs > 1e-15]
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
            
        D = entropy(probs_neg1)
        M = max(entropy(probs_0), entropy(probs_1), entropy(probs_2))
        
        if M < 1e-9:
            return 0.0, 0.0
        return D, M

    def objective(v):
        """
        v: 100-length array of log-weights (unconstrained)
        Returns negative ratio to be minimized
        """
        # Softmax to get normalized weights
        # Shift for numerical stability
        v_shifted = v - np.max(v)
        W_flat = np.exp(v_shifted)
        W_flat = W_flat / np.sum(W_flat)
        
        D, M = compute_entropies_from_weights(W_flat)
        if M < 1e-9:
            return 0.0
        return -D / M

    def run_optimization(v0):
        # Use L-BFGS-B, it handles large dimensions well
        # Since v is unconstrained, we could use BFGS, but L-BFGS-B is robust
        res = minimize(objective, v0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        return res

    # Try multiple random starts
    best_val = -np.inf
    best_v = None
    
    np.random.seed(42)
    
    # Start 1: Uniform distribution (v=0)
    v0_uniform = np.zeros(100)
    res = run_optimization(v0_uniform)
    if res.fun < best_val:
        best_val = res.fun
        best_v = res.x
        
    # Start 2-5: Random perturbations
    for i in range(5):
        v0_random = np.random.randn(100) * 0.5
        res = run_optimization(v0_random)
        if res.fun < best_val:
            best_val = res.fun
            best_v = res.x
            
    # Convert best_v back to weights
    v_best = best_v
    v_shifted = v_best - np.max(v_best)
    P_flat = np.exp(v_shifted)
    P = P_flat / np.sum(P_flat)
    P = P.reshape(10, 10)
    
    return P

# EVOLVE-BLOCK-END
