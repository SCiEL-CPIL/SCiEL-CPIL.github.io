# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses a parametric ansatz P(x,y) ∝ exp(-α x² - β y² - γ xy) and optimizes
    (α, β, γ) via differential evolution to maximize D/M where D = H_{-1} and M = max(H_0, H_1, H_2).

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    x_vals = np.arange(n)
    y_vals = np.arange(n)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
    # P[y, x] so rows are y, columns are x
    X_grid = X  # shape (10, 10), X_grid[y, x] = x
    Y_grid = Y  # shape (10, 10), Y_grid[y, x] = y

    def compute_P(alpha, beta, gamma):
        """Compute normalized P given parameters."""
        logP = -alpha * X_grid**2 - beta * Y_grid**2 - gamma * X_grid * Y_grid
        # For numerical stability
        logP -= np.max(logP)
        P = np.exp(logP)
        s = np.sum(P)
        if s <= 0 or not np.isfinite(s):
            return None
        P /= s
        return P

    def entropy_of_projection(P, slope):
        """Compute Shannon entropy in bits of the projection x + slope*y."""
        vals = X_grid + slope * Y_grid
        # Pool by unique values
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        proj_probs = np.zeros(len(unique_vals))
        for i in range(len(unique_vals)):
            mask = (inverse == i)
            proj_probs[i] = np.sum(P[mask])
        # Ignore zero probabilities
        nonzero = proj_probs > 1e-15
        if not np.any(nonzero):
            return 0.0
        p = proj_probs[nonzero]
        p = p / np.sum(p)  # renormalize
        return -np.sum(p * np.log2(p))

    def objective(params):
        """
        Compute -D/M for minimization. 
        Returns a large penalty if invalid.
        """
        alpha, beta, gamma = params
        P = compute_P(alpha, beta, gamma)
        if P is None:
            return 1e6  # Penalty for invalid P
            
        D = entropy_of_projection(P, -1.0)
        H0 = entropy_of_projection(P, 0.0)
        H1 = entropy_of_projection(P, 1.0)
        H2 = entropy_of_projection(P, 2.0)
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return 1e6  # Penalty for invalid M
            
        return -D / M

    # Define bounds for the parameters
    # x, y in [0, 9]. x^2 in [0, 81].
    # To get non-uniform distributions, coefficients should be in a range that makes the exponent vary significantly.
    # If alpha is very small, P is uniform. If very large, P is concentrated.
    # Let's try a wide range to let differential_evolution explore.
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-10.0, 10.0)]
    
    # Use differential evolution for global optimization
    # tol=1e-8, maxiter=200, popsize=15 (default is 15)
    # seed=42 for reproducibility
    result = differential_evolution(objective, bounds, seed=42, maxiter=300, tol=1e-8, polish=True)
    
    best_params = result.x
    best_P = compute_P(*best_params)
    
    if best_P is None:
        # Fallback to uniform
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        # Clip and normalize
        best_P = np.clip(best_P, 0, None)
        s = np.sum(best_P)
        if s <= 1e-9:
            best_P = np.ones((n, n), dtype=np.float64) / (n * n)
        else:
            best_P /= s

    return best_P

# EVOLVE-BLOCK-END
