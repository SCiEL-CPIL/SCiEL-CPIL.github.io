"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Record start time for deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit (or 150s if that's the hard kill)
    # The prompt says "evaluator KILLS the program after 150s". 
    # "Each evaluation must finish within 120 s" is in the contract.
    # Let's aim for 100s to be safe.
    deadline = start_time + 100.0
    
    n_points = x.size
    
    # 1. Construct Gaussian RBF Basis
    # We use a fixed set of centers to form a basis. 
    # The parameters `params` will be the weights for these basis functions.
    # Since params has length 10, we can use:
    # - 1 bias term
    # - 9 RBF terms (or fewer, but let's use all 9 to maximize capacity)
    
    # Centers: Spread them across the domain [0.5, 2]
    # A 3x3x3 grid would be 27 centers, too many.
    # Let's use 9 centers strategically placed.
    # Maybe a 3x3 grid in the x-y plane, ignoring z for centers? 
    # Or a sparse 3D grid.
    # Let's try a 3x3 grid in (x,y) and center z at 1.25 for all? 
    # No, better to spread in 3D.
    # Let's use 9 centers from a coarse 2x2x2 grid plus center? 
    # 2x2x2 = 8 corners + 1 center = 9 centers.
    
    corners = [0.5, 2.0]
    center_val = 1.25
    
    centers = []
    # 8 corners
    for cx in corners:
        for cy in corners:
            for cz in corners:
                centers.append([cx, cy, cz])
    # 1 center
    centers.append([center_val, center_val, center_val])
    
    centers = np.array(centers)
    n_centers = len(centers)
    
    # Width (sigma) of Gaussians
    # Domain width is 1.5. 
    # If sigma is too small, basis is very local (high frequency).
    # If sigma is too large, basis is smooth (low frequency).
    # Let's use a moderate sigma, e.g., 0.5.
    sigma = 0.5
    
    # Precompute basis matrix B for the training data
    # B[i, j] = exp(-||x_i - c_j||^2 / (2 * sigma^2))
    
    # Reshape data for distance calculation
    X_train = np.column_stack((x, y, z))
    
    # Compute squared distances
    # X_train: (N, 3)
    # Centers: (M, 3)
    # dist_sq = ||X_i - C_j||^2 = ||X_i||^2 + ||C_j||^2 - 2 X_i C_j^T
    
    X_sq = np.sum(X_train**2, axis=1)[:, np.newaxis] # (N, 1)
    C_sq = np.sum(centers**2, axis=1)[np.newaxis, :] # (1, M)
    X_C = X_train @ centers.T # (N, M)
    
    dist_sq = X_sq + C_sq - 2 * X_C
    # Clip to avoid negative due to numerical errors
    dist_sq = np.maximum(dist_sq, 0)
    
    B_train = np.exp(-dist_sq / (2 * sigma**2))
    
    # The model is: pred = params[0] + sum(params[1:10] * B[:, 0:9])
    # We need to map params (length 10) to weights.
    # Let w = params.
    # pred = w[0] + B_train @ w[1:]
    
    # 2. Optimization Strategy
    # We have `params` passed in, but the evaluator fits them.
    # The evaluator calls `minimize` with `x0=np.ones(10)`.
    # Our job is to define the function that `minimize` will optimize.
    # However, `equation` is called *during* the optimization with the current `params`.
    # Wait, the contract says: "The evaluator fits params... minimizing the training MSE."
    # "equation(x, y, z, params) -> np.ndarray"
    # This implies `equation` is the objective function's prediction part.
    # The evaluator handles the `minimize` call.
    
    # BUT, the inspiration says: "implement a local, time-bounded multi-start BFGS optimization inside the equation"
    # If I do that, I am ignoring the `params` passed by the evaluator?
    # The evaluator passes `params` (initially ones).
    # If I ignore `params` and do my own fit, I am effectively returning the best prediction I can find quickly.
    # But the evaluator calls `equation` many times during its *own* BFGS run.
    # If I do a full fit every time `equation` is called, it will be extremely slow and likely timeout.
    
    # Re-reading the contract:
    # "The evaluator fits params... minimizing the training MSE."
    # This suggests the evaluator calls `equation(x, y, z, current_params)` to get predictions, calculates MSE, and updates params.
    # So `equation` should be a fast, deterministic mapping from params to predictions.
    
    # However, the inspirations suggest that the single BFGS call by the evaluator is insufficient or prone to local minima, so one should do *internal* optimization.
    # How can I do internal optimization if `equation` is called repeatedly with different params?
    # Perhaps the strategy is:
    # 1. On the first call (or when params are "default"), perform a robust fit and cache the result?
    # 2. Or, interpret `params` as not just weights, but as a seed for a deterministic pseudo-random search? No, params are continuous.
    
    # Let's look at the "inspiration 1" again: "implement a local, time-bounded multi-start BFGS optimization inside the equation"
    # This is tricky because `equation` is called ~50-100 times by the outer BFGS.
    # If I do a full inner BFGS every time, it will take 100 * (inner time).
    # If inner time is 1s, total 100s. That's risky.
    
    # Alternative interpretation:
    # Maybe I should use `params` to define the basis itself?
    # E.g., params[0:3] are center coordinates for a single RBF?
    # No, params length is 10.
    
    # Let's stick to the standard interpretation: `equation` computes predictions given `params`.
    # The "full_rewrite" strategy might be to change the *basis* such that the linear fit in `params` space is more effective, or to use a non-linear parameterization that BFGS handles better.
    
    # The current program uses a polynomial basis.
    # The "different strategy" is to use a Gaussian RBF basis.
    # Why is this different?
    # Polynomials are global and can suffer from Runge's phenomenon or ill-conditioning.
    # RBFs are local and can model complex shapes.
    # With 9 RBFs + bias, we have 10 parameters.
    # The mapping `params -> prediction` is linear in params.
    # `minimize` will find the least squares solution.
    # The only "optimization" is the least squares fit, which BFGS should handle well if the basis is not ill-conditioned.
    # RBFs can be ill-conditioned if centers are close.
    
    # To make this "materially different", I will use the RBF basis.
    # I will NOT do internal multi-start BFGS because `equation` is called iteratively.
    # Instead, I will ensure the basis is well-conditioned.
    
    # One issue: The evaluator starts params at ones.
    # If the basis is scaled such that the optimal params are near 1, BFGS converges faster.
    # We can center the basis functions.
    # B_centered = B - mean(B)
    # Then the bias term handles the mean.
    
    # Let's compute the centered basis.
    B_mean = np.mean(B_train, axis=0)
    B_centered = B_train - B_mean
    
    # The prediction is:
    # pred = params[0] + sum(params[i] * B_centered[:, i-1])
    # This is still linear in params.
    
    # Is this materially different from the polynomial?
    # Yes, the functional form is different.
    # Polynomials: x, x^2, xy...
    # RBFs: exp(-d^2)
    
    # Let's implement this.
    
    # Check time
    if time.time() > deadline:
        # Fallback: return mean of y (if we had it, but we don't in this function signature? 
        # Wait, the signature is equation(x, y, z, params).
        # 'y' is the input variable y, not the target!
        # The contract says: "Symbolic regression with 3 input(s)... equation(x, y, z, params)"
        # And "Data: ... inputs drawn uniformly ... targets are noiseless."
        # The evaluator fits params to minimize MSE.
        # The function receives x, y, z (inputs) and params.
        # It does NOT receive the target values inside `equation`.
        # So I cannot compute MSE inside `equation`.
        # I just return predictions.
        
        # If I time out, I should return something valid.
        # Return params[0] as a constant prediction.
        return np.full(n_points, params[0])
    
    # Construct prediction
    # params[0] is bias
    # params[1:10] are weights for the 9 RBFs
    
    # Linear combination
    pred = params[0] + B_centered @ params[1:10]
    
    return pred

# EVOLVE-BLOCK-END
