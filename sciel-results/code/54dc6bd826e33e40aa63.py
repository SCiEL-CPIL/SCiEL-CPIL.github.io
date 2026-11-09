import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where D is entropy of X-Y and M is max entropy of X, X+Y, X+2Y.
    Uses numerical optimization on the probability simplex.
    """
    n = 10
    start_time = time.time()
    timeout = 600
    
    # Precompute projection indices
    # x, y in 0..9
    x_coords, y_coords = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    # P[y, x]
    # Values for projections:
    val_diff = x_coords - y_coords       # X - Y
    val_x = x_coords                    # X
    val_sum = x_coords + y_coords       # X + Y
    val_2y = x_coords + 2 * y_coords    # X + 2Y
    
    # Flatten indices for easier vectorization
    # P is 10x10, flatten to 100
    # val_diff_flat, etc.
    v_diff = val_diff.flatten()
    v_x = val_x.flatten()
    v_sum = val_sum.flatten()
    v_2y = val_2y.flatten()
    
    # Unique values and indices for each projection
    # For entropy calculation, we need to sum P over cells with the same value.
    # We can use np.bincount or similar.
    
    def compute_entropies(P_flat):
        """
        P_flat: array of shape (100,)
        Returns D, H0, H1, H2
        """
        # Ensure P is valid
        P_flat = np.clip(P_flat, 1e-15, None)
        P_flat = P_flat / np.sum(P_flat)
        
        # Helper to compute entropy from values and probabilities
        def entropy_from_values(vals, probs):
            # vals: array of values
            # probs: array of probabilities corresponding to vals
            # Group by value
            unique_vals, inverse_indices = np.unique(vals, return_inverse=True)
            # Sum probabilities for each unique value
            grouped_probs = np.zeros(len(unique_vals))
            np.add.at(grouped_probs, inverse_indices, probs)
            
            # Filter zeros
            mask = grouped_probs > 1e-15
            if not np.any(mask):
                return 0.0
            gp = grouped_probs[mask]
            return -np.sum(gp * np.log2(gp))
        
        D = entropy_from_values(v_diff, P_flat)
        H0 = entropy_from_values(v_x, P_flat)
        H1 = entropy_from_values(v_sum, P_flat)
        H2 = entropy_from_values(v_2y, P_flat)
        
        return D, H0, H1, H2

    def objective(z):
        """
        z: log probabilities (unnormalized)
        Returns negative score to minimize.
        Uses smooth max for M.
        """
        if time.time() - start_time > timeout - 10:
            # Return a large penalty to stop optimization
            return 1e10
            
        P_unnorm = np.exp(z - np.max(z)) # for stability
        P_flat = P_unnorm / np.sum(P_unnorm)
        
        D, H0, H1, H2 = compute_entropies(P_flat)
        
        # Smooth max
        beta = 10.0
        M_smooth = (1.0 / beta) * np.log(np.exp(beta * H0) + np.exp(beta * H1) + np.exp(beta * H2))
        
        # We want to maximize D / M.
        # Approximate by maximizing D - M_smooth.
        # If D > M, ratio > 1.
        return -(D - M_smooth)

    # Initializations
    best_score = -np.inf
    best_P = None
    
    initializations = []
    
    # 1. Uniform
    P_uni = np.ones(100) / 100.0
    initializations.append(np.log(P_uni))
    
    # 2. Anti-diagonal (x+y=9)
    P_anti = np.zeros(100)
    for i in range(10):
        x = i
        y = 9 - i
        P_anti[y * 10 + x] = 1.0 / 10.0
    initializations.append(np.log(P_anti + 1e-15))
    
    # 3. Random sparse
    for _ in range(5):
        P_rand = np.random.rand(100)
        P_rand[P_rand < 0.5] = 0
        P_rand /= np.sum(P_rand)
        initializations.append(np.log(P_rand + 1e-15))
        
    # 4. Diagonal band |x-y| <= 1
    P_band = np.zeros(100)
    for i in range(10):
        for j in range(10):
            if abs(i - j) <= 1:
                P_band[j * 10 + i] = 1.0
    P_band /= np.sum(P_band)
    initializations.append(np.log(P_band + 1e-15))

    for z0 in initializations:
        if time.time() - start_time > timeout - 60:
            break
            
        try:
            res = minimize(objective, z0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success:
                z_opt = res.x
                P_unnorm = np.exp(z_opt - np.max(z_opt))
                P_opt = P_unnorm / np.sum(P_unnorm)
                D, H0, H1, H2 = compute_entropies(P_opt)
                M = max(H0, H1, H2)
                if M > 1e-9:
                    score = D / M
                    if score > best_score:
                        best_score = score
                        best_P = P_opt
        except Exception:
            pass

    if best_P is None:
        # Fallback to uniform
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P.reshape((10, 10))
        
    # Final validation and clipping
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P /= s
    else:
        best_P = np.ones((10, 10)) / 100.0
        
    return best_P
