import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Precompute bin indices for each slope
    # P is indexed [y, x] in the return, but let's stick to [x, y] internally for clarity or just map carefully.
    # Task says P[y, x]. So P[i, j] corresponds to y=i, x=j.
    
    # Create coordinate arrays
    ys, xs = np.indices((n, n))
    
    # Calculate values for each slope
    # Slope -1: x - y
    vals_m1 = xs - ys
    # Slope 0: x
    vals_0 = xs
    # Slope 1: x + y
    vals_1 = xs + ys
    # Slope 2: x + 2y
    vals_2 = xs + 2 * ys
    
    # Determine unique values and bin counts
    # For entropy calculation, we need the sum of P in each bin.
    
    def compute_entropies(P_flat):
        """
        P_flat: 1D array of length 100, flattened P[y, x].
        Returns D, M.
        """
        P = P_flat.reshape(n, n)
        
        # Helper to compute entropy of projection
        def entropy_of_projection(vals, P):
            # vals is 2D array of same shape as P
            # We need to sum P for each unique value in vals
            unique_vals, inverse = np.unique(vals, return_inverse=True)
            # Sum P over the inverse indices
            # bincount is efficient
            proj_probs = np.bincount(inverse.ravel(), weights=P.ravel(), minlength=len(unique_vals))
            # Filter out zeros to avoid log(0)
            proj_probs = proj_probs[proj_probs > 1e-15]
            if len(proj_probs) == 0:
                return 0.0
            return -np.sum(proj_probs * np.log2(proj_probs))
        
        D = entropy_of_projection(vals_m1, P)
        H0 = entropy_of_projection(vals_0, P)
        H1 = entropy_of_projection(vals_1, P)
        H2 = entropy_of_projection(vals_2, P)
        M = max(H0, H1, H2)
        
        if M < 1e-9:
            return 0.0, 0.0
        return D, M

    def objective(theta):
        # Softmax to get probabilities
        # Subtract max for stability
        theta_max = np.max(theta)
        exp_theta = np.exp(theta - theta_max)
        P_flat = exp_theta / np.sum(exp_theta)
        
        D, M = compute_entropies(P_flat)
        if M == 0:
            return 0.0
        ratio = D / M
        return -ratio

    # Initializations
    initializations = [
        np.zeros(n*n), # Uniform
        np.random.randn(n*n) * 0.1, # Random
        np.log(np.ones((n,n)) * (1 + np.abs(xs - ys))).flatten(), # Diagonal spread
    ]
    
    best_P = np.ones(n*n) / (n*n)
    best_score = -1
    
    start_time = time.time()
    time_limit = 600 # seconds
    
    for i, init in enumerate(initializations):
        if time.time() - start_time > time_limit - 60:
            break
            
        # SLSQP optimization
        try:
            res = minimize(objective, init, method='SLSQP', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
            
            if res.success or (res.fun < best_score and not np.any(np.isnan(res.x))):
                # Convert back to P
                theta = res.x
                theta_max = np.max(theta)
                P_flat = np.exp(theta - theta_max)
                P_flat = P_flat / np.sum(P_flat)
                
                # Verify validity
                D, M = compute_entropies(P_flat)
                score = D / M if M > 1e-9 else 0
                
                if score > best_score:
                    best_score = score
                    best_P = P_flat
        except Exception:
            pass

    # Final polish with a few more iterations if time allows
    if time.time() - start_time < time_limit - 10:
        try:
            res = minimize(objective, best_P.reshape(-1) if False else np.log(best_P + 1e-15), method='SLSQP', 
                           options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False})
            if res.fun < best_score:
                theta = res.x
                theta_max = np.max(theta)
                P_flat = np.exp(theta - theta_max)
                P_flat = P_flat / np.sum(P_flat)
                D, M = compute_entropies(P_flat)
                if M > 1e-9 and D/M > best_score:
                    best_P = P_flat
        except Exception:
            pass

    # Reshape to (10, 10)
    P_final = best_P.reshape(n, n)
    
    # Ensure constraints
    P_final = np.clip(P_final, 0, None)
    total = np.sum(P_final)
    if total < 1e-9:
        P_final = np.ones((n, n)) / (n*n)
    else:
        P_final = P_final / total
        
    return P_final
