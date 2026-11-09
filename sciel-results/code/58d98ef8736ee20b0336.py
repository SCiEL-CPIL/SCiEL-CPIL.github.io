import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Parameterize the 10x10 grid probabilities using log-probabilities (logit space) 
       to naturally enforce positivity and sum-to-1 (via softmax).
    2. Use Sequential Least Squares Programming (SLSQP) for local optimization.
       SLSQP is chosen over L-BFGS-B because it handles the non-linear nature of the 
       entropy ratio more robustly in the presence of potential local minima, and 
       it is highly efficient for N=100.
    3. Employ a "Structured Multi-Start" approach: Instead of random starts, we use 
       50 diverse analytic initializations (uniform, diagonal, anti-diagonal, corner, 
       Gaussian, etc.) to ensure we cover the main basins of attraction.
    4. A deadline guard ensures we return the best valid result found before the 
       600s limit.
    """
    n = 10
    deadline = time.time() + 590.0  # 10s safety margin

    # Grid coordinates
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten
    x_flat = xs.flatten()
    y_flat = ys.flatten()
    
    # Precompute bin indices for entropies
    # D = H(x - y)
    d_val = (x_flat - y_flat)
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    num_d = len(d_vals)
    
    # M = max(H(x), H(x+y), H(x+2y))
    m0_val = x_flat
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    num_m0 = len(m0_vals)
    
    m1_val = (x_flat + y_flat)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    num_m1 = len(m1_vals)
    
    m2_val = (x_flat + 2 * y_flat)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    num_m2 = len(m2_vals)

    def compute_entropies(p_flat):
        """Calculate H_d, H_m0, H_m1, H_m2 from a probability vector."""
        # Normalize just in case (though softmax output is normalized)
        p_flat = p_flat / np.sum(p_flat)
        
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        def shannon(p_dist):
            # Clip to avoid log(0)
            p_nonzero = p_dist[p_dist > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def objective_and_grad(log_p_params):
        """
        Objective function for SLSQP.
        Input: log_p_params (100,)
        Returns: -D/M (to minimize), gradient
        """
        if time.time() > deadline:
            return 1e9, np.zeros(100)
            
        # Softmax for stability
        log_p = log_p_params - np.max(log_p_params)
        p_flat = np.exp(log_p)
        sum_p = np.sum(p_flat)
        if sum_p < 1e-15:
            return 1e9, np.zeros(100)
        p_flat = p_flat / sum_p
        
        H_d, M = compute_entropies(p_flat)
        
        if M < 1e-9:
            return 1e9, np.zeros(100)
        
        ratio = H_d / M
        return -ratio, None # SLSQP can handle None gradient (numerical), but let's try analytic if possible? 
        # For simplicity and robustness, we let SLSQP compute numerical gradients. 
        # 100 dims is small enough for numerical gradients to be fast and accurate.

    # Generate diverse initializations
    def generate_starts():
        starts = []
        # 1. Uniform
        starts.append(np.zeros(100))
        
        # 2. Diagonal heavy (x=y)
        diag = np.zeros(100)
        diag[np.arange(10)*10 + np.arange(10)] = 1.0
        starts.append(np.log(diag + 1e-6))
        
        # 3. Anti-diagonal heavy (x+y=9)
        anti = np.zeros(100)
        for i in range(10):
            anti[i*10 + (9-i)] = 1.0
        starts.append(np.log(anti + 1e-6))
        
        # 4. Corners
        corners = np.zeros(100)
        corners[0] = 1.0
        corners[9] = 1.0
        corners[90] = 1.0
        corners[99] = 1.0
        starts.append(np.log(corners + 1e-6))
        
        # 5. Center Gaussian
        center = np.zeros(100)
        for i in range(10):
            for j in range(10):
                center[i*10+j] = np.exp(-((i-4.5)**2 + (j-4.5)**2) / 5.0)
        starts.append(np.log(center + 1e-6))
        
        # 6. Random Dirichlet (several)
        rng = np.random.default_rng(42)
        for _ in range(45):
            alpha = rng.uniform(0.5, 5.0, 100)
            p_init = rng.dirichlet(alpha)
            starts.append(np.log(p_init + 1e-6))
            
        return starts

    best_score = -np.inf
    best_p_flat = np.ones(100) / 100.0

    starts = generate_starts()
    
    # Sort starts by initial objective to try good ones first? Or just iterate.
    # We'll iterate through them.
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                start, 
                method='SLSQP', 
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < -best_score:
                # Reconstruct P
                log_p = res.x - np.max(res.x)
                p_flat = np.exp(log_p)
                p_flat = p_flat / np.sum(p_flat)
                
                H_d, M = compute_entropies(p_flat)
                if M > 1e-9:
                    score = H_d / M
                    if score > best_score:
                        best_score = score
                        best_p_flat = p_flat
        except Exception:
            continue

    # Final validation and formatting
    P = best_p_flat.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
