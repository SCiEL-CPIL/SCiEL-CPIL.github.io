import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s buffer for 150s limit
    
    n = 8
    num_pairs = n * (n - 1) // 2
    
    def log_product_minus_scaled_log_diameter(pts, k=100):
        # pts is (n, 2)
        # Calculate all pairwise distances
        # Use broadcasting
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # Extract upper triangle to avoid self-distances and double counting
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        
        # Avoid log(0)
        d = np.maximum(d, 1e-10)
        
        log_prod = np.sum(np.log(d))
        
        # Smoothed diameter
        # sum(d^k)^(1/k)
        # To avoid overflow with large k, use log-sum-exp trick or just compute carefully
        # d is roughly < 10. d^100 is 10^100, which is fine in float64 (max ~1e308)
        sum_dk = np.sum(d ** k)
        log_diam = np.log(sum_dk) / k
        
        # We want to minimize: -log_prod + num_pairs * log_diam
        # Wait, the objective is to maximize log_prod / num_pairs - log_diam?
        # Score is proportional to (Product of dists) / (Diameter)^num_pairs
        # Log Score = sum(log d) - num_pairs * log(Diam)
        # We minimize the negative: -sum(log d) + num_pairs * log(Diam)
        
        return -log_prod + num_pairs * log_diam

    def gradient(pts, k=100):
        # Analytic gradient is complex due to max. 
        # Let's rely on numerical gradient or use the smooth approximation which is differentiable.
        # The smooth approximation is differentiable.
        # But computing it analytically is tedious. 
        # Let's just use the function value and let scipy handle gradients (approx or analytic if possible).
        # Actually, providing the function value is enough for BFGS/L-BFGS-B which approximate gradients.
        # Or we can use 'SLSQP' which also approximates.
        pass

    # Initializations
    best_score = -np.inf
    best_pts = None

    # 1. Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Perturbed Regular Octagon
    rng = np.random.default_rng(42)
    init2 = init1 + rng.normal(0, 0.01, size=init1.shape)
    
    # 3. Antipodal Pairs (4 pairs)
    # Points: (1,0), (-1,0), (0,1), (0,-1), (1,1)/sqrt(2), (-1,-1)/sqrt(2), (1,-1)/sqrt(2), (-1,1)/sqrt(2)
    # This is just the regular octagon. 
    # Let's try a distorted one: 
    # 2 points at (1,0), (-1,0)
    # 2 points at (0,1), (0,-1)
    # 4 points on a smaller circle? No, diameter must be 1.
    # Let's just use random points on the unit circle and random points inside.
    init3 = rng.uniform(-1, 1, size=(n, 2))
    
    # 4. Another random start
    init4 = rng.uniform(-1, 1, size=(n, 2))

    inits = [init1, init2, init3, init4]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Minimize the negative log-score
            # Method: BFGS or L-BFGS-B
            # We don't have bounds, but the function grows if points go to infinity.
            # The term -log_prod dominates if points cluster (log -> -inf, -log -> inf).
            # The term +num_pairs*log_diam dominates if points spread (log_diam -> log(inf)).
            # So the minimum is well-defined.
            
            res = minimize(log_product_minus_scaled_log_diameter, init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.success:
                # Calculate the actual score for this result
                # Score = exp( -res.fun )  ? 
                # res.fun = -log_prod + num_pairs * log_diam
                # -res.fun = log_prod - num_pairs * log_diam = log(Prod / Diam^num_pairs)
                # So Prod / Diam^num_pairs = exp(-res.fun)
                
                # The problem asks for score = Delta / n^n after scaling to diameter 2.
                # Let P be the optimized points.
                # D = max dist in P.
                # P_scaled = P / D * 2.
                # Distances in P_scaled are 2 * d_ij / D.
                # Product in P_scaled = (2/D)^28 * Product in P.
                # Delta = (2/D)^28 * Prod_P.
                # Score = Delta / 8^8.
                
                # Let's just compute the final score directly to be sure.
                pts = res.x.reshape(n, 2)
                
                # Scale to diameter 2
                dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
                iu = np.triu_indices(n, 1)
                d = dists[iu]
                D = np.max(d)
                if D < 1e-9:
                    continue
                pts_scaled = pts / D * 2.0
                
                # Recalculate distances
                dists_scaled = np.linalg.norm(pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :], axis=2)
                d_scaled = dists_scaled[iu]
                
                # Product of distances
                # Use log to avoid underflow/overflow
                log_delta = np.sum(np.log(d_scaled))
                delta = np.exp(log_delta)
                
                score = delta / (n ** n)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts_scaled

        except Exception as e:
            pass

    # If no optimization succeeded, return regular octagon
    if best_pts is None:
        angles = 2 * np.pi * np.arange(n) / n
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.0 # Diameter is 2*sin(pi/8) * 2? No.
        # Regular octagon on unit circle has diameter 2*sin(pi/8)*2? No.
        # Distance between (1,0) and (cos(2pi/8), sin(2pi/8)) is chord length.
        # Max distance is between (1,0) and (-1,0) which is 2.
        # So unit circle regular octagon has diameter 2.
        # This matches the required scaling.
        
    return best_pts.tolist()
