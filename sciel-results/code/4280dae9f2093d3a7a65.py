import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s kill
    
    # 1. Define the objective function
    # We want to maximize the product of distances, which is equivalent to maximizing the sum of log distances.
    # Constraint: diameter <= 2.
    # Strategy: Use a smooth penalty for the diameter constraint.
    # Let D_max be the maximum distance. We want D_max <= 2.
    # A smooth approximation to max(x) is logsumexp(x).
    # We can penalize max(0, D_max - 2) using a soft hinge or just penalize (D_max - 2)^2 if D_max > 2.
    # However, the "epigraph" idea suggests introducing a variable t >= max distance.
    # But simpler: just use a smooth penalty for any distance > 2.
    # Penalty = sum(softplus(d_ij - 2)^2)
    
    def log_product(pts):
        # pts is 8x2
        d = np.sqrt(np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=-1))
        np.fill_diagonal(d, 1.0) # avoid log(0)
        if np.any(d < 1e-9):
            return -1e10
        return np.sum(np.log(d))

    def softplus(x):
        # Stable softplus
        return np.where(x > 20, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(0, x))
        # Actually, standard softplus(x) = log(1 + exp(x)).
        # For large x, it is x. For small x, it is log(2).
        # We want to penalize violation of d <= 2.
        # Let's use a smooth penalty: sum(softplus(d_ij - 2)^2)
        
    def objective(x):
        # x is 16 values: x0..y7
        pts = x.reshape(8, 2)
        
        # Calculate distances
        d = np.sqrt(np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=-1))
        np.fill_diagonal(d, 0)
        
        # Log product term
        # Add small epsilon to avoid log(0) if points coincide
        log_prod = np.sum(np.log(d + 1e-12))
        
        # Penalty term for diameter > 2
        # We use a smooth hinge: softplus(d - 2)
        # Squaring it makes it C1 smooth and penalizes larger violations more heavily.
        violations = softplus(d - 2.0)
        penalty = np.sum(violations**2)
        
        # We want to maximize log_prod, so minimize -log_prod + penalty
        # Weight the penalty heavily to ensure constraint satisfaction
        weight = 10.0
        return -log_prod + weight * penalty

    # 2. Initial guesses
    # Start with regular octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    P_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_x = P_oct.flatten().copy()
    best_val = objective(best_x)
    
    # 3. Optimization loop
    # Use L-BFGS-B or SLSQP. L-BFGS-B is faster but doesn't handle non-smoothness as well if penalty isn't smooth enough.
    # softplus is smooth. So L-BFGS-B should work well.
    
    # To escape local minima, we can try multiple restarts with small perturbations.
    # Also, we can try different initial configurations.
    
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Perturb the current best or the octagon
        if i == 0:
            x0 = P_oct.flatten().copy()
        else:
            # Mix of best found and random perturbation
            x0 = best_x + np.random.normal(0, 0.05, 16)
            # Occasionally try a more random start
            if i % 3 == 0:
                x0 = P_oct.flatten() + np.random.normal(0, 0.2, 16)
        
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.success:
            val = res.fun
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        
        # Check time
        if time.time() - start_time > timeout - 10:
            break

    # 4. Final Polish
    # Use SLSQP with tighter tolerances on the best result found
    if time.time() - start_time < timeout - 5:
        res_polish = minimize(objective, best_x, method='SLSQP', options={'maxiter': 500, 'ftol': 1e-14})
        if res_polish.success:
            best_x = res_polish.x

    pts = best_x.reshape(8, 2)
    
    # 5. Scale to diameter 2
    d = np.sqrt(np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=-1))
    np.fill_diagonal(d, 0)
    diam = np.max(d)
    
    if diam > 0:
        pts = pts * (2.0 / diam)
        
    # 6. Verify distinctness
    # If any points are too close, nudge them slightly
    for i in range(8):
        for j in range(i+1, 8):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist < 1e-6:
                # Nudge point j
                pts[j] += np.random.normal(0, 1e-4, 2)
    
    # Final scale check after nudging
    d = np.sqrt(np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=-1))
    np.fill_diagonal(d, 0)
    diam = np.max(d)
    if diam > 0:
        pts = pts * (2.0 / diam)

    return pts.tolist()
