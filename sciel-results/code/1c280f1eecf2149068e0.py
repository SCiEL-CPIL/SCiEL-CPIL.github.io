# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


def _is_valid(centers, sides, tol=1e-7):
    """Check if the placement is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check boundaries
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            cx_i, cy_i = centers[i]
            cx_j, cy_j = centers[j]
            s_i, s_j = sides[i], sides[j]
            t = (s_i + s_j) / 2 - tol
            if abs(cx_i - cx_j) < t and abs(cy_i - cy_j) < t:
                return False
    return True


def _objective(x, n=10, lambda_pen=100.0):
    """
    Objective function for optimization.
    We want to MAXIMIZE sum(sides).
    scipy.optimize.differential_evolution MINIMIZES.
    So we return -sum(sides) + penalty.
    
    x layout: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    """
    centers = x[0:3*n:3].reshape(-1, 2) # Wait, this slicing is wrong.
    # Correct extraction:
    # x[0], x[1], x[2] -> cx0, cy0, s0
    # x[3], x[4], x[5] -> cx1, cy1, s1
    centers = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
    sides = np.array([x[3*i+2] for i in range(n)])
    
    # Ensure sides are positive for calculation stability, though bounds handle it
    sides = np.maximum(sides, 1e-9)
    
    total_side = np.sum(sides)
    
    penalty = 0.0
    
    # Boundary penalty
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Distance outside
        dx = max(0, -tol + (cx - s/2), (cx + s/2) - (1 + tol)) # This logic is flawed, let's do it properly
        # Left violation
        v_l = max(0, -tol - (cx - s/2))
        # Right violation
        v_r = max(0, (cx + s/2) - (1 + tol))
        # Bottom violation
        v_b = max(0, -tol - (cy - s/2))
        # Top violation
        v_t = max(0, (cy + s/2) - (1 + tol))
        penalty += v_l**2 + v_r**2 + v_b**2 + v_t**2
        
    # Overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            cx_i, cy_i = centers[i]
            cx_j, cy_j = centers[j]
            s_i, s_j = sides[i], sides[j]
            t = (s_i + s_j) / 2 - tol
            # If they overlap, the distance in x AND y is less than t
            # We want to push them apart.
            # A common smooth penalty is max(0, t - |dx|)^2 * max(0, t - |dy|)^2 ? No, that's product.
            # Simpler: If |dx| < t and |dy| < t, we have overlap.
            # Penalty should be large when they are deep inside each other.
            # Let's use: if |dx| < t and |dy| < t, penalty += (t - |dx|)^2 * (t - |dy|)^2? 
            # Or simpler: penalty += max(0, t - |dx|)^2 + max(0, t - |dy|)^2? 
            # The latter allows "sliding" along one axis while violating the other.
            # The product form is better for "separating" them.
            # Actually, the standard "non-overlap" constraint is:
            # |dx| >= t OR |dy| >= t.
            # Violation is when |dx| < t AND |dy| < t.
            # Let px = max(0, t - abs(cx_i - cx_j))
            # Let py = max(0, t - abs(cy_i - cy_j))
            # If both > 0, they overlap.
            # Penalty = px * py is a good measure of the "area" of overlap in the constraint space.
            px = max(0, t - abs(cx_i - cx_j))
            py = max(0, t - abs(cy_i - cy_j))
            if px > 0 and py > 0:
                penalty += px * py
                
    return -total_side + lambda_pen * penalty


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # Baseline: 3x3 grid
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    best_sides = np.full(n, side, dtype=np.float64)
    best_value = float(np.sum(best_sides))
    
    # Define bounds for DE
    # cx, cy in [0, 1]
    # s in [0.01, 1.0] (s must be > 0, max 1)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((1e-3, 1.0)) # s
        
    # Run Differential Evolution
    # We use a moderate population size and max iterations to balance quality and time
    # sigma=0.5 for mutation scale
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            args=(n, 100.0),
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            polish=False, # We will do our own polish
            updating='immediate'
        )
        
        if res.success and res.x is not None:
            x_best = res.x
            centers = np.array([[x_best[3*i], x_best[3*i+1]] for i in range(n)])
            sides = np.array([x_best[3*i+2] for i in range(n)])
            
            # Refine with local optimization (L-BFGS-B)
            # We need a smooth objective for this too
            def _local_obj(x):
                return _objective(x, n, 100.0)
            
            # Run a few local minimizations from the best DE point
            try:
                res_local = minimize(
                    _local_obj, 
                    x_best, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                if res_local.success:
                    x_final = res_local.x
                    centers = np.array([[x_final[3*i], x_final[3*i+1]] for i in range(n)])
                    sides = np.array([x_final[3*i+2] for i in range(n)])
            except Exception:
                pass
            
            if _is_valid(centers, sides):
                value = float(np.sum(sides))
                if value > best_value:
                    best_value = value
                    best_centers = centers.copy()
                    best_sides = sides.copy()
                    
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Additional random restarts if time permits
    if time.time() < deadline - 30:
        np.random.seed(123)
        for restart in range(5):
            if time.time() > deadline:
                break
            # Random initial population for DE? Or just a random start for local?
            # Let's try a random start and local polish
            x_rand = np.random.uniform(0, 1, 3*n)
            # Ensure sides are small enough to fit? No, let local optimizer handle it.
            # But local optimizer might get stuck in invalid regions.
            # Let's just run a shorter DE with different seed
            try:
                res = differential_evolution(
                    _objective, 
                    bounds, 
                    args=(n, 100.0),
                    seed=100+restart,
                    maxiter=50,
                    popsize=10,
                    tol=1e-6,
                    polish=False
                )
                if res.success:
                    x_best = res.x
                    centers = np.array([[x_best[3*i], x_best[3*i+1]] for i in range(n)])
                    sides = np.array([x_best[3*i+2] for i in range(n)])
                    
                    def _local_obj(x):
                        return _objective(x, n, 100.0)
                        
                    res_local = minimize(
                        _local_obj, 
                        x_best, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12}
                    )
                    if res_local.success:
                        x_final = res_local.x
                        centers = np.array([[x_final[3*i], x_final[3*i+1]] for i in range(n)])
                        sides = np.array([x_final[3*i+2] for i in range(n)])
                        
                        if _is_valid(centers, sides):
                            value = float(np.sum(sides))
                            if value > best_value:
                                best_value = value
                                best_centers = centers.copy()
                                best_sides = sides.copy()
            except Exception:
                pass

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
