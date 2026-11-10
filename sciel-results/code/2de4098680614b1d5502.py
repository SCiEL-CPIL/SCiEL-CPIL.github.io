import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))

def _energy_and_grad(x, n=10, tol=1e-7):
    """
    Computes the energy (negative sum of sides + penalties) and its gradient.
    x is a flattened array of length 3n: [cx0, cy0, s0, cx1, cy1, s1, ...]
    We minimize E = -sum(sides) + K * (overlap_penalty + boundary_penalty)
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Ensure sides are positive for calculation stability, though bounds handle this
    sides_safe = np.maximum(sides, 1e-6)
    
    energy = -np.sum(sides)
    grad = np.zeros_like(x)
    
    # Gradient of objective: d(-sum(s))/ds_i = -1
    grad[2*n:] -= 1.0
    
    # Penalty parameters
    K = 100.0  # Penalty weight
    alpha = 10.0 # Sharpness of soft min/max
    
    # 1. Boundary Penalties
    # For each square i, we want cx_i - s_i/2 >= 0 and cx_i + s_i/2 <= 1
    # Penalty: soft_min(0, boundary_violation)^2
    # Let v = min(0, cx - s/2, 1 - cx - s/2, cy - s/2, 1 - cy - s/2)
    # Actually, easier to just penalize violations directly with smooth functions.
    
    for i in range(n):
        cx, cy, s = centers[i, 0], centers[i, 1], sides_safe[i]
        
        # Left boundary: cx - s/2 should be >= 0. Violation v = -(cx - s/2)
        v_left = -(cx - s/2)
        if v_left > 0:
            pen = v_left ** 2
            # d(pen)/d(cx) = -2*v_left
            # d(pen)/d(s) = 2*v_left * 0.5
            grad[2*i] += K * (-2 * v_left)
            grad[2*n + i] += K * (2 * v_left * 0.5)
            energy += K * pen
            
        # Right boundary: cx + s/2 should be <= 1. Violation v = (cx + s/2) - 1
        v_right = (cx + s/2) - 1
        if v_right > 0:
            pen = v_right ** 2
            grad[2*i] += K * (2 * v_right)
            grad[2*n + i] += K * (2 * v_right * 0.5)
            energy += K * pen
            
        # Bottom boundary: cy - s/2 should be >= 0. Violation v = -(cy - s/2)
        v_bot = -(cy - s/2)
        if v_bot > 0:
            pen = v_bot ** 2
            grad[2*i + 1] += K * (-2 * v_bot)
            grad[2*n + i] += K * (2 * v_bot * 0.5)
            energy += K * pen
            
        # Top boundary: cy + s/2 should be <= 1. Violation v = (cy + s/2) - 1
        v_top = (cy + s/2) - 1
        if v_top > 0:
            pen = v_top ** 2
            grad[2*i + 1] += K * (2 * v_top)
            grad[2*n + i] += K * (2 * v_top * 0.5)
            energy += K * pen

    # 2. Overlap Penalties
    # For pair (i, j), overlap if |dx| < t AND |dy| < t, where t = (si + sj)/2 - tol
    # We use a smooth approximation for the "AND" condition and the max(0, t - dist)
    # Let dx = cx_i - cx_j, dy = cy_i - cy_j
    # t = (s_i + s_j)/2 - tol
    # pen_x = max(0, t - |dx|)
    # pen_y = max(0, t - |dy|)
    # Overlap penalty = pen_x * pen_y (this is 0 if either is 0, and positive if both > 0)
    # This is differentiable everywhere except where pen_x or pen_y is 0 (kink), 
    # but subgradient is 0 there, so it's fine for L-BFGS.
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides_safe[i] + sides_safe[j]) / 2 - tol
            
            # pen_x = max(0, t - |dx|)
            adx = abs(dx)
            pen_x = max(0.0, t - adx)
            # pen_y = max(0, t - |dy|)
            ady = abs(dy)
            pen_y = max(0.0, t - ady)
            
            if pen_x > 0 and pen_y > 0:
                pen = pen_x * pen_y
                energy += K * pen
                
                # Gradients
                # d(pen)/d(dx) = d(pen_x * pen_y)/d(dx) = (d pen_x/d dx) * pen_y
                # d pen_x/d dx = -sign(dx) if pen_x > 0 else 0
                sign_dx = 1.0 if dx > 0 else -1.0
                d_pen_dx = -sign_dx * pen_y
                d_pen_dy = -sign_dy * pen_x if False else 0 # placeholder, need sign_dy
                
                sign_dy = 1.0 if dy > 0 else -1.0
                d_pen_dy = -sign_dy * pen_x
                
                # d(pen)/d(t) = pen_x + pen_y
                # d(t)/d(s_i) = 0.5, d(t)/d(s_j) = 0.5
                
                # Chain rule to centers and sides
                # d/d(cx_i) = d_pen_dx
                # d/d(cx_j) = -d_pen_dx
                # d/d(cy_i) = d_pen_dy
                # d/d(cy_j) = -d_pen_dy
                # d/d(s_i) = d_pen_dt * 0.5
                # d/d(s_j) = d_pen_dt * 0.5
                
                d_pen_dt = pen_x + pen_y
                
                grad[2*i] += K * d_pen_dx
                grad[2*j] -= K * d_pen_dx
                grad[2*i + 1] += K * d_pen_dy
                grad[2*j + 1] -= K * d_pen_dy
                grad[2*n + i] += K * d_pen_dt * 0.5
                grad[2*n + j] += K * d_pen_dt * 0.5

    return energy, grad

def _project_feasible(centers, sides, tol=1e-7):
    """
    Simple projection to make solution strictly feasible.
    1. Clamp sides to (0, 1].
    2. Clamp centers to stay within [s/2, 1-s/2].
    3. Iterative separation for overlaps.
    """
    n = len(sides)
    centers = centers.copy().astype(np.float64)
    sides = sides.copy().astype(np.float64)
    
    sides = np.clip(sides, 0.001, 1.0)
    
    for i in range(n):
        centers[i, 0] = max(sides[i] / 2, min(1 - sides[i] / 2, centers[i, 0]))
        centers[i, 1] = max(sides[i] / 2, min(1 - sides[i] / 2, centers[i, 1]))
    
    # Iterative overlap resolution
    for _ in range(100):
        resolved = True
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                adx = abs(dx)
                ady = abs(dy)
                
                if adx < t and ady < t:
                    resolved = False
                    pen_x = t - adx
                    pen_y = t - ady
                    
                    if pen_x < pen_y:
                        push = pen_x / 2 + 1e-9
                        if dx >= 0:
                            centers[i, 0] += push
                            centers[j, 0] -= push
                        else:
                            centers[i, 0] -= push
                            centers[j, 0] += push
                    else:
                        push = pen_y / 2 + 1e-9
                        if dy >= 0:
                            centers[i, 1] += push
                            centers[j, 1] -= push
                        else:
                            centers[i, 1] -= push
                            centers[j, 1] += push
                    
                    for k in [i, j]:
                        centers[k, 0] = max(sides[k] / 2, min(1 - sides[k] / 2, centers[k, 0]))
                        centers[k, 1] = max(sides[k] / 2, min(1 - sides[k] / 2, centers[k, 1]))
        if resolved:
            break
            
    # Final safety shrink if still invalid
    if not _is_valid(centers, sides, tol):
        for i in range(n):
            for _ in range(50):
                if _is_valid(centers, sides, tol):
                    break
                sides[i] *= 0.95
                centers[i, 0] = max(sides[i] / 2, min(1 - sides[i] / 2, centers[i, 0]))
                centers[i, 1] = max(sides[i] / 2, min(1 - sides[i] / 2, centers[i, 1]))
                
    return centers, sides

def _generate_seeds():
    """Generate diverse initial configurations."""
    seeds = []
    
    # 1. Grid
    c = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                  [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3]])
    s = np.full(10, 0.18)
    seeds.append((c, s))
    
    # 2. Heterogeneous: 1 large, 9 small
    c = np.array([[0.25, 0.25],
                  [0.75, 0.25], [0.5, 0.25], [0.25, 0.5],
                  [0.75, 0.5], [0.5, 0.5], [0.25, 0.75],
                  [0.75, 0.75], [0.5, 0.75], [0.5, 0.9]])
    s = np.array([0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])
    seeds.append((c, s))
    
    # 3. Random
    rng = np.random.default_rng(42)
    s = rng.uniform(0.1, 0.3, size=10)
    c = np.zeros((10, 2))
    for i in range(10):
        c[i, 0] = rng.uniform(s[i]/2, 1 - s[i]/2)
        c[i, 1] = rng.uniform(s[i]/2, 1 - s[i]/2)
    seeds.append((c, s))
    
    # 4. Corner packing
    c = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8],
                  [0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5],
                  [0.5, 0.5], [0.5, 0.6]])
    s = np.array([0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])
    seeds.append((c, s))
    
    return seeds

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize sum of side lengths.
    Uses gradient-based optimization (L-BFGS-B) on a differentiable energy landscape.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 570
    
    seeds = _generate_seeds()
    
    best_centers = None
    best_sides = None
    best_val = -float('inf')
    
    for seed_idx, (centers, sides) in enumerate(seeds):
        if time.time() > deadline - 30:
            break
            
        # Ensure seed is valid
        if not _is_valid(centers, sides, tol):
            centers, sides = _project_feasible(centers, sides, tol)
            if not _is_valid(centers, sides, tol):
                continue
                
        # Flatten for optimizer
        x0 = np.concatenate([centers.flatten(), sides])
        
        # Bounds
        # cx, cy in [0, 1]
        # s in [0.01, 1.0]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # cx
            bounds.append((0.0, 1.0)) # cy
        for i in range(n):
            bounds.append((0.01, 1.0)) # s
            
        # Optimization
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 50}
            )
            
            x_opt = res.x
            c_opt = x_opt[:2*n].reshape(n, 2)
            s_opt = x_opt[2*n:]
            
            # Project to ensure strict validity
            c_proj, s_proj = _project_feasible(c_opt, s_opt, tol)
            
            if _is_valid(c_proj, s_proj, tol):
                val = _objective(c_proj, s_proj)
                if val > best_val:
                    best_val = val
                    best_centers = c_proj.copy()
                    best_sides = s_proj.copy()
                    
        except Exception as e:
            pass
            
        # Also check the seed itself
        if _is_valid(centers, sides, tol):
            val = _objective(centers, sides)
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_sides = sides.copy()
                
    # If no valid result, fall back to a simple valid grid
    if best_centers is None:
        best_centers = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                                 [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3]], dtype=np.float64)
        best_sides = np.full(10, 0.15, dtype=np.float64)
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
