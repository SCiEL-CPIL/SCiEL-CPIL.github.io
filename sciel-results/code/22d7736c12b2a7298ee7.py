# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check bounds: each square must be inside [-tol, 1+tol]^2
    half = sides / 2.0
    if np.any(centers[:, 0] - half < -tol) or np.any(centers[:, 0] + half > 1 + tol):
        return False
    if np.any(centers[:, 1] - half < -tol) or np.any(centers[:, 1] + half > 1 + tol):
        return False
    
    # Check non-overlap: for i < j, not (|cx_i - cx_j| < t AND |cy_i - cy_j| < t)
    # where t = (s_i + s_j)/2 - tol
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def _objective(params, tol=1e-7):
    """
    Objective function: minimize negative sum of sides + penalty for violations.
    params: [s_0..s_9, cx_0..cx_9, cy_0..cy_9]
    """
    sides_p = params[:10]
    cx = params[10:20]
    cy = params[20:30]
    
    if np.any(sides_p <= 0):
        return 1e10
    
    penalty = 0.0
    half = sides_p / 2.0
    
    # Bound penalties
    penalty += np.sum(np.maximum(0, tol - (cx - half)) * 5000)
    penalty += np.sum(np.maximum(0, (cx + half) - (1 - tol)) * 5000)
    penalty += np.sum(np.maximum(0, tol - (cy - half)) * 5000)
    penalty += np.sum(np.maximum(0, (cy + half) - (1 - tol)) * 5000)
    
    # Overlap penalties
    for i in range(10):
        for j in range(i + 1, 10):
            t = (sides_p[i] + sides_p[j]) / 2.0 - tol
            overlap_x = np.maximum(0, t - abs(cx[i] - cx[j]))
            overlap_y = np.maximum(0, t - abs(cy[i] - cy[j]))
            if overlap_x > 0 and overlap_y > 0:
                penalty += (overlap_x * overlap_y) * 5000
    
    return -np.sum(sides_p) + penalty

def _gradient_descent(centers, sides, max_iters=500, tol=1e-7, start_time=None, deadline=None):
    """
    Custom gradient descent with adaptive step size for the square packing problem.
    Uses subgradient of the penalty function.
    """
    n = 10
    centers = centers.copy().astype(np.float64)
    sides = sides.copy().astype(np.float64)
    
    step_size = 0.01
    min_step = 1e-8
    max_step = 0.1
    patience = 50
    no_improve_count = 0
    best_score = -np.sum(sides)
    
    for it in range(max_iters):
        if start_time is not None and deadline is not None:
            if time.time() > deadline:
                break
        
        # Compute subgradient
        grad_s = np.zeros(n)
        grad_cx = np.zeros(n)
        grad_cy = np.zeros(n)
        
        half = sides / 2.0
        
        # Bound subgradients
        for i in range(n):
            if cx_i := centers[i, 0]:
                pass
            if centers[i, 0] - half[i] < tol:
                grad_cx[i] += 5000
            if centers[i, 0] + half[i] > 1 - tol:
                grad_cx[i] -= 5000
            if centers[i, 1] - half[i] < tol:
                grad_cy[i] += 5000
            if centers[i, 1] + half[i] > 1 - tol:
                grad_cy[i] -= 5000
            # Side bound: if square touches boundary, don't grow
            if centers[i, 0] - half[i] <= tol + 1e-9 or centers[i, 0] + half[i] >= 1 - tol - 1e-9:
                grad_s[i] += 100
            if centers[i, 1] - half[i] <= tol + 1e-9 or centers[i, 1] + half[i] >= 1 - tol - 1e-9:
                grad_s[i] += 100
        
        # Overlap subgradients
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                adx = abs(dx)
                ady = abs(dy)
                
                if adx < t and ady < t:
                    # Overlapping - apply repulsive subgradient
                    pen_x = t - adx
                    pen_y = t - ady
                    
                    # Gradient w.r.t. sides
                    grad_s[i] += 5000 * pen_y * 0.5
                    grad_s[j] += 5000 * pen_x * 0.5
                    
                    # Gradient w.r.t. centers
                    if dx > 0:
                        grad_cx[i] += 5000 * pen_y
                        grad_cx[j] -= 5000 * pen_y
                    elif dx < 0:
                        grad_cx[i] -= 5000 * pen_y
                        grad_cx[j] += 5000 * pen_y
                    else:
                        grad_cx[i] += 5000 * pen_y * 0.1
                        grad_cx[j] -= 5000 * pen_y * 0.1
                    
                    if dy > 0:
                        grad_cy[i] += 5000 * pen_x
                        grad_cy[j] -= 5000 * pen_x
                    elif dy < 0:
                        grad_cy[i] -= 5000 * pen_x
                        grad_cy[j] += 5000 * pen_x
                    else:
                        grad_cy[i] += 5000 * pen_x * 0.1
                        grad_cy[j] -= 5000 * pen_x * 0.1
        
        # Gradient of -sum(sides) is -1 for each side
        grad_s -= 1.0
        
        # Adaptive step size
        norm_grad = np.sqrt(np.sum(grad_s**2) + np.sum(grad_cx**2) + np.sum(grad_cy**2))
        if norm_grad < 1e-10:
            break
        
        # Take step
        new_sides = sides - step_size * grad_s
        new_cx = centers[:, 0] - step_size * grad_cx
        new_cy = centers[:, 1] - step_size * grad_cy
        
        # Clip to valid range
        new_sides = np.maximum(new_sides, 1e-6)
        new_half = new_sides / 2.0
        new_cx = np.clip(new_cx, new_half, 1 - new_half)
        new_cy = np.clip(new_cy, new_half, 1 - new_half)
        
        new_centers = np.stack([new_cx, new_cy], axis=1)
        
        # Check if valid and improved
        if _is_valid(new_centers, new_sides):
            new_score = np.sum(new_sides)
            if new_score > best_score:
                best_score = new_score
                centers = new_centers
                sides = new_sides
                no_improve_count = 0
                step_size = min(max_step, step_size * 1.1)
            else:
                no_improve_count += 1
                step_size = max(min_step, step_size * 0.9)
        else:
            no_improve_count += 1
            step_size = max(min_step, step_size * 0.8)
        
        if no_improve_count > patience:
            break
    
    return centers, sides

def _simulated_annealing(centers, sides, max_iters=10000, tol=1e-7, start_time=None, deadline=None):
    """
    Simulated annealing for square packing.
    """
    n = 10
    centers = centers.copy().astype(np.float64)
    sides = sides.copy().astype(np.float64)
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.9995
    best_score = -np.sum(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    for it in range(max_iters):
        if start_time is not None and deadline is not None:
            if time.time() > deadline:
                break
        
        # Random perturbation
        perturbation = T * 0.5
        new_centers = centers + np.random.randn(n, 2) * perturbation
        new_sides = sides + np.random.randn(n) * perturbation * 0.5
        
        # Clip to valid range
        new_sides = np.maximum(new_sides, 1e-6)
        new_half = new_sides / 2.0
        new_centers[:, 0] = np.clip(new_centers[:, 0], new_half, 1 - new_half)
        new_centers[:, 1] = np.clip(new_centers[:, 1], new_half, 1 - new_half)
        
        # Evaluate
        if _is_valid(new_centers, new_sides):
            new_score = np.sum(new_sides)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                best_sides = new_sides.copy()
            
            # Accept if better or with probability exp(-delta/T)
            current_score = np.sum(sides)
            if new_score >= current_score:
                centers = new_centers
                sides = new_sides
            else:
                delta = current_score - new_score
                if np.random.random() < np.exp(-delta / T):
                    centers = new_centers
                    sides = new_sides
        
        T = max(T_min, T * cooling)
    
    return best_centers, best_sides

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 axis-aligned squares in the unit square.
    
    Uses diverse initial configurations, simulated annealing, gradient descent,
    and local refinement to find near-optimal packings.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10-second safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. 3x3 grid + 1 center
    k = 3
    side = 1.0 / k
    for i in range(9):
        cols = i % k
        rows = i // k
        initial_configs.append({
            'cx': (cols + 0.5) * side,
            'cy': (rows + 0.5) * side,
            's': side
        })
    initial_configs.append({'cx': 0.5, 'cy': 0.5, 's': 0.1})
    
    # 2. 2x5 grid
    k1, k2 = 2, 5
    side1 = 1.0 / k1
    side2 = 1.0 / k2
    for i in range(10):
        cols = i % k2
        rows = i // k2
        initial_configs.append({
            'cx': (cols + 0.5) * side2,
            'cy': (rows + 0.5) * side1,
            's': min(side1, side2)
        })
    
    # 3. 4x3 grid (12 cells, use 10)
    k1, k2 = 4, 3
    side1 = 1.0 / k1
    side2 = 1.0 / k2
    for i in range(10):
        cols = i % k2
        rows = i // k2
        initial_configs.append({
            'cx': (cols + 0.5) * side2,
            'cy': (rows + 0.5) * side1,
            's': min(side1, side2)
        })
    
    # 4. Random configurations
    for _ in range(10):
        cx = rng.uniform(0.15, 0.85, size=10)
        cy = rng.uniform(0.15, 0.85, size=10)
        s = rng.uniform(0.08, 0.3, size=10)
        for i in range(10):
            initial_configs.append({'cx': cx[i], 'cy': cy[i], 's': s[i]})
    
    # 5. Perturbed 3x3 grid
    k = 3
    side = 1.0 / k
    for i in range(9):
        cols = i % k
        rows = i // k
        cx = (cols + 0.5) * side + rng.normal(0, 0.03)
        cy = (rows + 0.5) * side + rng.normal(0, 0.03)
        s = side + rng.normal(0, 0.01)
        initial_configs.append({'cx': np.clip(cx, 0.05, 0.95), 'cy': np.clip(cy, 0.05, 0.95), 's': max(0.05, s)})
    initial_configs.append({'cx': 0.5, 'cy': 0.5, 's': 0.1})
    
    # 6. Perturbed 2x5 grid
    k1, k2 = 2, 5
    side1 = 1.0 / k1
    side2 = 1.0 / k2
    for i in range(10):
        cols = i % k2
        rows = i // k2
        cx = (cols + 0.5) * side2 + rng.normal(0, 0.02)
        cy = (rows + 0.5) * side1 + rng.normal(0, 0.02)
        s = min(side1, side2) + rng.normal(0, 0.01)
        initial_configs.append({'cx': np.clip(cx, 0.05, 0.95), 'cy': np.clip(cy, 0.05, 0.95), 's': max(0.05, s)})
    
    # Process each initial configuration
    for config_idx in range(len(initial_configs)):
        if time.time() > deadline - 10:
            break
        
        centers = np.array([[initial_configs[config_idx * 10 + i]['cx'], 
                           initial_configs[config_idx * 10 + i]['cy']] for i in range(10)], dtype=np.float64)
        sides = np.array([initial_configs[config_idx * 10 + i]['s'] for i in range(10)], dtype=np.float64)
        
        # Check initial validity
        update_best(centers, sides)
        
        # Simulated annealing
        if time.time() < deadline - 20:
            centers_sa, sides_sa = _simulated_annealing(centers, sides, max_iters=5000, 
                                                         start_time=start_time, deadline=deadline)
            update_best(centers_sa, sides_sa)
        
        # Gradient descent from SA result
        if time.time() < deadline - 15:
            centers_gd, sides_gd = _gradient_descent(centers_sa, sides_sa, max_iters=300,
                                                      start_time=start_time, deadline=deadline)
            update_best(centers_gd, sides_gd)
        
        # L-BFGS-B local refinement
        if time.time() < deadline - 10:
            params0 = np.concatenate([sides_gd, centers_gd[:, 0], centers_gd[:, 1]])
            try:
                result = minimize(
                    _objective,
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                params_opt = result.x
                sides_opt = np.maximum(params_opt[:10], 1e-6)
                cx_opt = np.clip(params_opt[10:20], 0, 1)
                cy_opt = np.clip(params_opt[20:30], 0, 1)
                centers_opt = np.stack([cx_opt, cy_opt], axis=1)
                update_best(centers_opt, sides_opt)
            except Exception:
                pass
    
    # Final polishing: run SA + GD + L-BFGS-B from best found
    if best_centers is not None and time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            centers_polish, sides_polish = _simulated_annealing(best_centers, best_sides, 
                                                                 max_iters=3000,
                                                                 start_time=start_time, deadline=deadline)
            update_best(centers_polish, sides_polish)
            
            centers_gd, sides_gd = _gradient_descent(centers_polish, sides_polish, max_iters=200,
                                                      start_time=start_time, deadline=deadline)
            update_best(centers_gd, sides_gd)
            
            params0 = np.concatenate([sides_gd, centers_gd[:, 0], centers_gd[:, 1]])
            try:
                result = minimize(
                    _objective,
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                params_opt = result.x
                sides_opt = np.maximum(params_opt[:10], 1e-6)
                cx_opt = np.clip(params_opt[10:20], 0, 1)
                cy_opt = np.clip(params_opt[20:30], 0, 1)
                centers_opt = np.stack([cx_opt, cy_opt], axis=1)
                update_best(centers_opt, sides_opt)
            except Exception:
                pass
    
    # If no valid solution found, return the grid baseline
    if best_centers is None:
        k = 3
        side = 1.0 / k
        idx = np.arange(9)
        cols = idx % k
        rows = idx // k
        cx = np.concatenate([(cols + 0.5) * side, [0.5]])
        cy = np.concatenate([(rows + 0.5) * side, [0.5]])
        sides = np.concatenate([np.full(9, side), [0.1]])
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = sides.astype(np.float64)
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
