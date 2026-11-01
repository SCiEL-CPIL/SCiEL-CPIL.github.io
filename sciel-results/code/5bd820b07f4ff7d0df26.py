import numpy as np
from scipy.optimize import minimize
import time

N = 32

def _penalty_and_grad(params):
    """
    Computes the penalty objective and its gradient for the circle packing problem.
    Objective: -sum(r) + lambda * (boundary_violation^2 + overlap_violation^2)
    """
    n = N
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]
    
    # 1. Boundary Penalty
    # Violations: x < r, x > 1-r, y < r, y > 1-r
    v_left = r - xy[:, 0]
    v_right = xy[:, 0] + r - 1
    v_bottom = r - xy[:, 1]
    v_top = xy[:, 1] + r - 1
    
    v_left = np.maximum(v_left, 0)
    v_right = np.maximum(v_right, 0)
    v_bottom = np.maximum(v_bottom, 0)
    v_top = np.maximum(v_top, 0)
    
    b_pen = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
    
    # Gradient of boundary penalty
    grad_xy = np.zeros((n, 2))
    grad_r = np.zeros(n)
    
    # d/dx of (r-x)^2 = -2(r-x) if r-x>0
    grad_xy[:, 0] += -2 * v_left
    grad_xy[:, 0] += 2 * v_right
    grad_xy[:, 1] += -2 * v_bottom
    grad_xy[:, 1] += 2 * v_top
    
    # d/dr of (r-x)^2 = 2(r-x) if r-x>0
    grad_r += 2 * v_left
    grad_r += 2 * v_right
    grad_r += 2 * v_bottom
    grad_r += 2 * v_top

    # 2. Overlap Penalty
    # Vectorized distance calculation
    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    dist_sq = dx**2 + dy**2
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    min_dist = r[:, None] + r[None, :]
    
    # Only consider upper triangle for pairs
    iu = np.triu_indices(n, k=1)
    d_ij = dist[iu]
    min_d_ij = min_dist[iu]
    
    # Overlap amount: max(0, min_dist - dist)
    overlap = np.maximum(0, min_d_ij - d_ij)
    
    o_pen = np.sum(overlap**2)
    
    # Gradient of overlap penalty
    # d/dr_i of (min_d - d)^2 = 2 * (min_d - d) if overlap > 0
    # d/dx_i of (min_d - d)^2 = -2 * (min_d - d) * (x_i - x_j) / d  ... wait, 
    # Let f = (r_i + r_j - dist)^2. 
    # d/dx_i f = 2 * (r_i + r_j - dist) * d(-dist)/dx_i = 2 * (r_i + r_j - dist) * (x_i - x_j) / dist
    # Note: dist = sqrt((x_i-x_j)^2 + ...). d(dist)/dx_i = (x_i-x_j)/dist.
    # So d(-dist)/dx_i = -(x_i-x_j)/dist = (x_j-x_i)/dist.
    # Wait, the term is (min_dist - dist). Derivative w.r.t x_i is -d(dist)/dx_i = -(x_i-x_j)/dist.
    # So grad_x_i = 2 * overlap * (-(x_i-x_j)/dist) = -2 * overlap * (x_i-x_j)/dist.
    # Let's check signs. If x_i < x_j, dist increases as x_i increases? No, dist = |x_j - x_i|.
    # If x_i < x_j, dist = x_j - x_i. d(dist)/dx_i = -1.
    # So d(-dist)/dx_i = 1.
    # Formula: grad_x_i = 2 * overlap * (x_j - x_i) / dist.
    
    # We need to scatter the gradients back to the full array.
    # Since we are using L-BFGS-B which expects a flat gradient, we build it carefully.
    
    # Current grad_xy is (n, 2). We need to add contributions from all pairs.
    # For each pair (i, j):
    # grad_xy[i] += -2 * overlap * (xy[i] - xy[j]) / dist
    # grad_xy[j] += -2 * overlap * (xy[j] - xy[i]) / dist
    # grad_r[i] += 2 * overlap
    # grad_r[j] += 2 * overlap
    
    # To do this vectorized, we can use np.add.at or similar, but loops over pairs (496 pairs) are actually quite fast in Python if not inside the inner loop of the optimizer. 
    # However, L-BFGS-B calls the gradient many times. 496 iterations is fine.
    
    # Let's use a loop for the scatter, it's safer and 496 is small.
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        o = overlap[k]
        if o > 0:
            d = d_ij[k]
            # Factor for position
            # grad_x_i = -2 * o * (x_i - x_j) / d
            # grad_x_j = -2 * o * (x_j - x_i) / d = 2 * o * (x_i - x_j) / d
            dx_val = xy[i, 0] - xy[j, 0]
            dy_val = xy[i, 1] - xy[j, 1]
            
            factor = 2.0 * o / d
            
            grad_xy[i, 0] -= factor * dx_val
            grad_xy[i, 1] -= factor * dy_val
            grad_xy[j, 0] += factor * dx_val
            grad_xy[j, 1] += factor * dy_val
            
            grad_r[i] += 2.0 * o
            grad_r[j] += 2.0 * o

    # Combine objective and gradient
    # We want to maximize sum(r), so minimize -sum(r).
    # Objective: -sum(r) + lambda * (b_pen + o_pen)
    lam = 100.0
    
    obj = -np.sum(r) + lam * (b_pen + o_pen)
    
    grad_xy_full = lam * grad_xy
    grad_r_full = -1.0 + lam * grad_r  # -1 from -sum(r), + lam*grad_r from penalty
    
    grad_flat = np.concatenate([grad_xy_full.flatten(), grad_r_full])
    
    return obj, grad_flat

def _is_valid(result, tol=1e-9):
    n = N
    xy = result[:2*n].reshape(n, 2)
    r = result[2*n:]
    
    if np.any(r < -tol):
        return False
    if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
        return False
    if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
        return False
    
    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
        
    return True

def _push_apart(params, n_iters=50):
    """
    Simple iterative push-apart to fix overlaps and boundary violations.
    """
    n = N
    xy = params[:2*n].reshape(n, 2).copy()
    r = params[2*n:].copy()
    
    for _ in range(n_iters):
        moved = False
        # Push apart overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = xy[j, 0] - xy[i, 0]
                dy = xy[j, 1] - xy[i, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[i] + r[j]
                
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    # Push apart by half the overlap each
                    push_x = (dx / dist) * (overlap / 2)
                    push_y = (dy / dist) * (overlap / 2)
                    
                    xy[i, 0] -= push_x
                    xy[i, 1] -= push_y
                    xy[j, 0] += push_x
                    xy[j, 1] += push_y
                    moved = True
                elif dist <= 1e-12:
                    # Random nudge
                    angle = np.random.uniform(0, 2*np.pi)
                    xy[i, 0] -= 0.01 * np.cos(angle)
                    xy[i, 1] -= 0.01 * np.sin(angle)
                    xy[j, 0] += 0.01 * np.cos(angle)
                    xy[j, 1] += 0.01 * np.sin(angle)
                    moved = True
        
        # Fix boundaries
        for i in range(n):
            if xy[i, 0] < r[i]:
                xy[i, 0] = r[i]
                moved = True
            if xy[i, 0] > 1 - r[i]:
                xy[i, 0] = 1 - r[i]
                moved = True
            if xy[i, 1] < r[i]:
                xy[i, 1] = r[i]
                moved = True
            if xy[i, 1] > 1 - r[i]:
                xy[i, 1] = 1 - r[i]
                moved = True
        
        if not moved:
            break
            
    return np.concatenate([xy.flatten(), r])

def _optimize(params, max_iter=100):
    """
    Run L-BFGS-B optimization.
    """
    try:
        result = minimize(
            lambda x: _penalty_and_grad(x)[0], 
            params, 
            jac=lambda x: _penalty_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        return result.x
    except Exception:
        return params

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s
    
    best_result = None
    best_score = -np.inf
    
    def get_score(result):
        if result is None:
            return -np.inf
        if not _is_valid(result):
            return -np.inf
        n = N
        r = result[2*n:]
        return np.sum(r)

    def get_initial_config(restart, rng):
        n = N
        if restart % 5 == 0:
            # 6x6 grid
            side = 6
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.005
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
            
        elif restart % 5 == 1:
            # Hexagonal
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
            
        elif restart % 5 == 2:
            # Random
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.05, 0.08, size=n)
            
        elif restart % 5 == 3:
            # 8x4 Grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.005
            xy += rng.uniform(-0.005, 0.005, size=(n, 2))
            
        else:
            # 4x8 Grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.005
            xy += rng.uniform(-0.005, 0.005, size=(n, 2))
            
        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(42)
    
    # Phase 1: Multi-start optimization
    n_restarts = 100
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        params = get_initial_config(restart, rng)
        
        # Pre-process: push apart to remove gross overlaps
        params = _push_apart(params, n_iters=20)
        
        # Optimize
        optimized = _optimize(params, max_iter=200)
        
        # Post-process: push apart again to ensure validity
        final = _push_apart(optimized, n_iters=50)
        
        score = get_score(final)
        if score > best_score:
            best_score = score
            best_result = final.copy()
            
        # Also check the raw optimized result, sometimes push_apart reduces radii? No, it moves centers.
        # But push_apart doesn't change radii. It might create new overlaps if not careful? 
        # The push_apart logic is conservative.
        
        score_opt = get_score(optimized)
        if score_opt > best_score:
            best_score = score_opt
            best_result = optimized.copy()

    # Phase 2: Polish best solution
    if best_result is not None and time.time() < deadline - 2:
        # Try to improve with higher precision
        polished = _optimize(best_result, max_iter=500)
        final_polished = _push_apart(polished, n_iters=50)
        
        score = get_score(final_polished)
        if score > best_score:
            best_score = score
            best_result = final_polished.copy()
            
        score_pol = get_score(polished)
        if score_pol > best_score:
            best_score = score_pol
            best_result = polished.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        # Final safety clip
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        
        if _is_valid(final_params):
            return np.column_stack([xy, r])
        else:
            # If invalid, try to shrink radii slightly to make it valid
            for _ in range(100):
                r *= 0.98
                final_params = np.concatenate([xy.flatten(), r])
                if _is_valid(final_params):
                    return np.column_stack([xy, r])
            return np.column_stack([xy, r])

    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
