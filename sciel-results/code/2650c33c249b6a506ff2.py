"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

DEADLINE = 115.0  # Safety margin for 120s limit

def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    
    # Check perimeter constraint
    if 2 * (W + H) > 4 + 1e-9:
        return False
        
    if W <= 0 or H <= 0:
        return False

    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Check radii positive
    if np.any(rs <= 0):
        return False

    # Check boundary
    # left >= 0, right <= W, bottom >= 0, top <= H
    if np.min(xs - rs) < -1e-9 or np.max(xs + rs) > W + 1e-9:
        return False
    if np.min(ys - rs) < -1e-9 or np.max(ys + rs) > H + 1e-9:
        return False

    # Check disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True

def _compute_score(circles, W, H):
    """Compute sum of radii if valid, else 0."""
    if not _check_validity(circles, W, H):
        return 0.0
    return np.sum(circles[:, 2])

def _smooth_max_grad(v, eps=1e-4):
    """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
    smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
    gsmax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
    return smax, gsmax

def _objective_and_grad(params, boundary_weight=200.0, overlap_weight=200.0, eps=1e-4):
    """
    Objective: minimize -sum(r) + penalties for violations.
    params: [W, H, x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    Returns (obj, grad) with analytical gradient.
    """
    W = params[0]
    H = params[1]
    circles = params[2:].reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Ensure W, H are positive for calculation stability, though bounds handle this
    W_safe = max(W, 1e-6)
    H_safe = max(H, 1e-6)

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    
    # d(-sum(r))/dr = -1
    grad[2 + 2::3] = -1.0 # Indices: 2 (x1), 3 (y1), 4 (r1) -> r is at 2+2=4, then +3... 
    # Wait, params layout:
    # 0: W, 1: H
    # 2: x1, 3: y1, 4: r1
    # 5: x2, 6: y2, 7: r2
    # So r indices are 4, 7, 10... which is 2 + 2 + 3*i.
    # Let's just use slicing carefully.
    # r is at indices 4, 7, ...
    r_indices = np.arange(4, len(params), 3)
    grad[r_indices] = -1.0

    # Perimeter constraint: 2(W + H) <= 4 => W + H <= 2
    # Violation: W + H - 2
    v_per = W + H - 2.0
    smax_per, gsmax_per = _smooth_max_grad(v_per, eps)
    pen_per = 1000.0 * smax_per ** 2
    obj += pen_per
    # d/dW and d/dH
    grad[0] += 2.0 * 1000.0 * smax_per * gsmax_per
    grad[1] += 2.0 * 1000.0 * smax_per * gsmax_per

    # Boundary penalties for circles
    # 1. x - r >= 0 => v = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    # d/dx: -2 * bw * smax * gsmax
    # d/dr:  2 * bw * smax * gsmax
    x_indices = np.arange(2, len(params), 3)
    r_indices = np.arange(4, len(params), 3)
    grad[x_indices] += -2.0 * boundary_weight * smax * gsmax
    grad[r_indices] += 2.0 * boundary_weight * smax * gsmax

    # 2. y - r >= 0 => v = r - y
    v = r - y
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    y_indices = np.arange(3, len(params), 3)
    grad[y_indices] += -2.0 * boundary_weight * smax * gsmax
    grad[r_indices] += 2.0 * boundary_weight * smax * gsmax

    # 3. x + r <= W => v = x + r - W
    v = x + r - W_safe
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    # d/dx: 2 * bw * smax * gsmax
    # d/dr: 2 * bw * smax * gsmax
    # d/dW: -sum(2 * bw * smax * gsmax)
    grad[x_indices] += 2.0 * boundary_weight * smax * gsmax
    grad[r_indices] += 2.0 * boundary_weight * smax * gsmax
    grad[0] += -np.sum(2.0 * boundary_weight * smax * gsmax)

    # 4. y + r <= H => v = y + r - H
    v = y + r - H_safe
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[y_indices] += 2.0 * boundary_weight * smax * gsmax
    grad[r_indices] += 2.0 * boundary_weight * smax * gsmax
    grad[1] += -np.sum(2.0 * boundary_weight * smax * gsmax)

    # 5. r >= 0 => v = -r
    v = -r
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[r_indices] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dist = 1e-12
                dx = 1e-12
                dy = 0.0
            
            v = r[i] + r[j] - dist
            smax, gsmax = _smooth_max_grad(v, eps)
            obj += overlap_weight * smax ** 2

            factor = 2.0 * overlap_weight * smax * gsmax
            
            # Gradients for x_i, y_i, r_i, x_j, y_j, r_j
            # d(v)/dx_i = -dx/dist
            # d(v)/dy_i = -dy/dist
            # d(v)/dr_i = 1
            # d(v)/dx_j = dx/dist
            # d(v)/dy_j = dy/dist
            # d(v)/dr_j = 1
            
            idx_xi = 2 + 3*i
            idx_yi = 3 + 3*i
            idx_ri = 4 + 3*i
            idx_xj = 2 + 3*j
            idx_yj = 3 + 3*j
            idx_rj = 4 + 3*j
            
            grad[idx_xi] += factor * (-dx / dist)
            grad[idx_yi] += factor * (-dy / dist)
            grad[idx_ri] += factor * 1.0
            
            grad[idx_xj] += factor * (dx / dist)
            grad[idx_yj] += factor * (dy / dist)
            grad[idx_rj] += factor * 1.0

    return obj, grad

def _generate_initial_guesses(W, H, num_guesses=10):
    """Generate diverse initial configurations."""
    guesses = []
    
    # 1. Grid-based
    # Try a 5x4 grid (20 circles) + 1 center, or 5x5 (25) take 21
    side = 5
    margin = 0.05
    w_eff = W - 2*margin
    h_eff = H - 2*margin
    dx = w_eff / side
    dy = h_eff / side
    
    grid_points = []
    for i in range(side):
        for j in range(side):
            x = margin + (i + 0.5) * dx
            y = margin + (j + 0.5) * dy
            grid_points.append([x, y])
    
    # Take first 21
    init_grid = np.array(grid_points[:N])
    # Estimate radius: min(dx, dy) / 2 * 0.9
    r_est = min(dx, dy) / 2.0 * 0.9
    init_grid = np.column_stack([init_grid, np.full(N, r_est)])
    guesses.append(init_grid)
    
    # 2. Random Jittered Grid
    for k in range(num_guesses - 1):
        jitter = 0.1 * min(dx, dy)
        jittered = init_grid.copy()
        jittered[:, 0] += np.random.uniform(-jitter, jitter, N)
        jittered[:, 1] += np.random.uniform(-jitter, jitter, N)
        # Clamp to bounds
        jittered[:, 0] = np.clip(jittered[:, 0], r_est, W - r_est)
        jittered[:, 1] = np.clip(jittered[:, 1], r_est, H - r_est)
        guesses.append(jittered)
        
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = 0.0
    best_circles = None
    best_W = 1.0
    best_H = 1.0
    
    # Initial aspect ratios to try: Square, Slightly Wide, Slightly Tall
    aspect_ratios = [1.0, 1.2, 1.5, 0.8, 0.7]
    
    # We will iterate through aspect ratios and restarts
    # Total budget: ~110s
    # Let's allocate time per aspect ratio
    
    num_restarts_per_ar = 4
    
    # Pre-compute W, H for each aspect ratio
    # P = 2(W+H) = 4 => W+H = 2
    # H = W / AR
    # W + W/AR = 2 => W(1 + 1/AR) = 2 => W = 2 / (1 + 1/AR) = 2*AR / (AR+1)
    
    configs = []
    for ar in aspect_ratios:
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)
        configs.append((W, H))
        
    for W, H in configs:
        if time.time() > deadline:
            break
            
        guesses = _generate_initial_guesses(W, H, num_restarts_per_ar)
        
        for guess in guesses:
            if time.time() > deadline:
                break
                
            # Setup bounds
            # W, H are fixed in this specific optimization run? 
            # Actually, it's better to optimize W, H freely but initialize them here.
            # However, optimizing W, H adds complexity. 
            # Let's fix W, H for the local optimization to ensure stability, 
            # but try multiple W, H pairs.
            # If we want to optimize W, H, we need to include them in params.
            # The _objective_and_grad function above includes W, H.
            # So we should optimize all 2 + 3N variables.
            
            params_init = np.array([W, H])
            params_init = np.concatenate([params_init, guess.flatten()])
            
            bounds = []
            # W, H bounds: must be > 0 and W+H <= 2
            # Since we handle W+H<=2 with penalty, we can set loose bounds
            # But L-BFGS-B needs explicit bounds.
            # Let's say W in [0.2, 1.8], H in [0.2, 1.8]
            # And rely on penalty for W+H<=2? 
            # Better: Constrain W+H <= 2 via bounds? No, that's a hyperplane.
            # We can constrain W in [0.1, 1.9] and H in [0.1, 1.9].
            # The penalty for W+H > 2 will push them back.
            bounds.append((0.1, 1.9)) # W
            bounds.append((0.1, 1.9)) # H
            
            for i in range(N):
                # x in [0, W], y in [0, H], r in [0, min(W,H)/2]
                # But W, H are variables. 
                # L-BFGS-B requires fixed bounds.
                # We can set bounds based on the max possible W, H (which is < 2).
                # x in [0, 2], y in [0, 2], r in [0, 1]
                bounds.append((0.0, 2.0))
                bounds.append((0.0, 2.0))
                bounds.append((1e-6, 1.0))
            
            try:
                result = minimize(
                    _objective_and_grad,
                    params_init,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
                )
                
                if time.time() > deadline:
                    break
                    
                x_opt = result.x
                W_opt = x_opt[0]
                H_opt = x_opt[1]
                circles_opt = x_opt[2:].reshape(N, 3)
                
                # Check validity
                if _check_validity(circles_opt, W_opt, H_opt):
                    score = _compute_score(circles_opt, W_opt, H_opt)
                    if score > best_score:
                        best_score = score
                        best_circles = circles_opt.copy()
                        best_W = W_opt
                        best_H = H_opt
                        
                        # Polish step if we have time
                        if time.time() < deadline - 10.0:
                            # Run a tighter optimization from this point
                            result_polish = minimize(
                                _objective_and_grad,
                                x_opt,
                                method='L-BFGS-B',
                                jac=True,
                                bounds=bounds,
                                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
                            )
                            if time.time() > deadline:
                                break
                            
                            x_polish = result_polish.x
                            W_pol = x_polish[0]
                            H_pol = x_polish[1]
                            circles_pol = x_polish[2:].reshape(N, 3)
                            
                            if _check_validity(circles_pol, W_pol, H_pol):
                                score_pol = _compute_score(circles_pol, W_pol, H_pol)
                                if score_pol > best_score:
                                    best_score = score_pol
                                    best_circles = circles_pol.copy()
                                    best_W = W_pol
                                    best_H = H_pol

            except Exception as e:
                pass

    # Fallback if no valid solution found (should not happen with grid init)
    if best_circles is None:
        # Return a simple valid configuration
        W = 1.0
        H = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    return best_circles
# EVOLVE-BLOCK-END
