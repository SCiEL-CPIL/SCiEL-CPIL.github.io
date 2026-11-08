import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _is_valid(xs, ys, rs, tol=1e-7):
    """Check if the packing is valid (no overlaps, all within bounds)."""
    # Check boundaries
    for i in range(N):
        if xs[i] < -tol or xs[i] > 1.0 + tol or ys[i] < -tol or ys[i] > 1.0 + tol:
            return False
        if rs[i] < -tol:
            return False
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    return True

def _score(xs, ys, rs):
    """Compute the score (sum of radii) if valid, else 0."""
    if _is_valid(xs, ys, rs):
        return np.sum(rs)
    return 0.0

def _make_initial_guess():
    """Create a diverse initial guess using a hexagonal-like grid."""
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    # Use a 6x5 grid layout for 26 circles (some cells empty)
    # Or a more structured layout. Let's use a 5x5 grid (25 circles) + 1 extra
    # Actually, let's just use a 6x5 grid and take the first 26
    rows = 5
    cols = 6
    for i in range(N):
        row = i // cols
        col = i % cols
        xs[i] = (col + 0.5) / cols
        ys[i] = (row + 0.5) / rows
        rs[i] = 0.48 / cols  # Slightly smaller than half-spacing
    
    return xs, ys, rs

def _smooth_objective(params, alpha=50.0):
    """
    Smooth objective: minimize -sum(r) + penalty for overlaps and boundary violations.
    We use a smooth approximation of the max(0, violation) penalty.
    """
    xs = params[0:N]
    ys = params[N:2*N]
    rs = params[2*N:3*N]
    
    # Objective: minimize -sum(rs)
    obj = -np.sum(rs)
    
    # Boundary penalties: circles must be inside [0,1]^2
    # For each circle i: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    # Penalty: smooth_max(0, r_i - x_i) + smooth_max(0, r_i - (1-x_i)) + ...
    def smooth_max(x, alpha=50.0):
        return np.logaddexp(0, alpha * x) / alpha
    
    boundary_viol = (
        smooth_max(rs - xs, alpha) +
        smooth_max(rs - (1.0 - xs), alpha) +
        smooth_max(rs - ys, alpha) +
        smooth_max(rs - (1.0 - ys), alpha)
    )
    obj += 10.0 * np.sum(boundary_viol)
    
    # Overlap penalties: for each pair (i,j), penalty if dist(i,j) < r_i + r_j
    # Smooth penalty: smooth_max(0, (r_i + r_j) - dist_ij)
    overlap_viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_dist = rs[i] + rs[j]
            overlap_viol += smooth_max(min_dist - dist, alpha)
    obj += 10.0 * overlap_viol
    
    return obj

def _smooth_objective_grad(params, alpha=50.0):
    """Gradient of the smooth objective."""
    xs = params[0:N]
    ys = params[N:2*N]
    rs = params[2*N:3*N]
    
    grad = np.zeros(3*N)
    
    def smooth_max_grad(x, alpha=50.0):
        # d/dx [logaddexp(0, alpha*x)/alpha] = sigmoid(alpha*x)
        return 1.0 / (1.0 + np.exp(-alpha * x))
    
    # Boundary gradient
    for i in range(N):
        g_x_left = smooth_max_grad(rs[i] - xs[i], alpha)
        g_x_right = -smooth_max_grad(rs[i] - (1.0 - xs[i]), alpha)
        g_y_bottom = smooth_max_grad(rs[i] - ys[i], alpha)
        g_y_top = -smooth_max_grad(rs[i] - (1.0 - ys[i]), alpha)
        
        grad[i] += 10.0 * (-g_x_left + g_x_right)  # d/dx_i
        grad[N + i] += 10.0 * (-g_y_bottom + g_y_top)  # d/dy_i
        grad[2*N + i] += 10.0 * (g_x_left + g_x_right + g_y_bottom + g_y_top)  # d/dr_i
    
    # Overlap gradient
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            dist_sq = dx*dx + dy*dy + 1e-12
            dist = np.sqrt(dist_sq)
            min_dist = rs[i] + rs[j]
            
            # d/d(dist) of smooth_max(min_dist - dist) = -smooth_max_grad(min_dist - dist)
            g = -smooth_max_grad(min_dist - dist, alpha)
            
            # Chain rule: d(dist)/d(x_i) = -dx/dist, d(dist)/d(x_j) = dx/dist
            grad[i] += 10.0 * g * (-dx / dist)
            grad[N + i] += 10.0 * g * (-dy / dist)
            grad[j] += 10.0 * g * (dx / dist)
            grad[N + j] += 10.0 * g * (dy / dist)
            grad[2*N + i] += 10.0 * smooth_max_grad(min_dist - dist, alpha)
            grad[2*N + j] += 10.0 * smooth_max_grad(min_dist - dist, alpha)
    
    # Objective gradient: d/d(r_i) of -sum(r) = -1
    grad[2*N:3*N] += -1.0
    
    return grad

def _relax_constraints(params, max_iters=100):
    """
    Project parameters back to a valid configuration by iteratively resolving overlaps and boundaries.
    This is a simple iterative projection method.
    """
    xs = params[0:N].copy()
    ys = params[N:2*N].copy()
    rs = params[2*N:3*N].copy()
    
    for _ in range(max_iters):
        changed = False
        
        # Boundary constraints
        for i in range(N):
            if xs[i] < rs[i]:
                xs[i] = rs[i]
                changed = True
            elif xs[i] > 1.0 - rs[i]:
                xs[i] = 1.0 - rs[i]
                changed = True
            if ys[i] < rs[i]:
                ys[i] = rs[i]
                changed = True
            elif ys[i] > 1.0 - rs[i]:
                ys[i] = 1.0 - rs[i]
                changed = True
        
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-8
                    if dist < 1e-8:
                        angle = np.random.rand() * 2 * np.pi
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    overlap = min_dist - dist
                    push = overlap * 0.5
                    xs[i] -= dx * push
                    ys[i] -= dy * push
                    xs[j] += dx * push
                    ys[j] += dy * push
                    changed = True
        
        if not changed:
            break
    
    return np.concatenate([xs, ys, rs])

def _optimize_from_guess(xs0, ys0, rs0, deadline):
    """Run L-BFGS-B optimization from a given initial guess."""
    params0 = np.concatenate([xs0, ys0, rs0])
    
    # Define bounds: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x_i
    for i in range(N):
        bounds.append((0.0, 1.0))  # y_i
    for i in range(N):
        bounds.append((0.0, 0.5))  # r_i
    
    best_params = params0.copy()
    best_score = _score(xs0, ys0, rs0)
    
    # Try multiple alpha values for the smooth penalty
    for alpha in [20.0, 50.0, 100.0]:
        if time.time() > deadline - 5:
            break
        
        result = minimize(
            _smooth_objective,
            params0,
            jac=_smooth_objective_grad,
            args=(alpha,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < _smooth_objective(params0, alpha):
            params = result.x
            # Relax to ensure validity
            params_relaxed = _relax_constraints(params)
            xs_r = params_relaxed[0:N]
            ys_r = params_relaxed[N:2*N]
            rs_r = params_relaxed[2*N:3*N]
            
            score = _score(xs_r, ys_r, rs_r)
            if score > best_score:
                best_score = score
                best_params = params_relaxed
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_params = None
    best_score = 0.0
    
    # Generate multiple diverse initial guesses
    np.random.seed(42)
    
    # 1. Grid-based initial guess
    xs0, ys0, rs0 = _make_initial_guess()
    params, score = _optimize_from_guess(xs0, ys0, rs0, deadline)
    if score > best_score:
        best_score = score
        best_params = params
    
    if time.time() > deadline - 10:
        return np.column_stack((best_params[0:N], best_params[N:2*N], best_params[2*N:3*N]))
    
    # 2. Random initial guesses
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        
        xs0 = np.random.rand(N) * 0.8 + 0.1
        ys0 = np.random.rand(N) * 0.8 + 0.1
        rs0 = np.random.rand(N) * 0.1 + 0.05
        
        params, score = _optimize_from_guess(xs0, ys0, rs0, deadline)
        if score > best_score:
            best_score = score
            best_params = params
    
    # 3. Hexagonal-like initial guess
    if time.time() > deadline - 5:
        return np.column_stack((best_params[0:N], best_params[N:2*N], best_params[2*N:3*N]))
    
    xs0 = np.zeros(N)
    ys0 = np.zeros(N)
    rs0 = np.zeros(N)
    spacing = 0.18
    for i in range(N):
        row = i // 6
        col = i % 6
        xs0[i] = 0.1 + col * spacing + (0.05 if row % 2 else 0.0)
        ys0[i] = 0.1 + row * spacing * 0.866
        rs0[i] = 0.08
    
    params, score = _optimize_from_guess(xs0, ys0, rs0, deadline)
    if score > best_score:
        best_score = score
        best_params = params
    
    # Final polish: run a few more optimization iterations from the best solution
    if time.time() > deadline - 3:
        return np.column_stack((best_params[0:N], best_params[N:2*N], best_params[2*N:3*N]))
    
    for alpha in [100.0, 200.0]:
        if time.time() > deadline - 1:
            break
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.0, 0.5))
        
        result = minimize(
            _smooth_objective,
            best_params,
            jac=_smooth_objective_grad,
            args=(alpha,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < _smooth_objective(best_params, alpha):
            params = result.x
            params_relaxed = _relax_constraints(params)
            xs_r = params_relaxed[0:N]
            ys_r = params_relaxed[N:2*N]
            rs_r = params_relaxed[2*N:3*N]
            
            score = _score(xs_r, ys_r, rs_r)
            if score > best_score:
                best_score = score
                best_params = params_relaxed
    
    return np.column_stack((best_params[0:N], best_params[N:2*N], best_params[2*N:3*N]))
