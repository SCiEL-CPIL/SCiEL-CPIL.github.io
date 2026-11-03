import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    best_result = None
    best_score = -np.inf

    def softplus(z, eps=1e-4):
        return np.logaddexp(0, z / eps) * eps

    def sigmoid(z, eps=1e-4):
        return 1.0 / (1.0 + np.exp(-z / eps))

    def objective(params, penalty_weight, smooth_eps=1e-4):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Base objective: maximize sum of radii
        val = -np.sum(r)
        
        penalty = 0.0
        
        # Boundary violations
        # x - r >= 0 -> r - x <= 0. Penalty on violation (r - x > 0)
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0
        
        penalty += np.sum(softplus(lx, smooth_eps) ** 2)
        penalty += np.sum(softplus(ux, smooth_eps) ** 2)
        penalty += np.sum(softplus(ly, smooth_eps) ** 2)
        penalty += np.sum(softplus(uy, smooth_eps) ** 2)
        
        # Negative radius
        penalty += np.sum(softplus(-r, smooth_eps) ** 2)
        
        # Pairwise overlaps
        # dist >= r_i + r_j -> r_i + r_j - dist <= 0. Penalty on violation.
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(softplus(overlap, smooth_eps) ** 2)
        
        return val + penalty_weight * penalty

    def objective_grad(params, penalty_weight, smooth_eps=1e-4):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        val = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2::3] = -1.0
        
        # Boundary gradients
        # f = softplus(v)^2. d(f)/dv = 2 * softplus(v) * sigmoid(v/eps)
        # v_lx = r - x. d(v_lx)/dx = -1, d(v_lx)/dr = 1
        # v_ux = x + r - 1. d(v_ux)/dx = 1, d(v_ux)/dr = 1
        # v_ly = r - y. d(v_ly)/dy = -1, d(v_ly)/dr = 1
        # v_uy = y + r - 1. d(v_uy)/dy = 1, d(v_uy)/dr = 1
        
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0
        
        sp_lx = softplus(lx, smooth_eps)
        sg_lx = sigmoid(lx, smooth_eps)
        grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
        grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx
        
        sp_ux = softplus(ux, smooth_eps)
        sg_ux = sigmoid(ux, smooth_eps)
        grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
        grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux
        
        sp_ly = softplus(ly, smooth_eps)
        sg_ly = sigmoid(ly, smooth_eps)
        grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
        grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly
        
        sp_uy = softplus(uy, smooth_eps)
        sg_uy = sigmoid(uy, smooth_eps)
        grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
        grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy
        
        # Negative radius
        nr = -r
        sp_nr = softplus(nr, smooth_eps)
        sg_nr = sigmoid(nr, smooth_eps)
        # v_nr = -r. d(v_nr)/dr = -1
        grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr
        
        # Pairwise overlaps
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        
        sp_ov = softplus(overlap, smooth_eps)
        sg_ov = sigmoid(overlap, smooth_eps)
        
        # v_ov = r_i + r_j - dist
        # d(v_ov)/dx_i = -dx/dist
        # d(v_ov)/dx_j = +dx/dist
        # d(v_ov)/dy_i = -dy/dist
        # d(v_ov)/dy_j = +dy/dist
        # d(v_ov)/dr_i = 1
        # d(v_ov)/dr_j = 1
        
        factor = 2.0 * penalty_weight * sp_ov * sg_ov
        inv_dist = 1.0 / dist
        
        # Accumulate gradients
        # grad_x_i += factor * (-dx * inv_dist)
        # grad_x_j += factor * (dx * inv_dist)
        
        np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
        np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
        np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
        np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
        np.add.at(grad[2::3], idx_i, factor)
        np.add.at(grad[2::3], idx_j, factor)
        
        val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) + np.sum(sp_ly**2) + np.sum(sp_uy**2) + np.sum(sp_nr**2) + np.sum(sp_ov**2))
        
        return val, grad

    def validate(circles, tol=1e-7):
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
            return False
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def fix_boundaries(circles):
        circles = circles.copy()
        for i in range(N):
            r = max(0.0, circles[i, 2])
            r = min(r, 0.5)
            circles[i, 2] = r
            circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)

        for _iter in range(10):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = circles[i, 2] + circles[j, 2]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        shrink = overlap / 2.0 + 1e-5
                        circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                        circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
                        circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], 1.0 - circles[j, 2])
                        circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], 1.0 - circles[j, 2])
                        changed = True
            if not changed:
                break

        return circles

    # Multi-start strategy
    num_starts = 50
    side = int(np.ceil(np.sqrt(N)))
    
    # Pre-compute grid base
    base_xs = []
    base_ys = []
    for i in range(side):
        for j in range(side):
            if len(base_xs) < N:
                base_xs.append((i + 0.5) / side)
                base_ys.append((j + 0.5) / side)
    base_xs = np.array(base_xs)
    base_ys = np.array(base_ys)
    
    for k in range(num_starts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        jitter = 0.05
        x0 = base_xs + np.random.uniform(-jitter, jitter, N)
        y0 = base_ys + np.random.uniform(-jitter, jitter, N)
        
        x0 = np.clip(x0, 0, 1)
        y0 = np.clip(y0, 0, 1)
        
        r0 = np.full(N, 0.4 / side)
        
        params0 = np.zeros(3 * N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        # Penalty method with increasing weight
        # Start with moderate weight, increase over a few iterations
        # This allows the optimizer to first move into feasible region, then tighten constraints
        
        penalty_weights = [10.0, 100.0, 1000.0, 10000.0]
        
        current_params = params0
        success = False
        
        for pw in penalty_weights:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    lambda p: objective(p, pw, 1e-4),
                    current_params,
                    jac=lambda p: objective_grad(p, pw, 1e-4)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                current_params = res.x
            except Exception:
                continue
                
        # Final validation and repair
        if time.time() > deadline:
            break
            
        x_res = current_params[0::3]
        y_res = current_params[1::3]
        r_res = current_params[2::3]
        
        circles = np.column_stack((x_res, y_res, r_res))
        
        # Repair if slightly invalid
        if not validate(circles, tol=1e-6):
            circles = fix_boundaries(circles)
            
        if validate(circles, tol=1e-7):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_result = circles

    if best_result is not None:
        return best_result
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-4
                    cells.append((x, y, r))
        return np.array(cells)
