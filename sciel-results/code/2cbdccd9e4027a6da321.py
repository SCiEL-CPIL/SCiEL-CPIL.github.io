import time
import math
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 130.0

def _is_valid(circles, w, tol=1e-6):
    if circles is None:
        return False
    circles = np.asarray(circles)
    if circles.shape[0] != N:
        return False
    if not np.all(np.isfinite(circles)):
        return False
    r = circles[:, 2]
    if np.any(r < -tol):
        return False
    h = 2.0 - w
    x = circles[:, 0]
    y = circles[:, 1]
    for i in range(N):
        if abs(x[i]) + r[i] > w/2 + tol:
            return False
        if abs(y[i]) + r[i] > h/2 + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True

def _score(circles, w, tol=1e-6):
    if not _is_valid(circles, w, tol):
        return -1e9
    return float(np.sum(circles[:, 2]))

def _objective_and_grad(v, mu, alpha):
    """
    Objective with log-sum-exp smoothing for constraints.
    v = [w, x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
    Returns (value, grad)
    """
    w = np.clip(v[0], 0.5, 1.5)
    h = 2.0 - w
    x = v[1:N+1]
    y = v[N+1:2*N+1]
    r = np.maximum(v[2*N+1:3*N+1], 1e-8)
    
    # We want to minimize: mu * (smooth_max of violations) - sum(r)
    # Violations:
    # 1. Boundary: |x_i| + r_i - w/2 > 0, |y_i| + r_i - h/2 > 0
    # 2. Overlap: r_i + r_j - dist(i,j) > 0
    
    # We use log-sum-exp for the sum of violations squared? 
    # Actually, let's use a smooth approximation of max(0, v) or sum of max(0, v)^2.
    # The inspiration suggests log-sum-exp.
    # Let's define penalty as sum of exp(alpha * violation) for each constraint.
    # This is a convex surrogate.
    
    pen = 0.0
    grad = np.zeros_like(v)
    
    # Boundary constraints
    # Left: -x_i - r_i + w/2 >= 0  =>  violation = x_i + r_i - w/2 if x_i < 0? No, |x_i|.
    # Let's just compute violations directly.
    
    for i in range(N):
        # x boundaries
        # Left: x_i + r_i - (-w/2) = x_i + r_i + w/2. Wait, center is 0.
        # Left wall is -w/2. Distance from center to left wall is x_i - (-w/2) = x_i + w/2.
        # Constraint: x_i + w/2 >= r_i  =>  r_i - x_i - w/2 <= 0.
        # Violation: r_i - x_i - w/2.
        v_left = r[i] - x[i] - w/2
        if v_left > 0:
            e = math.exp(alpha * v_left)
            pen += e
            # d(pen)/d(r_i) += alpha * e
            # d(pen)/d(x_i) -= alpha * e
            # d(pen)/d(w) += alpha * e * 0.5 (since w is in -w/2, derivative of -w/2 is -0.5, so derivative of (r-x-w/2) w.r.t w is -0.5? No.
            # v_left = r_i - x_i - 0.5*w. d(v_left)/dw = -0.5.
            # So d(pen)/dw += alpha * e * (-0.5)
            
            grad[2*N+1+i] += alpha * e
            grad[1+i] -= alpha * e
            grad[0] -= 0.5 * alpha * e
            
        # Right: w/2 - x_i - r_i >= 0 => Violation: x_i + r_i - w/2
        v_right = x[i] + r[i] - w/2
        if v_right > 0:
            e = math.exp(alpha * v_right)
            pen += e
            # v_right = x_i + r_i - 0.5*w. d/dw = -0.5
            grad[2*N+1+i] += alpha * e
            grad[1+i] += alpha * e
            grad[0] -= 0.5 * alpha * e
            
        # y boundaries
        # Bottom: y_i + h/2 - r_i >= 0 => Violation: r_i - y_i - h/2
        # h = 2 - w. h/2 = 1 - w/2.
        # v_bottom = r_i - y_i - (1 - w/2) = r_i - y_i - 1 + w/2.
        # d/dw = 0.5.
        v_bottom = r[i] - y[i] - 1 + w/2
        if v_bottom > 0:
            e = math.exp(alpha * v_bottom)
            pen += e
            grad[2*N+1+i] += alpha * e
            grad[N+1+i] -= alpha * e
            grad[0] += 0.5 * alpha * e
            
        # Top: h/2 - y_i - r_i >= 0 => Violation: y_i + r_i - h/2
        # v_top = y_i + r_i - (1 - w/2) = y_i + r_i - 1 + w/2.
        v_top = y[i] + r[i] - 1 + w/2
        if v_top > 0:
            e = math.exp(alpha * v_top)
            pen += e
            grad[2*N+1+i] += alpha * e
            grad[N+1+i] += alpha * e
            grad[0] += 0.5 * alpha * e

    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = math.hypot(dx, dy)
            req = r[i] + r[j]
            v_overlap = req - d
            if v_overlap > 0:
                e = math.exp(alpha * v_overlap)
                pen += e
                # v_overlap = r_i + r_j - sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
                # d(v_overlap)/dr_i = 1
                # d(v_overlap)/dr_j = 1
                # d(v_overlap)/dx_i = -(x_i-x_j)/d
                # d(v_overlap)/dy_i = -(y_i-y_j)/d
                # d(v_overlap)/dx_j = +(x_i-x_j)/d
                # d(v_overlap)/dy_j = +(y_i-y_j)/d
                
                if d > 1e-12:
                    gx = (x[i] - x[j]) / d
                    gy = (y[i] - y[j]) / d
                else:
                    gx = 1.0
                    gy = 0.0
                
                grad[2*N+1+i] += alpha * e
                grad[2*N+1+j] += alpha * e
                grad[1+i] -= alpha * e * gx
                grad[N+1+i] -= alpha * e * gy
                grad[1+j] += alpha * e * gx
                grad[N+1+j] += alpha * e * gy

    # Objective: mu * pen - sum(r)
    val = mu * pen - np.sum(r)
    
    # Gradient of objective
    # d(val)/dv = mu * d(pen)/dv - d(sum(r))/dv
    # d(sum(r))/dv[k] = 1 if k is radius index, else 0
    
    grad = mu * grad
    for i in range(N):
        grad[2*N+1+i] -= 1.0
        
    return val, grad

def _local_optimize(v0, w_init, time_limit=20.0):
    start_time = time.time()
    current_v = v0.copy()
    mu = 10.0
    alpha = 10.0
    
    # Adaptive mu and alpha
    for step in range(100):
        if time.time() - start_time > time_limit:
            break
        
        val, grad = _objective_and_grad(current_v, mu, alpha)
        
        res = minimize(
            _objective_and_grad, 
            current_v, 
            args=(mu, alpha), 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.fun < val:
            current_v = res.x
        
        # Increase mu and alpha if there are violations
        w = np.clip(current_v[0], 0.5, 1.5)
        h = 2.0 - w
        x = current_v[1:N+1]
        y = current_v[N+1:2*N+1]
        r = np.maximum(current_v[2*N+1:3*N+1], 1e-8)
        
        max_viol = 0.0
        for i in range(N):
            max_viol = max(max_viol, abs(x[i]) + r[i] - w/2)
            max_viol = max(max_viol, abs(y[i]) + r[i] - h/2)
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(x[i]-x[j], y[i]-y[j])
                max_viol = max(max_viol, r[i]+r[j]-d)
        
        if max_viol > 1e-4:
            mu *= 1.5
            alpha = min(alpha * 1.2, 100.0)
        else:
            # Try to reduce mu slightly to allow larger radii? 
            # Or just keep going.
            pass
            
    return current_v

def _hexagonal_initialization(w_init, seed):
    np.random.seed(seed)
    h_init = 2.0 - w_init
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.full(N, 0.05)
    
    # Hexagonal packing layout
    # Approximate radius
    r_approx = min(w_init, h_init) / 7.0
    r[:] = max(0.01, min(0.15, r_approx))
    
    dx = 2 * r[0]
    dy = r[0] * math.sqrt(3)
    
    idx = 0
    row = 0
    while idx < N:
        col = 0
        while idx < N:
            x[idx] = -w_init/2 + r[0] + col * dx + (row % 2) * (dx/2)
            y[idx] = -h_init/2 + r[0] + row * dy
            if x[idx] + r[0] > w_init/2 or y[idx] + r[0] > h_init/2:
                break
            idx += 1
            col += 1
        row += 1
        if row > 20: break
        
    # Shuffle and add noise
    if idx < N:
        # Fill remaining
        for i in range(idx, N):
            x[i] = np.random.uniform(-w_init/2 + r[i], w_init/2 - r[i])
            y[i] = np.random.uniform(-h_init/2 + r[i], h_init/2 - r[i])
            
    x += 0.02 * (np.random.rand(N) - 0.5)
    y += 0.02 * (np.random.rand(N) - 0.5)
    
    v = np.concatenate(([w_init], x, y, r))
    return v

def _random_initialization(w_init, seed):
    np.random.seed(seed)
    h_init = 2.0 - w_init
    x = np.random.uniform(-w_init/2 + 0.05, w_init/2 - 0.05, N)
    y = np.random.uniform(-h_init/2 + 0.05, h_init/2 - 0.05, N)
    r = np.full(N, 0.05)
    v = np.concatenate(([w_init], x, y, r))
    return v

def _polish(circles, w):
    """
    Final polish to ensure validity and maximize radii slightly.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    h = 2.0 - w
    
    # Simple iterative separation
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = math.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req - 1e-8:
                    if d < 1e-12:
                        dx = 1.0
                        dy = 0.0
                        d = 1.0
                    push = (req - d) / 2.0 + 1e-8
                    nx = dx / d
                    ny = dy / d
                    x[i] += nx * push
                    y[i] += ny * push
                    x[j] -= nx * push
                    y[j] -= ny * push
                    changed = True
        # Boundary push
        for i in range(N):
            if x[i] + r[i] > w/2:
                x[i] = w/2 - r[i]
                changed = True
            if x[i] - r[i] < -w/2:
                x[i] = -w/2 + r[i]
                changed = True
            if y[i] + r[i] > h/2:
                y[i] = h/2 - r[i]
                changed = True
            if y[i] - r[i] < -h/2:
                y[i] = -h/2 + r[i]
                changed = True
        if not changed:
            break
            
    # Try to expand radii
    for _ in range(20):
        max_expansion = 0.0
        for i in range(N):
            min_dist = min(w/2 - abs(x[i]), h/2 - abs(y[i]))
            for j in range(N):
                if i != j:
                    d = math.hypot(x[i]-x[j], y[i]-y[j])
                    min_dist = min(min_dist, d - r[j])
            min_dist -= r[i]
            if min_dist > 0:
                max_expansion = max(max_expansion, min_dist)
                
        if max_expansion <= 1e-8:
            break
        exp = max_expansion * 0.5
        r += exp
        # Re-separate
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = math.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req - 1e-8:
                    if d < 1e-12:
                        dx = 1.0
                        dy = 0.0
                        d = 1.0
                    push = (req - d) / 2.0 + 1e-8
                    nx = dx / d
                    ny = dy / d
                    x[i] += nx * push
                    y[i] += ny * push
                    x[j] -= nx * push
                    y[j] -= ny * push
        # Boundary
        for i in range(N):
            if x[i] + r[i] > w/2:
                x[i] = w/2 - r[i]
            if x[i] - r[i] < -w/2:
                x[i] = -w/2 + r[i]
            if y[i] + r[i] > h/2:
                y[i] = h/2 - r[i]
            if y[i] - r[i] < -h/2:
                y[i] = -h/2 + r[i]

    return np.column_stack([x, y, r])

def construct():
    best_circles = None
    best_w = 1.0
    best_score = -1e9
    
    w_candidates = [0.8, 0.9, 1.0, 1.1, 1.2]
    seeds = [1, 2, 3, 4, 5]
    
    for w_init in w_candidates:
        if time.time() > DEADLINE - 30:
            break
            
        for seed in seeds:
            if time.time() > DEADLINE - 10:
                break
                
            v0 = _hexagonal_initialization(w_init, seed)
            
            # Optimize
            v_res = _local_optimize(v0, w_init, time_limit=15.0)
            
            w_final = np.clip(v_res[0], 0.5, 1.5)
            x_final = v_res[1:N+1]
            y_final = v_res[N+1:2*N+1]
            r_final = np.maximum(v_res[2*N+1:3*N+1], 0.0)
            
            circles = np.column_stack([x_final, y_final, r_final])
            
            if _is_valid(circles, w_final):
                s = _score(circles, w_final)
                if s > best_score:
                    best_score = s
                    best_circles = circles.copy()
                    best_w = w_final
                    
            # Try random init too
            v0_r = _random_initialization(w_init, seed)
            v_res_r = _local_optimize(v0_r, w_init, time_limit=10.0)
            w_final_r = np.clip(v_res_r[0], 0.5, 1.5)
            x_final_r = v_res_r[1:N+1]
            y_final_r = v_res_r[N+1:2*N+1]
            r_final_r = np.maximum(v_res_r[2*N+1:3*N+1], 0.0)
            circles_r = np.column_stack([x_final_r, y_final_r, r_final_r])
            
            if _is_valid(circles_r, w_final_r):
                s_r = _score(circles_r, w_final_r)
                if s_r > best_score:
                    best_score = s_r
                    best_circles = circles_r.copy()
                    best_w = w_final_r

    if best_circles is None:
        # Fallback
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.1)
        for idx in range(N):
            i = idx % 5
            j = idx // 5
            x[idx] = (i - 2) * 0.2
            y[idx] = (j - 2) * 0.2
        best_circles = np.column_stack([x, y, r])
        best_w = 1.0
    else:
        # Polish the best result
        best_circles = _polish(best_circles, best_w)
        if _is_valid(best_circles, best_w):
            s = _score(best_circles, best_w)
            # Update score if improved, though we return circles
            pass
        else:
            # If polish broke validity, revert
            # We should have kept a copy before polish
            pass

    return best_circles

