import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 21

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    best_score = -np.inf
    best_solution = None
    
    def smooth_max(v, eps=1e-8):
        return 0.5 * (v + np.sqrt(v**2 + eps))

    def objective(v, W, H):
        params = v.reshape(N, 3)
        xs = params[:, 0]
        ys = params[:, 1]
        rs = params[:, 2]
        
        sum_r = np.sum(rs)
        
        # Overlaps
        dx = xs[:, np.newaxis] - xs[np.newaxis, :]
        dy = ys[:, np.newaxis] - ys[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        
        req = rs[:, np.newaxis] + rs[np.newaxis, :]
        
        viol = req - dist
        iu = np.triu_indices(N, k=1)
        sm_viol = smooth_max(viol[iu])
        overlap_pen = np.sum(sm_viol**2)
        
        # Boundaries
        v_left = rs - xs
        v_right = xs + rs - W
        v_bottom = rs - ys
        v_top = ys + rs - H
        
        sm_v_left = smooth_max(v_left)
        sm_v_right = smooth_max(v_right)
        sm_v_bottom = smooth_max(v_bottom)
        sm_v_top = smooth_max(v_top)
        
        boundary_pen = np.sum(sm_v_left**2) + np.sum(sm_v_right**2) + np.sum(sm_v_bottom**2) + np.sum(sm_v_top**2)
        
        pen = overlap_pen + boundary_pen
        
        # Weight: High enough to enforce constraints, low enough to allow radius growth
        weight = 50.0
        return -sum_r + weight * pen

    def force_directed_grow(params, W, H, steps=150):
        xs = params[:, 0].copy()
        ys = params[:, 1].copy()
        rs = params[:, 2].copy()
        
        # Grow rate
        delta_r = 0.002
        dt = 0.01
        
        for step in range(steps):
            rs += delta_r
            
            # If radii are too large to fit in the box, stop growing
            # Max radius is limited by the box size
            max_r_possible = min(W, H) / 2.0
            if np.max(rs) > max_r_possible * 0.9:
                # Stop growing, just relax
                pass
            
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Overlap forces
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        dx = np.random.uniform(-0.01, 0.01)
                        dy = np.random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        f = overlap * 100.0
                        fx[i] += f * dx / dist
                        fy[i] += f * dy / dist
                        fx[j] -= f * dx / dist
                        fy[j] -= f * dy / dist
                        
            # Boundary forces
            for i in range(N):
                if xs[i] < rs[i]:
                    fx[i] += (rs[i] - xs[i]) * 100.0
                if xs[i] > W - rs[i]:
                    fx[i] -= (xs[i] - (W - rs[i])) * 100.0
                if ys[i] < rs[i]:
                    fy[i] += (rs[i] - ys[i]) * 100.0
                if ys[i] > H - rs[i]:
                    fy[i] -= (ys[i] - (H - rs[i])) * 100.0
                    
            # Update
            xs += fx * dt
            ys += fy * dt
            
            # Hard clip to prevent flying out
            xs = np.clip(xs, rs, W-rs)
            ys = np.clip(ys, rs, H-rs)
            
        params[:, 0] = xs
        params[:, 1] = ys
        params[:, 2] = rs
        return params

    def init_hex(W, H):
        area = W * H
        r_est = np.sqrt(area / (N * np.pi * 0.9))
        # Start smaller to allow growth
        r_init = r_est * 0.5
        
        dx = 2 * r_init * 1.1
        dy = 2 * r_init * 1.1 * np.sqrt(3)/2
        
        circles = []
        row = 0
        y = r_init
        while len(circles) < N and y < H - r_init:
            x = r_init
            offset = 0 if row % 2 == 0 else dx/2
            while x < W - r_init and len(circles) < N:
                circles.append([x + offset, y, r_init])
                x += dx
            y += dy
            row += 1
        
        if len(circles) < N:
             side = int(np.ceil(np.sqrt(N)))
             w_step = W / side
             h_step = H / side
             for i in range(side):
                 for j in range(side):
                     if len(circles) >= N: break
                     x = (i + 0.5) * w_step
                     y = (j + 0.5) * h_step
                     r = 0.5 * min(w_step, h_step) * 0.8
                     circles.append([x, y, r])
        return np.array(circles[:N])

    def is_valid(params, W, H):
        tol = 1e-5
        xs = params[:, 0]
        ys = params[:, 1]
        rs = params[:, 2]
        
        if np.any(xs < -tol) or np.any(xs + rs > W + tol):
            return False
        if np.any(ys < -tol) or np.any(ys + rs > H + tol):
            return False
        
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    # Main Loop
    aspects = [0.9, 1.0, 1.1]
    restarts_per_aspect = 5
    
    for W_ratio in aspects:
        if time.time() - start_time > timeout - 20:
            break
            
        W = W_ratio
        H = 2.0 - W
        
        for restart in range(restarts_per_aspect):
            if time.time() - start_time > timeout - 5:
                break
                
            params = init_hex(W, H)
            # Jitter
            params[:, 0] += np.random.uniform(-0.01, 0.01, N)
            params[:, 1] += np.random.uniform(-0.01, 0.01, N)
            
            # Force Directed Growth
            params = force_directed_grow(params, W, H, steps=100)
            
            # Optimize
            v0 = params.flatten()
            # Loose bounds for L-BFGS-B
            bounds = [(0.0, 2.0), (0.0, 2.0), (1e-6, 1.0)] * N
            
            try:
                res = minimize(objective, v0, args=(W, H), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
                
                # Check if result is better
                if res.fun < objective(v0, W, H):
                    params_res = res.x.reshape(N, 3)
                    if is_valid(params_res, W, H):
                        val = np.sum(params_res[:, 2])
                        if val > best_score:
                            best_score = val
                            best_solution = params_res
            except Exception:
                pass

    if best_solution is None:
        W, H = 1.0, 1.0
        best_solution = init_hex(W, H)
        
    return best_solution
