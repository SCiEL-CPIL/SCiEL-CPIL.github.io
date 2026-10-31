# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist[mask] < (min_dist[mask] - tol)):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(100):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _init_hexagonal(seed=0):
    """Hexagonal packing initialization, slightly jittered."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Create a dense hexagonal grid
    # Radius approx 0.09 for 26 circles in unit square?
    # 5x5 grid is 25. Add 1.
    # Let's use a 6x6 grid of small circles and pick the best 26?
    # Or just place them in a hex pattern.
    
    # Simple approach: Place in a grid, then optimize.
    # Better: Hexagonal lattice points.
    points = []
    r_base = 0.09
    dx = 2 * r_base
    dy = np.sqrt(3) * r_base
    
    for row in range(7):
        for col in range(7):
            x = col * dx + (row % 2) * (dx / 2)
            y = row * dy
            # Center the grid
            x += 0.5 - (6 * dx) / 2
            y += 0.5 - (6 * dy) / 2
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
                
    # If we have enough points, use them. If not, add random.
    # We need 26 points.
    if len(points) < N:
        # Fill with random
        while len(points) < N:
            points.append(rng.uniform(0.1, 0.9, 2))
            
    # Shuffle and take N
    idx = rng.choice(len(points), N, replace=False)
    for i in range(N):
        pos[i] = points[idx[i]]
        radii[i] = r_base * 0.9 # Start slightly smaller to allow movement
        
    # Add jitter
    pos += rng.uniform(-0.01, 0.01, (N, 2))
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _solve_trust_constr(pos, radii, deadline):
    """Solve using Trust-Constr which is robust for non-convex constraints."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
        
    def cons_fun(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        # Boundary constraints: p - r >= 0, 1 - p - r >= 0
        c_bound = np.concatenate([
            p[:, 0] - r,
            1.0 - p[:, 0] - r,
            p[:, 1] - r,
            1.0 - p[:, 1] - r
        ])
        
        # Overlap constraints: dist - (r_i + r_j) >= 0
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12) # Avoid 0/0 in grad?
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        c_overlap = (dist - min_dist)[mask]
        
        return np.concatenate([c_bound, c_overlap])
        
    # Gradient of objective
    def obj_grad(x):
        grad = np.zeros_like(x)
        grad[2*N:] = -1.0
        return grad
        
    # Gradient of constraints
    def cons_grad(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        
        # Boundary grads
        # p[:,0] - r: grad_p[:,0] = 1, grad_r = -1
        # 1 - p[:,0] - r: grad_p[:,0] = -1, grad_r = -1
        # ...
        
        # This is complex to vectorize manually for Trust-Constr if not using Jacobian.
        # Trust-Constr can use finite differences if not provided, but it's slow.
        # Let's provide the Jacobian.
        
        n_cons = 4 * N + N * (N - 1) // 2
        jac = np.zeros((n_cons, 2 * N + N))
        
        # Boundaries
        # Indices 0..N-1: x_i - r_i
        jac[0:N, 0:2:N] = 1.0
        jac[0:N, 2*N:3*N] = -1.0
        
        # Indices N..2N-1: 1 - x_i - r_i
        jac[N:2*N, 0:2:N] = -1.0
        jac[N:2*N, 2*N:3*N] = -1.0
        
        # Indices 2N..3N-1: y_i - r_i
        jac[2*N:3*N, 1:2:N] = 1.0
        jac[2*N:3*N, 2*N:3*N] = -1.0
        
        # Indices 3N..4N-1: 1 - y_i - r_i
        jac[3*N:4*N, 1:2:N] = -1.0
        jac[3*N:4*N, 2*N:3*N] = -1.0
        
        # Overlaps
        # c_k = dist_ij - (r_i + r_j)
        # d(c_k)/dx_i = (x_i - x_j) / dist
        # d(c_k)/dy_i = (y_i - y_j) / dist
        # d(c_k)/dr_i = -1
        # d(c_k)/dr_j = -1
        
        idx = 4 * N
        for i in range(N):
            for j in range(i + 1, N):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                
                if dist > 1e-9:
                    jac[idx, 2*i] = dx / dist
                    jac[idx, 2*i+1] = dy / dist
                    jac[idx, 2*j] = -dx / dist
                    jac[idx, 2*j+1] = -dy / dist
                    jac[idx, 2*N + i] = -1.0
                    jac[idx, 2*N + j] = -1.0
                idx += 1
                
        return jac

    try:
        res = minimize(obj, x0, method='trust-constr', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun, 'jac': cons_grad}],
                       options={'maxiter': 100, 'gtol': 1e-5, 'xtol': 1e-5})
        if res.success:
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_pos = None
    best_radii = None
    best_score = -1

    # Multi-start with Hexagonal + Random
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        if seed == 0:
            pos, radii = _init_hexagonal(0)
        else:
            # Random init with slight structure
            rng = np.random.RandomState(seed)
            pos = rng.uniform(0.1, 0.9, (N, 2))
            radii = np.full(N, 0.08)
            pos, radii = _post_process(pos, radii)
            
        res = _solve_trust_constr(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    if best_pos is None:
        # Fallback
        pos, radii = _init_hexagonal(0)
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
