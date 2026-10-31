"""Circle packing in perimeter-4 rectangle: Force-directed + L-BFGS-B + Hex Init."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill

def _time_ok():
    return time.time() < DEADLINE

def _hex_init(W, H, n_circles=N):
    """Initialize circles in a hexagonal pattern scaled to fit the rectangle."""
    # Estimate average radius
    area = W * H
    # Hexagonal packing density is ~0.9069
    # Total area of circles = n * pi * r^2
    # n * pi * r^2 <= 0.9 * area
    r_est = np.sqrt(0.9 * area / (n_circles * np.pi))
    
    cx = np.zeros(n_circles)
    cy = np.zeros(n_circles)
    r = np.zeros(n_circles)
    
    # Generate hex grid points
    # Spacing based on r_est
    dx = 2.0 * r_est * 0.95 # Slight overlap allowed initially? No, keep disjoint.
    dy = dx * np.sqrt(3) / 2
    
    # Generate enough points
    nx = int(W / dx) + 5
    ny = int(H / dy) + 5
    
    pts = []
    for j in range(ny):
        for i in range(nx):
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            if j % 2 == 1:
                x += dx / 2
            if 0 <= x <= W and 0 <= y <= H:
                pts.append((x, y))
                
    # If not enough points, generate more or just use random if grid is too coarse
    if len(pts) < n_circles:
        # Fallback to random
        cx = np.random.uniform(0.1*W, 0.9*W, n_circles)
        cy = np.random.uniform(0.1*H, 0.9*H, n_circles)
        r = np.full(n_circles, r_est * 0.5)
        return np.column_stack([cx, cy, r])
        
    # Sort points by distance from center to pick central ones? Or just take first N?
    # Taking first N from grid generation is fine.
    # Shuffle to break symmetry?
    random.shuffle(pts)
    
    for i in range(n_circles):
        cx[i] = pts[i][0]
        cy[i] = pts[i][1]
        r[i] = r_est * 0.8 # Start slightly smaller to avoid immediate constraint violations
        
    return np.column_stack([cx, cy, r])

def _force_relax(x, W, H, n_iters=200, dt=0.01):
    """
    Force-directed relaxation.
    Circles repel each other and are attracted to center/max radius.
    We want to maximize sum of radii. 
    Simplified: Maximize minimum distance between centers and walls.
    """
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    
    for it in range(n_iters):
        if not _time_ok():
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                if d < 1e-6:
                    dx = random.uniform(-0.01, 0.01)
                    dy = random.uniform(-0.01, 0.01)
                    d = np.hypot(dx, dy)
                
                # Desired distance: r[i] + r[j]
                # If d < desired, push apart.
                # Use a strong repulsion if overlapping.
                desired = r[i] + r[j]
                if d < desired:
                    # Force proportional to overlap
                    f = (desired - d) / d
                    fx[i] += f * dx
                    fy[i] += f * dy
                    fx[j] -= f * dx
                    fy[j] -= f * dy
                else:
                    # Mild repulsion to spread them out
                    f = 0.1 / (d * d + 1e-6)
                    fx[i] += f * dx
                    fy[i] += f * dy
                    fx[j] -= f * dx
                    fy[j] -= f * dy
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 5.0
            # Right wall
            if W - cx[i] < r[i]:
                fx[i] -= (r[i] - (W - cx[i])) * 5.0
            # Bottom wall
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 5.0
            # Top wall
            if H - cy[i] < r[i]:
                fy[i] -= (r[i] - (H - cy[i])) * 5.0
            
            # Gravity towards center? No, we want them to push out to walls.
            # Actually, to maximize radii, they should push against each other and walls.
            # The repulsion handles the "push out".
            
        # Update positions
        cx += dt * fx
        cy += dt * fy
        
        # Clamp to rectangle
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        
        # Update radii to max possible given new positions
        for i in range(n):
            # Max radius limited by walls
            r_max = min(cx[i], W - cx[i], cy[i], H - cy[i])
            # Max radius limited by neighbors
            for j in range(n):
                if i == j: continue
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                r_lim = d - r[j]
                if r_lim < r_max:
                    r_max = r_lim
            r[i] = max(0.0, r_max)
            
    return np.column_stack([cx, cy, r])

def _vectorized_objective(x, W, H):
    """
    Objective function for minimize.
    We want to maximize sum(r).
    Minimize -sum(r) + penalty(overlaps).
    """
    n = N
    cx = x[0:n]
    cy = x[n:2*n]
    r = x[2*n:3*n]
    
    # Ensure r is positive
    r = np.maximum(r, 1e-6)
    
    # Wall constraints
    # cx >= r, cx <= W-r => r <= min(cx, W-cx)
    # cy >= r, cy <= H-r => r <= min(cy, H-cy)
    wall_limits = np.minimum(np.minimum(cx, W - cx), np.minimum(cy, H - cy))
    
    # Circle constraints
    # d_ij >= r_i + r_j
    # Penalty if d_ij < r_i + r_j
    
    penalty = 0.0
    obj = -np.sum(r)
    
    # Vectorized distance calculation
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    d = np.hypot(dx, dy)
    
    # r_sum[i,j] = r_i + r_j
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Violation: r_sum - d > 0
    violations = np.maximum(0.0, r_sum - d)
    
    # Quadratic penalty
    penalty = np.sum(violations ** 2) * 100.0
    
    # Wall violations
    wall_viol = np.maximum(0.0, r - wall_limits)
    wall_pen = np.sum(wall_viol ** 2) * 100.0
    
    return obj + penalty + wall_pen

def _construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Aspect ratios
    Ws = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for W in Ws:
        if not _time_ok():
            break
        H = 2.0 - W
        
        # Multiple restarts
        for seed in [42, 123, 456]:
            if not _time_ok():
                break
            random.seed(seed)
            np.random.seed(seed)
            
            # 1. Hex Init
            x0 = _hex_init(W, H)
            
            # 2. Force Relax
            x_rel = _force_relax(x0, W, H, n_iters=100, dt=0.05)
            
            # 3. L-BFGS-B Polish
            # Bounds: 
            # cx in [0, W], cy in [0, H], r in [0, min(W,H)/2]
            bounds = []
            for i in range(N):
                bounds.append((0.0, W))
            for i in range(N):
                bounds.append((0.0, H))
            for i in range(N):
                bounds.append((0.0, min(W, H) / 2.0))
                
            try:
                res = minimize(_vectorized_objective, x_rel.flatten(), method='L-BFGS-B', 
                               bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
                
                if res.success or (res.fun < _vectorized_objective(x_rel.flatten(), W, H)):
                    x_opt = res.x.reshape(3, N).T
                    # Validate and calculate sum
                    cx = np.clip(x_opt[:, 0], 0.0, W)
                    cy = np.clip(x_opt[:, 1], 0.0, H)
                    r = np.maximum(x_opt[:, 2], 1e-6)
                    
                    # Enforce constraints strictly for final score
                    for i in range(N):
                        r_max = min(cx[i], W - cx[i], cy[i], H - cy[i])
                        for j in range(N):
                            if i == j: continue
                            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                            r_lim = d - r[j]
                            if r_lim < r_max:
                                r_max = r_lim
                        r[i] = max(0.0, r_max)
                        
                    s = np.sum(r)
                    if s > best_sum:
                        best_sum = s
                        best_arr = np.column_stack([cx, cy, r])
                        best_W = W
                        best_H = H
                        
            except Exception as e:
                pass
                
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        x0 = _hex_init(W, H)
        best_arr = x0
        best_W, best_H = W, H
        
    return best_arr

# Keep the required function name
def construct():
    return _construct()
