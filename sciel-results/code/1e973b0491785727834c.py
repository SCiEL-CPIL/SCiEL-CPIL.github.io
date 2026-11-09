import numpy as np
import time
from scipy.optimize import minimize

N = 200

def compute_radius(points):
    # points: (N, 2)
    # Calculate min pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    
    # Calculate min wall distance
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return r

def _energy_and_grad(flat, power=4.0):
    """
    Compute repulsive inverse-power energy and its gradient.
    Energy = sum_{i<j} 1/d_ij^power + sum_i (1/wall_dist_i)^power
    Minimizing this maximizes the minimum distance (softly).
    """
    pts = flat.reshape(N, 2)
    energy = 0.0
    grad = np.zeros_like(flat)
    
    # Pairwise repulsion
    for i in range(N):
        for j in range(i + 1, N):
            d = pts[i] - pts[j]
            dist_sq = d[0]*d[0] + d[1]*d[1]
            if dist_sq < 1e-12:
                dist_sq = 1e-12
            dist = np.sqrt(dist_sq)
            # 1/dist^power = dist^(-power)
            # d(1/dist^power)/d(pts[i]) = -power * dist^(-power-2) * d
            factor = power * dist_sq ** (-(power/2.0 + 1.0))
            energy += dist_sq ** (-power/2.0)
            grad[2*i] += factor * d[0]
            grad[2*i+1] += factor * d[1]
            grad[2*j] -= factor * d[0]
            grad[2*j+1] -= factor * d[1]
    
    # Wall repulsion
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Left wall: x + 0.5
    d = x + 0.5
    d[d < 1e-6] = 1e-6
    energy += np.sum(d ** (-power))
    grad[0::2] += power * d ** (-power - 1)
    
    # Right wall: 0.5 - x
    d = 0.5 - x
    d[d < 1e-6] = 1e-6
    energy += np.sum(d ** (-power))
    grad[0::2] -= power * d ** (-power - 1)
    
    # Bottom wall: y + 0.5
    d = y + 0.5
    d[d < 1e-6] = 1e-6
    energy += np.sum(d ** (-power))
    grad[1::2] += power * d ** (-power - 1)
    
    # Top wall: 0.5 - y
    d = 0.5 - y
    d[d < 1e-6] = 1e-6
    energy += np.sum(d ** (-power))
    grad[1::2] -= power * d ** (-power - 1)
    
    return energy, grad

def _hex_init(rows):
    """Generate hexagonal grid initialization."""
    cols = int(np.ceil(N / rows))
    xs = []
    ys = []
    for r in range(rows):
        x_offset = 0.5 if r % 2 == 1 else 0.0
        for c in range(cols):
            x = c + x_offset
            y = r * np.sqrt(3) / 2.0
            xs.append(x)
            ys.append(y)
    
    pts = np.column_stack((xs, ys))
    
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        while len(pts) < N:
            pts = np.vstack([pts, np.array([np.random.rand()-0.5, np.random.rand()-0.5])])
            
    # Center and scale
    center = np.mean(pts, axis=0)
    pts -= center
    max_coord = np.max(np.abs(pts))
    scale = 0.45 / max_coord if max_coord > 0 else 1.0
    pts *= scale
    
    return pts

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    best_r = 0.0
    best_points = None
    
    # Generate initializations
    inits = []
    
    # Hexagonal inits with different aspect ratios
    for rows in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
        pts = _hex_init(rows)
        inits.append(pts)
    
    # Random inits - these help escape lattice local minima
    np.random.seed(42)
    for _ in range(15):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        flat = init.flatten()
        
        # Multi-stage optimization with increasing power
        # Start with low power (smooth landscape), increase for precision
        powers = [2, 4, 8, 16]
        
        for power in powers:
            if time.time() > deadline - 5:
                break
                
            def obj(flat_x):
                e, _ = _energy_and_grad(flat_x, power)
                return e
            
            def obj_grad(flat_x):
                _, g = _energy_and_grad(flat_x, power)
                return g
            
            try:
                res = minimize(
                    obj, flat, jac=obj_grad, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
                )
                flat = res.x
            except Exception:
                pass
            
            # Check radius after each power stage
            pts = flat.reshape(N, 2)
            r = compute_radius(pts)
            if r > best_r:
                best_r = r
                best_points = pts.copy()
    
    # Final polish with very high power if time remains
    if best_points is not None and time.time() < deadline - 10:
        flat = best_points.flatten()
        
        for power in [32, 64]:
            if time.time() > deadline - 5:
                break
                
            def obj(flat_x):
                e, _ = _energy_and_grad(flat_x, power)
                return e
            
            def obj_grad(flat_x):
                _, g = _energy_and_grad(flat_x, power)
                return g
            
            try:
                res = minimize(
                    obj, flat, jac=obj_grad, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
                )
                flat = res.x
            except Exception:
                pass
            
            pts = flat.reshape(N, 2)
            r = compute_radius(pts)
            if r > best_r:
                best_r = r
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points
