"""Combined approach: softplus-penalized smooth optimization with force-directed refinement."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

# Deadline guard: keep safety margin before 150s
_DEADLINE_MARGIN = 10.0

def _softplus(x, beta=200.0):
    """Numerically stable softplus."""
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=200.0):
    """Gradient of softplus: sigmoid(beta*x)."""
    return 1.0 / (1.0 + np.exp(-np.clip(beta * x, -500, 500)))

def _objective_and_grad(x, W, H, beta=200.0, eq_weight=0.01):
    """
    Softplus-penalized objective with analytical gradients.
    x: shape (3N,) = [x0, y0, r0, x1, y1, r1, ...]
    Minimizes: -sum(r) + boundary penalties + pairwise overlap penalties + equalization penalty.
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    # Equalization pressure to encourage similar radii
    if eq_weight > 0:
        r_mean = np.mean(rs)
        r_var = np.mean((rs - r_mean) ** 2)
        obj += eq_weight * r_var
        grad[2::3] += eq_weight * 2.0 * (rs - r_mean) / N

    # Boundary penalties (circles must stay within [0,W] x [0,H])
    v1 = rs - xs  # left boundary
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - W  # right boundary
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys  # bottom boundary
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - H  # top boundary
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs  # positive radius
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)

        grad[3 * i] += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)

        idx = np.arange(i + 1, N)
        grad[3 * idx] += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    # Gradient of -sum(r)
    grad[2::3] += -1.0

    return obj, grad

def _check_valid(x, W, H, tol=1e-7):
    """Check if a flat array [x0,y0,r0,...] is valid."""
    for i in range(N):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        if ri < 1e-9:
            return False
        if xi - ri < -tol or xi + ri > W + tol:
            return False
        if yi - ri < -tol or yi + ri > H + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
            dist_sq = (xi-xj)**2 + (yi-yj)**2
            min_dist = ri + rj
            if dist_sq < (min_dist - tol)**2:
                return False
    return True

def _sum_radii(x):
    return float(np.sum(x[2::3]))

def _hex_init(W, H, seed=42):
    """Hexagonal lattice init over several row configurations."""
    rng = random.Random(seed)
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
        [7, 7, 7],
        [6, 7, 8],
        [8, 7, 6],
        [5, 6, 5, 5],
        [6, 5, 6, 4],
        [4, 6, 5, 6],
        [7, 6, 8],
        [8, 6, 7],
        [3, 6, 6, 6],
        [6, 6, 6, 3],
        [2, 6, 6, 7],
        [7, 6, 6, 2],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = W / (max_cols + 0.5)
        est_r_v = H / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = W / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        # Quick validity check
        valid = True
        for i in range(N):
            xi, yi, ri = circles[i]
            if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > W + 1e-6 or yi - ri < -1e-6 or yi + ri > H + 1e-6:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    dist_sq = (xi-xj)**2 + (yi-yj)**2
                    min_dist = ri + rj
                    if dist_sq < (min_dist - 1e-7)**2:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
    if best is None:
        # Fallback grid
        a = int(np.ceil(np.sqrt(N * W / H)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(W / (a + 0.5), H / (b + 0.5)) * 0.9
            x = (col + 0.5) * W / a
            y = (row + 0.5) * H / b
            circles.append([x, y, r])
        best = np.array(circles)
    return best

def _grid_init(W, H):
    """Simple grid initialization."""
    a = int(np.ceil(np.sqrt(N * W / H)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(W / (a + 0.5), H / (b + 0.5)) * 0.9
        x = (col + 0.5) * W / a
        y = (row + 0.5) * H / b
        circles.append([x, y, r])
    return np.array(circles)

def _random_init(W, H, n_random=5, scale=0.3):
    """Random initializations."""
    inits = []
    for _ in range(n_random):
        centers = np.random.uniform(0.1, 0.9, size=(N, 2)) * np.array([W, H])
        radii = np.full(N, 0.05)
        arr = np.column_stack([centers, radii])
        inits.append(arr)
    return inits

def _force_directed_refine(x, W, H, steps=150, dt=0.01, repulse_strength=1.0, attract_strength=0.02):
    """Physics-based force-directed refinement to escape local minima."""
    x = x.copy()
    centers = x[:2*N].reshape(N, 2).copy()
    radii = x[2*N:].copy()
    
    for step in range(steps):
        forces = np.zeros_like(centers)
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                if dist < 1e-8:
                    dist = 1e-8
                    diff = np.array([1e-8, 0.0])
                if dist < min_dist:
                    force = repulse_strength * (min_dist - dist) / dist
                    forces[i] += force * diff
                    forces[j] -= force * diff
        
        # Boundary forces: push circles away from walls
        for i in range(N):
            # Left wall
            if centers[i, 0] - radii[i] < 0:
                forces[i, 0] += repulse_strength * (radii[i] - centers[i, 0])
            # Right wall
            if centers[i, 0] + radii[i] > W:
                forces[i, 0] -= repulse_strength * (centers[i, 0] + radii[i] - W)
            # Bottom wall
            if centers[i, 1] - radii[i] < 0:
                forces[i, 1] += repulse_strength * (radii[i] - centers[i, 1])
            # Top wall
            if centers[i, 1] + radii[i] > H:
                forces[i, 1] -= repulse_strength * (centers[i, 1] + radii[i] - H)
        
        # Mild attraction to center to keep things compact
        cx, cy = W / 2, H / 2
        for i in range(N):
            forces[i] += -attract_strength * (centers[i] - np.array([cx, cy]))
        
        # Update positions
        centers += dt * forces
        x[:2*N] = centers.reshape(-1)
    
    return x

def _optimize_single_aspect(W, H, max_time=30.0, start_time=None):
    """Optimize for a single aspect ratio (W, H) with multiple restarts."""
    if start_time is None:
        start_time = time.time()
    
    best_x = None
    best_val = -np.inf
    
    # Generate initializations
    inits = []
    
    # Hexagonal
    hex_circles = _hex_init(W, H, seed=42)
    inits.append(hex_circles.flatten())
    
    # Grid
    grid_circles = _grid_init(W, H)
    inits.append(grid_circles.flatten())
    
    # Random
    for rand_init in _random_init(W, H, n_random=3):
        inits.append(rand_init.flatten())
    
    for init_x in inits:
        if time.time() - start_time > max_time:
            break
        
        # Force-directed refinement first
        refined = _force_directed_refine(init_x.copy(), W, H, steps=80, dt=0.008)
        
        # Check if valid after force-directed
        if _check_valid(refined, W, H):
            val = _sum_radii(refined)
            if val > best_val:
                best_val = val
                best_x = refined.copy()
        
        # Optimize with scipy
        if time.time() - start_time > max_time - 5:
            break
        
        try:
            result = minimize(
                _objective_and_grad,
                refined,
                args=(W, H, 200.0, 0.01),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-9}
            )
            
            opt_x = result.x
            
            # Check if valid
            if _check_valid(opt_x, W, H):
                val = _sum_radii(opt_x)
                if val > best_val:
                    best_val = val
                    best_x = opt_x.copy()
            
            # Second round: force-directed + optimize again
            if time.time() - start_time > max_time - 3:
                break
            
            refined2 = _force_directed_refine(opt_x.copy(), W, H, steps=50, dt=0.004)
            
            try:
                result2 = minimize(
                    _objective_and_grad,
                    refined2,
                    args=(W, H, 300.0, 0.005),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                
                opt_x2 = result2.x
                
                if _check_valid(opt_x2, W, H):
                    val = _sum_radii(opt_x2)
                    if val > best_val:
                        best_val = val
                        best_x = opt_x2.copy()
            except:
                pass
                
        except:
            pass
    
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    best_x = None
    best_val = -np.inf
    
    # Try multiple aspect ratios (W, H) with 2*(W+H) = 4, so W+H = 2
    # Aspect ratios to try
    aspect_ratios = [
        (1.0, 1.0),        # Square
        (1.2, 0.8),        # Slightly wide
        (0.8, 1.2),        # Slightly tall
        (1.4, 0.6),        # Wide
        (0.6, 1.4),        # Tall
        (1.6, 0.4),        # Very wide
        (0.4, 1.6),        # Very tall
        (1.3, 0.7),        # Medium wide
        (0.7, 1.3),        # Medium tall
        (1.1, 0.9),        # Near square
        (0.9, 1.1),        # Near square
    ]
    
    # Sort by estimated potential (closer to square tends to pack better)
    aspect_ratios.sort(key=lambda wh: abs(wh[0] - wh[1]))
    
    for W, H in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Estimate time per aspect ratio based on remaining time
        remaining = deadline - time.time()
        aspects_left = len(aspect_ratios) - aspect_ratios.index((W, H))
        time_per_aspect = max(5.0, remaining / aspects_left)
        
        x, val = _optimize_single_aspect(W, H, max_time=time_per_aspect, start_time=start_time)
        
        if x is not None and val > best_val:
            best_val = val
            best_x = x
    
    # If no valid solution found, fall back to a simple grid
    if best_x is None:
        W, H = 1.0, 1.0
        grid_circles = _grid_init(W, H)
        best_x = grid_circles.flatten()
        best_val = _sum_radii(best_x)
    
    # Final validation and return
    if not _check_valid(best_x, W, H):
        # Try to fix by scaling down
        for scale in [0.95, 0.9, 0.85, 0.8, 0.7]:
            x_scaled = best_x.copy()
            x_scaled[2::3] *= scale  # Scale radii
            x_scaled[0::3] *= scale  # Scale x positions (keep centered)
            x_scaled[1::3] *= scale  # Scale y positions
            # Re-center
            cx = (W / 2)
            cy = (H / 2)
            x_scaled[0::3] = x_scaled[0::3] * (W / max(1.0, np.max(x_scaled[0::3]) - np.min(x_scaled[0::3]))) + (W - (np.max(x_scaled[0::3]) - np.min(x_scaled[0::3]))) / 2
            x_scaled[1::3] = x_scaled[1::3] * (H / max(1.0, np.max(x_scaled[1::3]) - np.min(x_scaled[1::3]))) + (H - (np.max(x_scaled[1::3]) - np.min(x_scaled[1::3]))) / 2
            
            if _check_valid(x_scaled, W, H):
                best_x = x_scaled
                best_val = _sum_radii(best_x)
                break
    
    # Convert to (N, 3) array format
    result = best_x.reshape(N, 3)
    
    return result
