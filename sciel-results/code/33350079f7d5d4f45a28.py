"""Improved circle packing: soft-penalty L-BFGS-B with lambda annealing + feasibility polish."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _make_hex_seed(scale=1.0):
    """Hexagonal packing seed scaled to fit in unit square."""
    # Hex grid: rows offset
    # Try to pack as many as possible in a hex arrangement
    # Use a grid of hex positions
    rows = 6
    cols = 5
    # Hex spacing
    dx = 1.0 / (cols + 0.5)
    dy = dx * np.sqrt(3) / 2
    # Scale to fit
    total_w = (cols - 1) * dx + dx  # approx
    total_h = (rows - 1) * dy + dy
    # Just generate positions
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * dx + (0.5 * dx if i % 2 == 1 else 0)
            y = (i + 0.5) * dy
            pts.append((x, y))
    pts = np.array(pts)
    # Normalize to [0,1]
    pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0) + 1e-12)
    pts = pts * 0.9 + 0.05
    # Take first N
    if len(pts) >= N:
        pts = pts[:N]
    else:
        # Pad with random
        extra = np.random.rand(N - len(pts), 2) * 0.8 + 0.1
        pts = np.vstack([pts, extra])
    # Initial radii: small
    r0 = np.full(N, 0.03)
    return pts, r0

def _make_random_seed(rng):
    pts = rng.uniform(0.1, 0.9, size=(N, 2))
    r0 = np.full(N, 0.03)
    return pts, r0

def _objective(params, lam):
    """
    params: (x, y, r) flattened, length 3*N
    lam: penalty weight for overlap
    Returns: -sum(r) + lam * overlap_penalty
    """
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Objective: maximize sum of radii -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Overlap penalty: for each pair, penalize if dist < r_i + r_j
    # Use soft penalty: max(0, r_i + r_j - dist)^2
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap * overlap
    obj += lam * penalty
    
    # Boundary penalty: circles must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    bpen = 0.0
    for i in range(N):
        if x[i] - r[i] < 0:
            bpen += (x[i] - r[i])**2
        if x[i] + r[i] > 1:
            bpen += (x[i] + r[i] - 1)**2
        if y[i] - r[i] < 0:
            bpen += (y[i] - r[i])**2
        if y[i] + r[i] > 1:
            bpen += (y[i] + r[i] - 1)**2
    obj += lam * bpen
    
    return obj

def _gradient(params, lam):
    """
    Gradient of the objective.
    """
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    # Gradient of -sum(r)
    gr -= 1.0
    
    # Overlap penalty gradient
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                # d/dx_i of overlap^2 = 2*overlap * (dx_i/dx_i) where dx = x_i - x_j
                # d(overlap)/dx_i = -dx/dist (since overlap = r_i + r_j - dist, d(dist)/dx_i = dx/dist)
                # Actually: overlap = r_i + r_j - dist
                # d(overlap)/dx_i = -dx/dist
                # d(overlap^2)/dx_i = 2*overlap * (-dx/dist)
                grad_factor = 2.0 * overlap / dist
                gx[i] += lam * grad_factor * dx
                gx[j] -= lam * grad_factor * dx
                gy[i] += lam * grad_factor * dy
                gy[j] -= lam * grad_factor * dy
                gr[i] += lam * 2.0 * overlap
                gr[j] += lam * 2.0 * overlap
    
    # Boundary penalty gradient
    for i in range(N):
        if x[i] - r[i] < 0:
            gx[i] += lam * 2.0 * (x[i] - r[i])
            gr[i] += lam * (-2.0 * (x[i] - r[i]))
        if x[i] + r[i] > 1:
            gx[i] += lam * 2.0 * (x[i] + r[i] - 1)
            gr[i] += lam * 2.0 * (x[i] + r[i] - 1)
        if y[i] - r[i] < 0:
            gy[i] += lam * 2.0 * (y[i] - r[i])
            gr[i] += lam * (-2.0 * (y[i] - r[i]))
        if y[i] + r[i] > 1:
            gy[i] += lam * 2.0 * (y[i] + r[i] - 1)
            gr[i] += lam * 2.0 * (y[i] + r[i] - 1)
    
    return np.concatenate([gx, gy, gr])

def _project_feasible(x, y, r):
    """Project to satisfy boundary constraints by reducing radii if needed."""
    for i in range(N):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        if r[i] > max_r:
            r[i] = max(0, max_r)
    # Project overlaps iteratively
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    dist = 1e-12
                required = r[i] + r[j] - dist
                if required > 0:
                    # Reduce both radii proportionally
                    total = r[i] + r[j]
                    if total > 0:
                        reduce_i = required * r[i] / total
                        reduce_j = required * r[j] / total
                        r[i] = max(0, r[i] - reduce_i)
                        r[j] = max(0, r[j] - reduce_j)
                    changed = True
        if not changed:
            break
    return x, y, r

def _is_valid(x, y, r, tol=1e-6):
    """Check if the configuration is valid (disjoint, inside square)."""
    for i in range(N):
        if x[i] - r[i] < -tol or x[i] + r[i] > 1 + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > 1 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, safety margin from 150s
    
    best_params = None
    best_value = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate diverse seeds
    seeds = []
    
    # Hex seed
    pts_hex, r_hex = _make_hex_seed()
    seeds.append(np.concatenate([pts_hex, r_hex]))
    
    # Random seeds
    for _ in range(10):
        pts_r, r_r = _make_random_seed(rng)
        seeds.append(np.concatenate([pts_r, r_r]))
    
    # Grid seed (similar to original but slightly better)
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    grid_params = np.array(cells[:N]).flatten()
    seeds.append(grid_params)
    
    # Lambda annealing schedule
    lam_start = 100.0
    lam_end = 1.0
    n_anneal_steps = 20
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() - start_time > deadline - 30:
            break
        
        params = seed.copy()
        
        # Anneal lambda
        for step in range(n_anneal_steps):
            if time.time() - start_time > deadline - 30:
                break
            
            frac = step / max(1, n_anneal_steps - 1)
            lam = lam_start * (lam_end / lam_start) ** frac
            
            # Run L-BFGS-B for a few iterations
            result = minimize(
                _objective,
                params,
                jac=_gradient,
                args=(lam,),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            params = result.x
            
            # Check feasibility and update best
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            
            # Project to feasible
            x_p, y_p, r_p = _project_feasible(x.copy(), y.copy(), r.copy())
            val = np.sum(r_p)
            
            if val > best_value:
                best_value = val
                best_params = np.concatenate([x_p, y_p, r_p])
        
        # Final polish: run with low lambda for more iterations
        if time.time() - start_time < deadline - 30:
            result = minimize(
                _objective,
                params,
                jac=_gradient,
                args=(lam_end,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            params = result.x
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            x_p, y_p, r_p = _project_feasible(x.copy(), y.copy(), r.copy())
            val = np.sum(r_p)
            if val > best_value:
                best_value = val
                best_params = np.concatenate([x_p, y_p, r_p])
    
    # If no valid result found, fall back to grid
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x = best_params[0:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Final feasibility check and projection
    x, y, r = _project_feasible(x.copy(), y.copy(), r.copy())
    
    # Ensure strict validity with tiny margin
    eps = 1e-7
    for i in range(N):
        r[i] = min(r[i], x[i] - eps, 1 - x[i] - eps, y[i] - eps, 1 - y[i] - eps)
        r[i] = max(r[i], 0)
    
    # Re-check overlaps and reduce radii if needed
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    dist = 1e-12
                required = r[i] + r[j] - dist + eps
                if required > 0:
                    total = r[i] + r[j]
                    if total > 0:
                        reduce_i = required * r[i] / total
                        reduce_j = required * r[j] / total
                        r[i] = max(0, r[i] - reduce_i)
                        r[j] = max(0, r[j] - reduce_j)
                    changed = True
        if not changed:
            break
    
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
