"""Pack 50 equal circles in a unit square using Hard-Disk Expansion + SLSQP Epigraph."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _make_hexagonal_grid(n_rows, n_cols):
    """Generate a hexagonal grid of points in [-0.5, 0.5]^2."""
    points = []
    dx = 1.0 / (n_cols - 1) if n_cols > 1 else 1.0
    dy = 1.0 / (n_rows - 1) if n_rows > 1 else 1.0
    for i in range(n_rows):
        for j in range(n_cols):
            x = -0.5 + j * dx + (i % 2) * dx * 0.5
            y = -0.5 + i * dy
            # Clip to square
            x = np.clip(x, -0.5, 0.5)
            y = np.clip(y, -0.5, 0.5)
            points.append([x, y])
    return np.array(points)

def _make_grid_rect(n_rows, n_cols):
    """Generate a rectangular grid of points in [-0.5, 0.5]^2."""
    xs = np.linspace(-0.5, 0.5, n_cols)
    ys = np.linspace(-0.5, 0.5, n_rows)
    points = []
    for y in ys:
        for x in xs:
            points.append([x, y])
    return np.array(points)

def _expand_radius(centers, r_target, max_iters=500, tol=1e-10):
    """
    Hard-disk radius expansion: iteratively increase r and project centers
    to satisfy non-overlap (dist >= 2r) and wall (dist_to_wall >= r) constraints.
    Uses simple pairwise repulsion for projection.
    """
    centers = centers.copy().astype(float)
    r = 0.01  # start small
    r_step = 0.0005
    
    for it in range(max_iters):
        if r >= r_target:
            break
        
        # Try to increase r
        r_new = min(r + r_step, r_target)
        
        # Check if current configuration is valid for r_new
        valid = True
        
        # Wall check
        if np.any(0.5 - np.abs(centers) < r_new - tol):
            valid = False
        
        # Pairwise check
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 2 * r_new - tol:
            valid = False
        
        if valid:
            r = r_new
            continue
        
        # Not valid: project centers to resolve conflicts
        # Simple repulsion: for each pair too close, push apart
        for _ in range(10):  # inner projection iterations
            moved = False
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # Find conflicts
            conflict = dist < 2 * r - tol
            if not np.any(conflict):
                # Only wall conflicts
                wall_viol = 0.5 - np.abs(centers) < r - tol
                if not np.any(wall_viol):
                    break
                # Push away from walls
                for i in range(N):
                    for k in range(2):
                        if wall_viol[i, k]:
                            # Push towards center
                            centers[i, k] = np.clip(centers[i, k], -0.5 + r, 0.5 - r)
                    moved = True
                break
            
            # Resolve pairwise conflicts
            for i in range(N):
                for j in range(i+1, N):
                    if conflict[i, j]:
                        d = dist[i, j]
                        if d < 1e-12:
                            # Coincident: random nudge
                            centers[i, 0] += 0.001 * (np.random.rand() - 0.5)
                            centers[j, 0] -= 0.001 * (np.random.rand() - 0.5)
                            continue
                        # Push apart to distance 2r
                        needed = 2 * r - d
                        ux = (centers[i, 0] - centers[j, 0]) / d
                        uy = (centers[i, 1] - centers[j, 1]) / d
                        centers[i, 0] += needed * 0.5 * ux
                        centers[i, 1] += needed * 0.5 * uy
                        centers[j, 0] -= needed * 0.5 * ux
                        centers[j, 1] -= needed * 0.5 * uy
                        moved = True
            
            # Wall projection
            centers = np.clip(centers, -0.5 + r, 0.5 - r)
            
            if not moved:
                break
        
        # If we couldn't resolve, keep r and try again with smaller step
        r_step *= 0.9
        if r_step < 1e-6:
            break
    
    return centers, r

def _slsqp_epigraph(x, N_circles):
    """
    SLSQP epigraph formulation:
    x = [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y, r]
    Minimize -r
    Subject to:
      0.5 - |c_ik| >= r  for all i, k
      |c_i - c_j| >= 2r  for all i < j
      -0.5 <= c_ik <= 0.5
    """
    centers = x[:N_circles*2].reshape(N_circles, 2)
    r = x[N_circles*2]
    
    # Objective: minimize -r
    obj = -r
    
    # Inequality constraints: g(x) >= 0
    constraints = []
    
    # Wall constraints: 0.5 - |c_ik| - r >= 0
    for i in range(N_circles):
        for k in range(2):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, k=k: 0.5 - abs(x[i*2+k]) - x[N_circles*2]
            })
    
    # Pairwise constraints: |c_i - c_j|^2 - (2r)^2 >= 0
    for i in range(N_circles):
        for j in range(i+1, N_circles):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[i*2] - x[j*2])**2 + (x[i*2+1] - x[j*2+1])**2 - 4*x[N_circles*2]**2
            })
    
    # Bounds: -0.5 <= c_ik <= 0.5, r >= 0
    bounds = [(-0.5, 0.5)] * (N_circles * 2) + [(0.0, 0.5)]
    
    return obj, constraints, bounds

def _slsqp_refine(centers, time_limit=10.0):
    """Use SLSQP to refine the configuration via epigraph formulation."""
    x0 = np.zeros(N * 2 + 1)
    x0[:N*2] = centers.ravel()
    x0[N*2] = _radius(centers)  # initial r
    
    start_time = time.time()
    
    try:
        res = minimize(
            lambda x: -x[N*2],  # minimize -r
            x0,
            method='SLSQP',
            bounds=[(-0.5, 0.5)] * (N * 2) + [(0.0, 0.5)],
            constraints=[
                # Wall constraints
                {'type': 'ineq', 'fun': lambda x: 0.5 - abs(x[i*2+k]) - x[N*2]}
                for i in range(N) for k in range(2)
            ] + [
                # Pairwise constraints
                {'type': 'ineq', 'fun': lambda x: (x[i*2] - x[j*2])**2 + (x[i*2+1] - x[j*2+1])**2 - 4*x[N*2]**2}
                for i in range(N) for j in range(i+1, N)
            ],
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.x is not None:
            refined_centers = res.x[:N*2].reshape(N, 2)
            # Clip to ensure validity
            refined_centers = np.clip(refined_centers, -0.5, 0.5)
            return refined_centers
    except Exception:
        pass
    
    return centers

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    best_centers = None
    best_r = 0.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Hexagonal grids with varying densities
    for rows in [7, 8, 9]:
        for cols in [7, 8, 9]:
            if rows * cols >= N:
                grid = _make_hexagonal_grid(rows, cols)
                # Randomly select N points
                idx = np.random.choice(len(grid), N, replace=False)
                initial_configs.append(grid[idx])
    
    # 2. Rectangular grids
    for rows in [7, 8]:
        for cols in [7, 8]:
            if rows * cols >= N:
                grid = _make_grid_rect(rows, cols)
                idx = np.random.choice(len(grid), N, replace=False)
                initial_configs.append(grid[idx])
    
    # 3. Random uniform
    for _ in range(5):
        initial_configs.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # 4. Slightly perturbed hexagonal
    base_hex = _make_hexagonal_grid(8, 8)
    if len(base_hex) >= N:
        idx = np.random.choice(len(base_hex), N, replace=False)
        perturbed = base_hex[idx] + np.random.normal(0, 0.01, size=(N, 2))
        perturbed = np.clip(perturbed, -0.49, 0.49)
        initial_configs.append(perturbed)
    
    # For each initial config, do hard-disk expansion then SLSQP polish
    for config in initial_configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Hard-disk expansion
        time_left_expand = min(8.0, DEADLINE - 15 - (time.time() - start_time))
        if time_left_expand < 1.0:
            continue
        
        expanded, r_exp = _expand_radius(config, r_target=0.072, max_iters=300)
        r_exp_actual = _radius(expanded)
        
        # SLSQP epigraph polish
        time_left_slsqp = min(10.0, DEADLINE - 5 - (time.time() - start_time))
        if time_left_slsqp > 1.0:
            polished = _slsqp_refine(expanded, time_limit=time_left_slsqp)
            r_pol = _radius(polished)
            
            if r_pol > best_r:
                best_r = r_pol
                best_centers = polished.copy()
        
        if r_exp_actual > best_r:
            best_r = r_exp_actual
            best_centers = expanded.copy()
        
        # If we've beaten the known best, we can stop early
        if best_r > 0.0715:
            break
    
    # Final polish on best if time remains
    if best_centers is not None and time.time() - start_time < DEADLINE - 3:
        time_left = DEADLINE - 3 - (time.time() - start_time)
        if time_left > 2.0:
            final_polished = _slsqp_refine(best_centers, time_limit=time_left)
            r_final = _radius(final_polished)
            if r_final > best_r:
                best_r = r_final
                best_centers = final_polished.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
        ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
        best_centers = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    
    return best_centers
