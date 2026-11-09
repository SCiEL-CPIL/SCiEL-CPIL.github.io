import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin from 150s limit

    def get_score(points):
        """Calculate the exact minimum radius for a given set of points."""
        if points is None:
            return 0.0
        points = np.asarray(points).reshape(-1, 3)
        if points.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        if np.any(np.abs(points) > 0.5 + 1e-9):
            return 0.0
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        wall_dists = 0.5 - np.abs(points)
        min_wall = np.min(np.min(wall_dists, axis=1))
        
        return min(min_pair, min_wall)

    def make_fcc_init():
        """Create FCC-based initialization for 100 points in unit cube."""
        # FCC lattice: points at corners and face centers of cubic cells
        # For a unit cube, use a 4x4x4 grid of FCC cells (64 corner + 96 face = 160 pts, take 100)
        # Better: use a structured arrangement inspired by FCC
        
        # FCC unit cell has 4 atoms: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
        # Scale to fit in [-0.5, 0.5]^3
        
        # Generate a 5x5x4 grid with FCC-like offsets
        # Layer approach: alternate between two offset grids
        points = []
        
        # Use 4 layers in z, each with 5x5 points
        # Odd layers: centered grid
        # Even layers: shifted by half-spacing
        nz = 4
        nx = ny = 5
        
        z_vals = np.linspace(-0.42, 0.42, nz)
        
        for k in range(nz):
            if k % 2 == 0:
                # Centered positions
                x_vals = np.linspace(-0.42, 0.42, nx)
                y_vals = np.linspace(-0.42, 0.42, ny)
            else:
                # Shifted by half spacing
                dx = 0.84 / (nx - 1) / 2
                dy = 0.84 / (ny - 1) / 2
                x_vals = np.linspace(-0.42 + dx, 0.42 - dx, nx)
                y_vals = np.linspace(-0.42 + dy, 0.42 - dy, ny)
            
            for y in y_vals:
                for x in x_vals:
                    points.append([x, y, z_vals[k]])
        
        points = np.array(points[:N])
        return points

    def make_grid_init(nx, ny, nz):
        """Create a standard grid initialization."""
        x = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        y = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        z = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        grid = np.array([(x, y, z) for z in z for y in y for x in x])
        return grid[:N]

    def force_directed_relax(points, n_steps=5000, learning_rate=0.001, momentum=0.9, 
                             wall_penalty=10.0, min_dist_target=0.22, time_limit=None):
        """
        Force-directed relaxation with Nesterov momentum.
        Forces push points apart when too close, and away from walls.
        """
        points = points.copy().astype(np.float64)
        velocity = np.zeros_like(points)
        best_points = points.copy()
        best_score = get_score(points)
        
        iu = np.triu_indices(N, k=1)
        
        for step in range(n_steps):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Nesterov lookahead
            lookahead = points + momentum * velocity
            
            # Compute forces
            forces = np.zeros_like(points)
            
            # Pairwise repulsion: push apart if closer than target
            diff = lookahead[iu[0]] - lookahead[iu[1]]
            dists = np.sqrt(np.sum(diff * diff, axis=1) + 1e-12)
            
            # Force magnitude: strong repulsion when close, weak when far
            # Use a smooth force: F = (target - d) / d * direction, capped
            overlap = np.maximum(min_dist_target - dists, 0)
            force_mag = overlap / (dists + 1e-12)
            # Cap force to prevent instability
            force_mag = np.minimum(force_mag, 5.0)
            
            # Apply forces
            for idx in range(len(iu[0])):
                i, j = iu[0][idx], iu[1][idx]
                f = force_mag[idx] * diff[idx]
                forces[i] += f
                forces[j] -= f
            
            # Wall forces: push inward if too close to walls
            for axis in range(3):
                # Left wall (x < -0.5 + margin)
                margin = 0.05
                close_left = lookahead[:, axis] < -0.5 + margin
                forces[close_left, axis] += wall_penalty * (-0.5 + margin - lookahead[close_left, axis])
                
                # Right wall (x > 0.5 - margin)
                close_right = lookahead[:, axis] > 0.5 - margin
                forces[close_right, axis] += wall_penalty * (0.5 - margin - lookahead[close_right, axis])
            
            # Update velocity with momentum
            velocity = momentum * velocity + learning_rate * forces
            points = points + velocity
            
            # Clip to bounds
            points = np.clip(points, -0.5 + 1e-10, 0.5 - 1e-10)
            
            # Track best
            if step % 100 == 0:
                s = get_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
        
        # Final score
        s = get_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        
        return best_points, best_score

    def objective(x_flat, beta):
        """Soft-min objective for L-BFGS-B polishing."""
        points = x_flat.reshape(-1, 3)
        
        iu = np.triu_indices(N, k=1)
        diff = points[iu[0]] - points[iu[1]]
        pair_dists = np.sqrt(np.sum(diff * diff, axis=1))
        
        wall_dists = 0.5 - np.abs(points)
        min_wall_per_point = np.min(wall_dists, axis=1)
        
        all_dists = np.concatenate([pair_dists, min_wall_per_point])
        
        max_d = np.max(all_dists)
        shifted = all_dists - max_d
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp < 1e-300:
            return -max_d
        soft_min = max_d - (1.0 / beta) * np.log(sum_exp)
        
        return -soft_min

    bounds = [(-0.5, 0.5)] * (N * 3)

    # Generate initial configurations
    inits = []
    
    # FCC-inspired initialization
    inits.append(make_fcc_init())
    
    # Standard grids
    inits.append(make_grid_init(5, 5, 4))
    inits.append(make_grid_init(4, 5, 5))
    inits.append(make_grid_init(5, 4, 5))
    
    # Jittered versions
    rng = np.random.default_rng(42)
    for _ in range(3):
        base = make_grid_init(5, 5, 4)
        jittered = base + rng.normal(0, 0.02, base.shape)
        jittered = np.clip(jittered, -0.45, 0.45)
        inits.append(jittered)
    
    # Evaluate initial configs
    best_score = 0.0
    best_points = inits[0].copy()
    for g in inits:
        s = get_score(g)
        if s > best_score:
            best_score = s
            best_points = g.copy()

    # Stage 1: Force-directed relaxation with Nesterov momentum on multiple inits
    for i, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        try:
            relaxed, score = force_directed_relax(
                init, 
                n_steps=8000, 
                learning_rate=0.0005, 
                momentum=0.92,
                wall_penalty=8.0,
                min_dist_target=0.22,
                time_limit=deadline - 25
            )
            
            if score > best_score:
                best_score = score
                best_points = relaxed.copy()
                
        except Exception:
            pass

    # Stage 2: Progressive soft-min L-BFGS-B polishing
    if time.time() < deadline - 15:
        current_points = best_points.copy()
        
        for beta in [100.0, 300.0, 600.0, 1200.0, 2500.0]:
            if time.time() > deadline - 10:
                break
                
            x0 = current_points.flatten()
            
            try:
                res = minimize(
                    lambda x: objective(x, beta),
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 80}
                )
                
                if time.time() > deadline - 8:
                    break
                    
                points = res.x.reshape(-1, 3)
                points = np.clip(points, -0.5 + 1e-12, 0.5 - 1e-12)
                
                s = get_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
                    current_points = points.copy()
                    
            except Exception:
                pass

    # Stage 3: Final high-precision polish
    if time.time() < deadline - 3:
        beta_final = 5000.0
        x0 = best_points.flatten()
        try:
            res = minimize(
                lambda x: objective(x, beta_final),
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            
            points = res.x.reshape(-1, 3)
            points = np.clip(points, -0.5 + 1e-12, 0.5 - 1e-12)
            
            s = get_score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            pass

    # Final safety check
    best_points = np.clip(best_points, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_points
