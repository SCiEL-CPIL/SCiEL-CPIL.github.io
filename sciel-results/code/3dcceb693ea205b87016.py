"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    def compute_radius(centers):
        """Compute the valid radius for a given set of centers."""
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        if min_dist_to_wall < 0:
            return 0.0
        n = len(centers)
        min_pair_dist = np.inf
        for i in range(n):
            diffs = centers[i+1:] - centers[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                d = np.min(dists)
                if d < min_pair_dist:
                    min_pair_dist = d
        if not np.isfinite(min_pair_dist):
            return 0.0
        r = min(min_dist_to_wall, min_pair_dist / 2.0)
        return max(0.0, r)

    def generate_hex_lattice(n_rows, n_cols_per_row):
        """Generate a hexagonal lattice inside the unit square."""
        pts = []
        if n_rows <= 0:
            return np.array([])
        d_max_x = 1.0 / n_cols_per_row
        d_max_h = 1.0 / ((n_rows - 1) * math.sqrt(3) / 2 + 1)
        d = min(d_max_x, d_max_h)
        
        x_offset = -0.5 + d / 2
        y_offset = -0.5 + d / 2
        
        for row in range(n_rows):
            y = y_offset + row * d * math.sqrt(3) / 2
            if row % 2 == 0:
                n_in_row = n_cols_per_row
                x_start = x_offset
            else:
                n_in_row = n_cols_per_row - 1
                x_start = x_offset + d / 2
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            pts = pts[:N]
        elif len(pts) < N:
            # Pad with random points near edges
            while len(pts) < N:
                x = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
                y = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
                pts = np.vstack([pts, [[x, y]]])
        
        return pts

    def simulated_annealing(centers, max_iters=5000, T0=0.01, T_end=0.0001, cooling=0.9995):
        """Simulated annealing to escape local minima."""
        centers = centers.copy()
        n = len(centers)
        T = T0
        best_centers = centers.copy()
        best_radius = compute_radius(centers)
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
                
            # Choose a random point to move
            i = np.random.randint(n)
            orig = centers[i].copy()
            
            # Random perturbation
            step_size = T * 0.5
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            centers[i] = orig + np.array([dx, dy])
            
            # Clamp to square
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            
            # Compute new radius
            new_radius = compute_radius(centers)
            
            # Accept if better or with probability exp((new - old)/T)
            if new_radius >= best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
            else:
                delta = new_radius - best_radius
                if delta < 0:
                    accept_prob = math.exp(delta / max(T, 1e-10))
                    if np.random.random() > accept_prob:
                        centers[i] = orig
                    else:
                        # Accept worse move
                        if new_radius > best_radius:
                            best_radius = new_radius
                            best_centers = centers.copy()
            
            # Cool down
            T = max(T * cooling, T_end)
            
            # Early termination if T is very small and no improvement
            if T <= T_end * 1.01:
                break
        
        return best_centers

    def soft_min_objective(x, beta=50.0):
        """Soft-min objective for gradient-based optimization."""
        c = x.reshape(-1, 2)
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        
        # Pairwise distances using vectorized computation
        min_pair = np.inf
        for i in range(N):
            diffs = c[i+1:] - c[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                d = np.min(dists)
                if d < min_pair:
                    min_pair = d
        
        if not np.isfinite(min_pair):
            return 1.0
        
        r = min(min_wall, min_pair / 2.0)
        return -r

    def l_bfgs_polish(centers, max_iter=200):
        """Polish using L-BFGS-B on the soft-min objective."""
        centers = centers.copy()
        x0 = centers.flatten()
        
        try:
            result = minimize(soft_min_objective, x0, method='L-BFGS-B',
                            bounds=[(-0.5, 0.5)] * (2 * N),
                            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            new_centers = result.x.reshape(-1, 2)
            # Clamp to square
            new_centers = np.clip(new_centers, -0.5, 0.5)
            return new_centers
        except Exception:
            return centers

    def coordinate_ascent_polish(centers, max_iter=20):
        """Final polish with coordinate ascent."""
        centers = centers.copy()
        for iteration in range(max_iter):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                orig = centers[i].copy()
                best_pos = orig
                best_r = compute_radius(centers)
                
                # Try moving in small steps in 8 directions
                for dx in [-0.0005, 0, 0.0005]:
                    for dy in [-0.0005, 0, 0.0005]:
                        if dx == 0 and dy == 0:
                            continue
                        new_center = orig + np.array([dx, dy])
                        if new_center[0] < -0.5 or new_center[0] > 0.5:
                            continue
                        if new_center[1] < -0.5 or new_center[1] > 0.5:
                            continue
                        centers[i] = new_center
                        r = compute_radius(centers)
                        if r > best_r:
                            best_r = r
                            best_pos = new_center.copy()
                
                if not np.allclose(best_pos, orig):
                    centers[i] = best_pos
                    improved = True
            
            if not improved:
                break
        
        return centers

    # Main strategy:
    # 1. Generate multiple hexagonal lattice initializations
    # 2. Apply simulated annealing to each to escape local minima
    # 3. Polish the best with L-BFGS-B
    # 4. Final coordinate ascent polish

    best_centers = None
    best_radius = 0.0

    # Try different hexagonal configurations with SA
    for n_rows in range(12, 18):
        if time.time() > deadline - 60:
            break
        n_cols = math.ceil(N / n_rows)
        
        centers = generate_hex_lattice(n_rows, n_cols)
        if len(centers) != N:
            continue
        
        # Simulated annealing
        centers_sa = simulated_annealing(centers, max_iters=3000, T0=0.008, T_end=0.00005, cooling=0.9997)
        r = compute_radius(centers_sa)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers_sa.copy()

    # Also try square grid as alternative
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    centers_sa = simulated_annealing(grid, max_iters=3000, T0=0.008, T_end=0.00005, cooling=0.9997)
    r = compute_radius(centers_sa)
    if r > best_radius:
        best_radius = r
        best_centers = centers_sa.copy()

    # Polish the best configuration with L-BFGS-B
    if best_centers is not None and time.time() < deadline - 25:
        polished = l_bfgs_polish(best_centers, max_iter=150)
        r = compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()

    # Final coordinate ascent polish
    if best_centers is not None and time.time() < deadline - 5:
        polished = coordinate_ascent_polish(best_centers, max_iter=15)
        r = compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()

    # Fallback
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
