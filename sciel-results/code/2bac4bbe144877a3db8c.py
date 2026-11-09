"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    def compute_radius(c):
        """Compute the maximum valid radius for a configuration."""
        c = c.reshape(-1, 2)
        D = cdist(c, c)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D) / 2.0
        dist_border = np.min(0.5 - np.abs(c))
        return min(min_pair, dist_border)

    def neg_radius(c):
        c = c.reshape(-1, 2)
        D = cdist(c, c)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D) / 2.0
        dist_border = np.min(0.5 - np.abs(c))
        return -min(min_pair, dist_border)

    def make_honeycomb(n=50):
        """Create a hexagonal packing initial configuration."""
        # Hexagonal lattice with spacing s
        # For a unit square [-0.5, 0.5]^2, we want to fit n points
        # Hexagonal packing: rows offset by s/2 in x, s*sqrt(3)/2 in y
        # Try different spacings to get close to 50 points
        
        best_c = None
        best_r = -np.inf
        
        for s in np.linspace(0.1, 0.2, 50):
            # Generate hexagonal lattice points
            points = []
            # Row spacing in y
            dy = s * np.sqrt(3) / 2
            # Start from center
            cx, cy = 0.0, 0.0
            
            # Generate points in hexagonal pattern
            for row in range(-10, 11):
                y = cy + row * dy
                if abs(y) > 0.5:
                    continue
                # Offset in x for odd rows
                x_offset = (s / 2) if (row % 2 != 0) else 0
                for col in range(-10, 11):
                    x = cx + col * s + x_offset
                    if abs(x) > 0.5 or abs(y) > 0.5:
                        continue
                    points.append([x, y])
            
            points = np.array(points)
            if len(points) < n:
                continue
            
            # Select n points that maximize min distance
            # Greedy: start from center, add farthest point each time
            selected = [points[np.argmin(np.sum(points**2, axis=1))]]
            for _ in range(n - 1):
                # For each unselected point, compute min distance to selected
                unselected = [i for i in range(len(points)) if i not in [np.where((points == s).all(axis=1))[0][0] for s in selected]]
                # Simpler: use index tracking
                if len(selected) >= len(points):
                    break
                remaining = list(range(len(points)))
                for s in selected:
                    if isinstance(s, (int, np.integer)) and s in remaining:
                        remaining.remove(s)
                
                if not remaining:
                    break
                
                best_idx = remaining[0]
                best_min_dist = -np.inf
                for idx in remaining:
                    min_d = np.inf
                    for s in selected:
                        if isinstance(s, (int, np.integer)):
                            d = np.linalg.norm(points[idx] - points[s])
                        else:
                            d = np.linalg.norm(points[idx] - s)
                        min_d = min(min_d, d)
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_idx = idx
                selected.append(best_idx)
            
            c = np.array([points[i] if isinstance(i, (int, np.integer)) else i for i in selected])
            if len(c) != n:
                continue
            
            r = compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        
        if best_c is not None:
            return best_c
        
        # Fallback: simple grid
        return make_grid(n)

    def make_grid(n=50):
        """Create a grid-based initial configuration."""
        # Use 8x8 grid, select 50 points
        m = 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        
        # Select 50 points from 64
        # Remove corners and some edge points
        remove_indices = set()
        remove_indices.update([(0,0),(0,7),(7,0),(7,7)])  # corners
        remove_indices.update([(0,1),(0,6),(7,1),(7,6)])  # near corners
        remove_indices.update([(1,0),(6,0),(1,7),(6,7)])  # mid edges
        remove_indices.update([(3,0),(4,7)])  # two more
        
        kept = []
        for i in range(m):
            for j in range(m):
                if (i,j) not in remove_indices:
                    kept.append(grid[i*m + j])
        
        centers = np.array(kept)
        if len(centers) != n:
            centers = grid[:n]
        
        return centers

    def simulate_annealing(c0, max_time=30.0, seed=0):
        """Simulated annealing to improve the configuration."""
        rng = np.random.RandomState(seed)
        c = c0.copy()
        r = compute_radius(c)
        best_c = c.copy()
        best_r = r
        
        t0 = time.time()
        T = 0.01  # Initial temperature
        cooling = 0.9995
        
        while time.time() - t0 < max_time and time_ok():
            # Random perturbation
            delta = rng.normal(0, T * 0.5, size=(N, 2))
            new_c = c + delta
            
            # Keep points inside square
            new_c = np.clip(new_c, -0.5 + 1e-10, 0.5 - 1e-10)
            
            new_r = compute_radius(new_c)
            delta_r = new_r - r
            
            if delta_r > 0 or rng.random() < np.exp(delta_r / (T + 1e-10)):
                c = new_c
                r = new_r
                if r > best_r:
                    best_r = r
                    best_c = c.copy()
            
            T *= cooling
        
        return best_c, best_r

    def gradient_refine(c0, max_time=20.0):
        """Gradient-based refinement using L-BFGS-B."""
        c = c0.copy().flatten()
        
        def objective(x):
            return neg_radius(x)
        
        result = minimize(objective, c, method='L-BFGS-B',
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if result.success or result.fun < objective(c):
            return result.x.reshape(-1, 2)
        
        return c0

    # Phase 1: Generate multiple initial configurations
    initial_configs = []
    
    # Grid-based
    grid_c = make_grid(N)
    initial_configs.append(grid_c)
    
    # Honeycomb-based
    honey_c = make_honeycomb(N)
    initial_configs.append(honey_c)
    
    # Random with repulsion
    for seed in [0, 1, 2]:
        rng = np.random.RandomState(seed)
        c = rng.uniform(-0.45, 0.45, size=(N, 2))
        # Simple repulsion to spread out
        for _ in range(100):
            for i in range(N):
                for j in range(i+1, N):
                    d = c[i] - c[j]
                    dist = np.linalg.norm(d)
                    if dist < 0.15:
                        push = d / (dist + 1e-10) * 0.005
                        c[i] += push
                        c[j] -= push
            c = np.clip(c, -0.45, 0.45)
        initial_configs.append(c)
    
    # Select best initial configuration
    best_c = initial_configs[0]
    best_r = compute_radius(best_c)
    for c in initial_configs[1:]:
        r = compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    # Phase 2: Gradient refinement on best initial
    if time_ok():
        refined = gradient_refine(best_c, max_time=20.0)
        r_refined = compute_radius(refined)
        if r_refined > best_r:
            best_r = r_refined
            best_c = refined.copy()
    
    # Phase 3: Simulated annealing from best
    if time_ok():
        sa_c, sa_r = simulate_annealing(best_c, max_time=40.0, seed=0)
        if sa_r > best_r:
            best_r = sa_r
            best_c = sa_c.copy()
    
    # Phase 4: Gradient refinement after SA
    if time_ok():
        refined2 = gradient_refine(best_c, max_time=20.0)
        r_refined2 = compute_radius(refined2)
        if r_refined2 > best_r:
            best_r = r_refined2
            best_c = refined2.copy()
    
    # Phase 5: More SA with different seeds
    if time_ok():
        for seed in [1, 2, 3]:
            if not time_ok():
                break
            sa_c, sa_r = simulate_annealing(best_c, max_time=15.0, seed=seed)
            if sa_r > best_r:
                best_r = sa_r
                best_c = sa_c.copy()
            # Quick gradient refine
            if time_ok():
                refined3 = gradient_refine(best_c, max_time=10.0)
                r3 = compute_radius(refined3)
                if r3 > best_r:
                    best_r = r3
                    best_c = refined3.copy()
    
    # Final validation
    result = best_c.copy()
    if not np.all(np.isfinite(result)):
        result = make_grid(N)
    
    # Ensure points are strictly inside
    result = np.clip(result, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return result

# EVOLVE-BLOCK-END
