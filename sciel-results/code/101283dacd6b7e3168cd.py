"""Simulated annealing approach for circle packing optimization."""

import numpy as np

N = 50


def construct():
    """
    Return the centers of 50 equal circles inside the unit disk.
    Uses simulated annealing to find a high-quality packing.
    """
    np.random.seed(42)
    
    def compute_radius(pts):
        """Compute the maximum radius for given centers."""
        dists_from_origin = np.sqrt(np.sum(pts**2, axis=1))
        if np.any(dists_from_origin > 1.0):
            return -1e6  # Invalid
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pairwise = np.min(dist_mat)
        
        r = min(min_pairwise / 2.0, 1.0 - np.max(dists_from_origin))
        return r
    
    def generate_initial_config():
        """Generate a good initial configuration using a combination of methods."""
        configs = []
        
        # Method 1: Hexagonal lattice
        spacing = 0.24
        sqrt3 = np.sqrt(3)
        pts = []
        for j in range(-8, 9):
            for i in range(-8, 9):
                x = i * spacing + (j % 2) * spacing / 2
                y = j * spacing * sqrt3 / 2
                if x**2 + y**2 < 0.82**2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            dists = np.sqrt(np.sum(pts**2, axis=1))
            idx = np.argsort(dists)[:N]
            configs.append(pts[idx])
        
        # Method 2: Random points with rejection
        for _ in range(3):
            pts = np.random.uniform(-0.85, 0.85, size=(N, 2))
            # Keep only points inside circle
            mask = np.sum(pts**2, axis=1) < 0.85**2
            if np.sum(mask) >= N:
                configs.append(pts[mask][:N])
            else:
                configs.append(pts)
        
        # Method 3: Ring-based configuration
        ring1_r = 0.0
        ring1_n = 1
        ring2_r = 0.25
        ring2_n = 6
        ring3_r = 0.45
        ring3_n = 12
        ring4_r = 0.62
        ring4_n = 18
        ring5_r = 0.75
        ring5_n = 13
        pts = []
        pts.append([0.0, 0.0])
        for r, n in [(ring2_r, ring2_n), (ring3_r, ring3_n), (ring4_r, ring4_n), (ring5_r, ring5_n)]:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        pts = np.array(pts[:N])
        configs.append(pts)
        
        # Choose the best initial config
        best_config = None
        best_r = -1e6
        for cfg in configs:
            r = compute_radius(cfg)
            if r > best_r:
                best_r = r
                best_config = cfg
        
        return best_config
    
    def simulated_annealing(pts, max_iters=50000, T0=0.05, T_end=1e-6, cooling=0.9995):
        """Run simulated annealing to improve the packing."""
        pts = pts.copy()
        current_r = compute_radius(pts)
        best_pts = pts.copy()
        best_r = current_r
        
        T = T0
        iters_without_improvement = 0
        
        for it in range(max_iters):
            # Select a random point to move
            i = np.random.randint(N)
            
            # Generate a random perturbation
            scale = T * 10 + 0.005
            dx = np.random.normal(0, scale)
            dy = np.random.normal(0, scale)
            
            new_pts = pts.copy()
            new_pts[i] = pts[i] + np.array([dx, dy])
            
            # Check if new point is within bounds
            if np.sum(new_pts[i]**2) > 0.99**2:
                continue
            
            new_r = compute_radius(new_pts)
            
            # Acceptance criterion
            if new_r > current_r:
                pts = new_pts
                current_r = new_r
                iters_without_improvement = 0
                
                if current_r > best_r:
                    best_r = current_r
                    best_pts = pts.copy()
            else:
                # Accept with probability exp((new_r - current_r) / T)
                if new_r > -1e5:  # Only accept if not catastrophically bad
                    prob = np.exp((new_r - current_r) / T)
                    if np.random.random() < prob:
                        pts = new_pts
                        current_r = new_r
                iters_without_improvement += 1
            
            # Cool down
            T = max(T_end, T * cooling)
            
            # Early stopping if no improvement for a long time and T is very low
            if iters_without_improvement > 5000 and T < 1e-4:
                break
        
        return best_pts, best_r
    
    # Generate initial configuration
    init_pts = generate_initial_config()
    
    # Run multiple simulated annealing runs with different random seeds
    best_pts = init_pts
    best_r = compute_radius(init_pts)
    
    for run in range(5):
        np.random.seed(42 + run * 1000)
        # Slightly perturb the initial config for diversity
        if run > 0:
            perturbed = init_pts + np.random.normal(0, 0.01, size=init_pts.shape)
            mask = np.sum(perturbed**2, axis=1) < 0.95**2
            if np.sum(mask) == N:
                init_pts = perturbed
            else:
                init_pts = init_pts
        
        result_pts, result_r = simulated_annealing(init_pts, max_iters=30000, T0=0.04, T_end=1e-6, cooling=0.9997)
        
        if result_r > best_r:
            best_r = result_r
            best_pts = result_pts
    
    return best_pts
