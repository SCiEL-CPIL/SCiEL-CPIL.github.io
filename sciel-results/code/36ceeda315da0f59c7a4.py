"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    def compute_radius(centers):
        """Compute the achievable radius for given centers."""
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary_dist = np.min(dist_to_boundary)
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r
    
    def _init_7x7_plus():
        """Create a 7x7 grid with one extra circle, giving r=1/14."""
        # 7x7 grid in [-0.5, 0.5]^2 with spacing 1/7
        # Centers at (-3/7, -3/7), (-2/7, -3/7), ..., (3/7, 3/7)
        # That's 49 circles. Add 1 more.
        spacing = 1.0 / 7.0
        coords = np.arange(7) * spacing - 3.0 * spacing  # -3/7 to 3/7
        grid = np.array([(x, y) for y in coords for x in coords])
        # Add one more circle - place it in a gap
        # The 7x7 grid has 49 cells, we need 50
        # Place the 50th at a position that maintains r=1/14
        # Actually, let's just use 8x8 grid and take first 50
        # 8x8 grid: spacing = 1/8, centers at -7/16, -5/16, ..., 7/16
        # That gives r = 1/16 = 0.0625, worse.
        
        # Better: use 7x7 grid (49 circles) with r=1/14, then add 1 circle
        # The 7x7 grid fills [-3/7, 3/7] x [-3/7, 3/7]
        # We can add a circle in the margin space
        # Or use a different arrangement
        
        # Let's try: 7x7 grid centered, spacing 1/7
        # This uses area [-3/7, 3/7]^2 = [−0.4286, 0.4286]^2
        # Margin to boundary: 0.5 - 3/7 = 1/14 ≈ 0.0714
        # So r = 1/14 works for the 7x7 grid
        # For the 50th circle, we need to find a spot
        
        # Alternative: use a slightly perturbed 7x7 + 1
        # Place 50th circle at (0, 0) and shift others? No, that's complex.
        
        # Simplest: start with 7x7 grid (49 pts) at spacing 1/7
        # and add 1 point at a corner-adjacent position
        # Actually, let's just use 50 points in a grid-like pattern
        
        # Best approach: 7x7 grid gives 49 points with r=1/14
        # Add 50th point - we'll place it and let relaxation fix it
        centers = grid.copy()
        # Add 50th at (0, 0) - it will overlap, but relaxation will fix
        extra = np.array([0.0, 0.0])
        centers = np.vstack([centers, extra])
        return centers
    
    def _relax(centers, r_target, n_iter=5000, dt=0.001):
        """Force-relax circles to achieve target radius r_target."""
        centers = centers.copy()
        N = len(centers)
        
        for it in range(n_iter):
            if time.time() > deadline - 10:
                break
            
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion for overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    d = centers[i] - centers[j]
                    dist = np.linalg.norm(d)
                    if dist < r_target * 2.0 and dist > 1e-12:
                        # Repel proportional to overlap
                        overlap = r_target * 2.0 - dist
                        force_mag = overlap * 0.5
                        direction = d / dist
                        forces[i] += direction * force_mag
                        forces[j] -= direction * force_mag
                    elif dist <= 1e-12:
                        # Coincident - push apart randomly
                        rand_dir = np.random.randn(2)
                        rand_dir /= np.linalg.norm(rand_dir)
                        forces[i] += rand_dir * 0.01
                        forces[j] -= rand_dir * 0.01
            
            # Wall repulsion
            for i in range(N):
                for k in range(2):
                    # Distance to left/right wall
                    if centers[i, k] < -0.5 + r_target:
                        forces[i, k] += (-0.5 + r_target - centers[i, k]) * 0.5
                    if centers[i, k] > 0.5 - r_target:
                        forces[i, k] -= (centers[i, k] - (0.5 - r_target)) * 0.5
            
            # Update positions
            centers += forces * dt
            # Clamp to stay in square
            centers = np.clip(centers, -0.5, 0.5)
        
        return centers
    
    def _grow(centers, r_start, r_end, n_steps=100):
        """Gradually grow radius from r_start to r_end, relaxing at each step."""
        r = r_start
        dr = (r_end - r_start) / n_steps
        
        for step in range(n_steps):
            if time.time() > deadline - 15:
                break
            
            r += dr
            # Relax to achieve this radius
            centers = _relax(centers, r, n_iter=2000)
        
        return centers, r
    
    # Initialize with 7x7+1 grid
    np.random.seed(42)
    centers = _init_7x7_plus()
    
    # Initial radius is about 1/14 for the 7x7 part, but the extra circle
    # will reduce it. Let's compute the actual starting radius.
    r_current = compute_radius(centers)
    
    # If the initial radius is too small due to the extra circle,
    # let's try a different initialization
    if r_current < 0.06:
        # Use a slightly perturbed 8x8 grid, take 50 points
        m = 8
        spacing = 1.0 / m
        coords = np.arange(m) * spacing - (m - 1) * spacing / 2.0
        grid = np.array([(x, y) for y in coords for x in coords])
        # Shuffle and take 50
        idx = np.random.choice(len(grid), 50, replace=False)
        centers = grid[idx].copy()
        r_current = compute_radius(centers)
    
    best_centers = centers.copy()
    best_r = r_current
    
    # Grow the radius
    r_target_start = r_current
    r_target_end = 0.072  # Slightly above Boll et al. value
    
    centers, r_final = _grow(centers, r_target_start, r_target_end, n_steps=50)
    r_final = compute_radius(centers)
    
    if r_final > best_r:
        best_r = r_final
        best_centers = centers.copy()
    
    # Try local optimization with Nelder-Mead on the best result
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        
        def negative_radius(centers_flat):
            c = centers_flat.reshape(-1, 2)
            return -compute_radius(c)
        
        x0 = best_centers.flatten()
        result = minimize(
            negative_radius,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        centers_nm = result.x.reshape(-1, 2)
        r_nm = compute_radius(centers_nm)
        
        if r_nm > best_r:
            best_r = r_nm
            best_centers = centers_nm.copy()
    
    # Final check and return
    final_r = compute_radius(best_centers)
    if final_r <= 0:
        # Fallback to a valid configuration
        m = 8
        spacing = 1.0 / m
        coords = np.arange(m) * spacing - (m - 1) * spacing / 2.0
        grid = np.array([(x, y) for y in coords for x in coords])
        best_centers = grid[:50].copy()
    
    return best_centers
# EVOLVE-BLOCK-END
