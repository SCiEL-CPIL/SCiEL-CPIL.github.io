import numpy as np
from scipy.spatial.distance import pdist
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    def compute_radius(pts):
        """Compute the valid radius for a configuration."""
        dists = pdist(pts)
        min_pair_dist = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist, min_wall_dist)
    
    def md_relaxation(r_target, time_limit=10.0, rng=None):
        """
        Molecular dynamics relaxation to find a valid configuration for radius r_target.
        Uses velocity Verlet integration with damping.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        start_time = time.time()
        
        # Initialize with a jittered grid
        nx, ny, nz = 5, 5, 4
        xs = np.linspace(-0.45, 0.45, nx)
        ys = np.linspace(-0.45, 0.45, ny)
        zs = np.linspace(-0.45, 0.45, nz)
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        pts = grid + rng.uniform(-0.03, 0.03, size=(N, 3))
        pts = np.clip(pts, -0.5 + r_target, 0.5 - r_target)
        
        # Initialize velocities to zero
        vels = np.zeros_like(pts)
        
        # MD parameters
        dt = 0.005
        damping = 0.995
        force_scale = 50.0
        
        best_pts = pts.copy()
        best_radius = compute_radius(pts)
        
        # Run MD simulation
        while time.time() - start_time < time_limit:
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsive forces
            dists = pdist(pts)
            n_pairs = N * (N - 1) // 2
            # Convert condensed distance matrix to full pairwise forces
            # This is inefficient but clear; for N=100 it's 4950 pairs
            idx_i = np.repeat(np.arange(N), N-1)
            idx_j = np.tile(np.arange(N), N-1)
            # Remove self-pairs and duplicates
            mask = idx_i < idx_j
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]
            
            # Recompute distances for these pairs
            diff = pts[idx_i] - pts[idx_j]
            dists_pair = np.linalg.norm(diff, axis=1)
            
            # Repulsive force: proportional to overlap
            min_dist = 2.0 * r_target
            overlaps = np.maximum(min_dist - dists_pair, 0.0)
            
            # Force direction
            for k in range(len(idx_i)):
                if dists_pair[k] > 1e-10:
                    force_dir = diff[k] / dists_pair[k]
                    force_mag = force_scale * overlaps[k]
                    forces[idx_i[k]] += force_dir * force_mag
                    forces[idx_j[k]] -= force_dir * force_mag
            
            # Wall forces
            for axis in range(3):
                # Lower wall
                overlap_low = np.maximum((r_target - pts[:, axis]), 0.0)
                forces[:, axis] += force_scale * overlap_low
                # Upper wall
                overlap_high = np.maximum((pts[:, axis] - (0.5 - r_target)), 0.0)
                forces[:, axis] -= force_scale * overlap_high
            
            # Velocity Verlet integration
            vels = vels * damping + forces * dt
            pts = pts + vels * dt
            
            # Clip to box (hard constraint)
            pts = np.clip(pts, -0.5 + r_target, 0.5 - r_target)
            
            # Track best configuration
            current_radius = compute_radius(pts)
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
        
        return best_pts, best_radius
    
    def anneal_with_md(r_target, time_limit=10.0, rng=None):
        """
        Combine MD with periodic reheating to escape local minima.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        start_time = time.time()
        best_pts = None
        best_radius = 0.0
        
        # Run multiple MD runs with different initial conditions
        n_runs = max(1, int(time_limit // 5))
        per_run_time = time_limit / n_runs
        
        for run in range(n_runs):
            if time.time() - start_time >= time_limit:
                break
            
            # Different initialization strategies
            if run == 0:
                # Grid initialization
                nx, ny, nz = 5, 5, 4
                xs = np.linspace(-0.45, 0.45, nx)
                ys = np.linspace(-0.45, 0.45, ny)
                zs = np.linspace(-0.45, 0.45, nz)
                grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
                pts = grid + rng.uniform(-0.03, 0.03, size=(N, 3))
            elif run == 1:
                # Random initialization
                pts = rng.uniform(-0.45, 0.45, size=(N, 3))
            else:
                # Perturbed best so far
                if best_pts is not None:
                    pts = best_pts + rng.normal(0, 0.05, size=(N, 3))
                else:
                    pts = rng.uniform(-0.45, 0.45, size=(N, 3))
            
            pts = np.clip(pts, -0.5 + r_target, 0.5 - r_target)
            
            # Run MD
            result_pts, result_radius = md_relaxation(r_target, time_limit=per_run_time, rng=rng)
            
            if result_radius > best_radius:
                best_radius = result_radius
                best_pts = result_pts
        
        return best_pts, best_radius
    
    # Binary search for the best radius
    lo = 0.10
    hi = 0.115
    
    best_radius = 0.0
    best_pts = None
    
    total_time_limit = 110.0
    start_total = time.time()
    rng = np.random.default_rng(42)
    
    # Coarse binary search
    for i in range(8):
        if time.time() - start_total > total_time_limit:
            break
        
        mid = (lo + hi) / 2.0
        
        # Allocate time based on search progress
        remaining = total_time_limit - (time.time() - start_total)
        sa_time = min(remaining * 0.3, 10.0)
        
        if sa_time < 2.0:
            break
        
        pts_candidate, r_achieved = anneal_with_md(mid, time_limit=sa_time, rng=rng)
        
        if r_achieved >= mid - 1e-6:
            # Valid configuration found
            if r_achieved > best_radius:
                best_radius = r_achieved
                best_pts = pts_candidate
            lo = mid
        else:
            hi = mid
    
    # Fine-tuning around the best radius found
    if best_pts is not None and time.time() - start_total < total_time_limit - 10.0:
        # Try to improve the radius slightly
        for _ in range(3):
            if time.time() - start_total > total_time_limit - 5.0:
                break
            r_try = best_radius + 0.0005
            pts_candidate, r_achieved = anneal_with_md(r_try, time_limit=5.0, rng=rng)
            if r_achieved > best_radius:
                best_radius = r_achieved
                best_pts = pts_candidate
            else:
                break
    
    # Fallback if nothing worked
    if best_pts is None:
        ticks = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_radius = compute_radius(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
