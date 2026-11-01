"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a continuous rigid-body relaxation approach:
    1. Treat circles as particles with repulsive spring forces
    2. Use an implicit symplectic integrator for stable large time steps
    3. Iteratively increase the target radius until convergence
    4. Return the best configuration found
    """
    
    def compute_radius(centers):
        """Compute the feasible radius for given centers."""
        dist_to_boundary = 1.0 - np.linalg.norm(centers, axis=1)
        min_boundary = np.min(dist_to_boundary)
        
        if len(centers) < 2:
            min_pair = np.inf
        else:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)
    
    def make_initial_ring(n, rings=5):
        """Create initial configuration using concentric rings."""
        points = [[0.0, 0.0]]
        remaining = n - 1
        for ring in range(1, rings + 1):
            if remaining <= 0:
                break
            n_on_ring = min(remaining, 6 * ring)
            radius = ring * 0.12
            for i in range(n_on_ring):
                angle = 2 * np.pi * i / n_on_ring
                points.append([radius * np.cos(angle), radius * np.sin(angle)])
            remaining -= n_on_ring
        
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 0.9)
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
        
        return np.array(points[:n])
    
    def relax(centers, target_r, n_steps=5000, dt=0.002, k_repel=100.0, k_boundary=200.0, damping=0.95):
        """
        Relax the configuration using a symplectic integrator with repulsive forces.
        
        Args:
            centers: initial positions (n, 2)
            target_r: target radius to achieve
            n_steps: number of integration steps
            dt: time step
            k_repel: spring constant for repulsion between circles
            k_boundary: spring constant for boundary repulsion
            damping: velocity damping factor
            
        Returns:
            relaxed positions (n, 2)
        """
        pos = centers.copy().astype(np.float64)
        vel = np.zeros_like(pos)
        
        for step in range(n_steps):
            # Compute forces
            forces = np.zeros_like(pos)
            n = len(pos)
            
            # Pairwise repulsion (circles should maintain distance >= 2*target_r)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10:
                        # Jitter if coincident
                        diff = np.array([np.random.uniform(-1e-5, 1e-5), 
                                        np.random.uniform(-1e-5, 1e-5)])
                        dist = np.linalg.norm(diff)
                    
                    min_dist = 2.0 * target_r
                    if dist < min_dist:
                        # Repulsive force proportional to overlap
                        overlap = min_dist - dist
                        force_mag = k_repel * overlap
                        force_dir = diff / dist
                        forces[i] += force_mag * force_dir
                        forces[j] -= force_mag * force_dir
            
            # Boundary repulsion (centers should stay within distance 1 - target_r from origin)
            max_radius = 1.0 - target_r
            for i in range(n):
                dist_from_origin = np.linalg.norm(pos[i])
                if dist_from_origin > max_radius:
                    overlap = dist_from_origin - max_radius
                    force_mag = k_boundary * overlap
                    force_dir = -pos[i] / dist_from_origin
                    forces[i] += force_mag * force_dir
            
            # Symplectic Euler integration (position update uses new velocity)
            vel = vel * damping + forces * dt
            pos = pos + vel * dt
            
            # Hard constraint: project back inside boundary if needed
            for i in range(n):
                dist_from_origin = np.linalg.norm(pos[i])
                if dist_from_origin > max_radius:
                    pos[i] = pos[i] * (max_radius / dist_from_origin)
                    # Kill radial velocity component
                    radial_vel = np.dot(vel[i], pos[i]) / (dist_from_origin * dist_from_origin)
                    vel[i] -= radial_vel * pos[i]
        
        return pos
    
    def try_packing(target_r, n_steps=3000, dt=0.002):
        """Try to pack circles with given target radius."""
        np.random.seed(42)
        centers = make_initial_ring(N, rings=6)
        
        # Relax to achieve the target radius
        relaxed = relax(centers, target_r, n_steps=n_steps, dt=dt)
        
        # Check if we achieved the target
        achieved_r = compute_radius(relaxed)
        return relaxed, achieved_r
    
    # Binary search for the maximum achievable radius
    lo = 0.05
    hi = 0.13
    
    # First, verify we can achieve the lower bound
    _, achieved_lo = try_packing(lo, n_steps=2000)
    if achieved_lo < lo * 0.9:
        # Lower bound too high, reduce
        lo = 0.03
        _, achieved_lo = try_packing(lo, n_steps=2000)
    
    # Binary search
    best_centers = None
    best_r = 0.0
    
    for iteration in range(12):
        mid = (lo + hi) / 2
        centers, achieved_r = try_packing(mid, n_steps=4000)
        
        if achieved_r >= mid * 0.95:
            # We can achieve this radius
            lo = mid
            best_r = achieved_r
            best_centers = centers
        else:
            hi = mid
        
        if hi - lo < 1e-6:
            break
    
    # If binary search didn't find a good solution, try direct optimization from best found
    if best_centers is None:
        best_centers, best_r = try_packing(0.12, n_steps=5000)
    
    # Final polish: try to improve with more iterations
    final_r = compute_radius(best_centers)
    if final_r > best_r:
        best_r = final_r
    
    # Try a few more relaxations from the best configuration with slightly higher target
    for trial in range(3):
        trial_r = best_r * (1.0 + 0.01 * trial)
        centers_trial, achieved_trial = try_packing(trial_r, n_steps=3000)
        if achieved_trial > best_r:
            best_r = achieved_trial
            best_centers = centers_trial
    
    return best_centers
# EVOLVE-BLOCK-END
