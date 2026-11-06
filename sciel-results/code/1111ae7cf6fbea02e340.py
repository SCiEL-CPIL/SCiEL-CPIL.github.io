import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
    def generate_hex_lattice(r):
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        k = int(1.0 / d) + 2
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    # Find a good initial radius
    r_low = 0.01
    r_high = 0.08
    
    # Ensure r_low gives enough points
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    # Binary search for max r in hex lattice
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    points = generate_hex_lattice(r_low)
    
    if len(points) < N:
        # Fallback random
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points * 0.5 # Scale down

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    # Add small random jitter to break symmetry and avoid degenerate states
    points += np.random.normal(0, 0.001, (N, 2))
    
    best_points = points.copy()
    best_r = 0.0
    
    def compute_radius(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Efficient pairwise distance using chunked computation to save memory
        n = len(pts)
        min_pair_sq = np.inf
        chunk_size = 100
        for i in range(0, n, chunk_size):
            j = min(i + chunk_size, n)
            diff = pts[i:j, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            # Zero out diagonal within chunk
            row_idx = np.arange(j - i)
            col_idx = np.arange(i, j)
            dists_sq[row_idx, col_idx] = np.inf
            chunk_min = np.min(dists_sq)
            if chunk_min < min_pair_sq:
                min_pair_sq = chunk_min
        
        min_pair = np.sqrt(min_pair_sq) / 2.0
        return min(min_pair, min_boundary)

    best_r = compute_radius(best_points)
    
    # Optimization Loop: Vectorized Gradient Descent on Potential Energy
    # Use smoother potential: E = sum_{i<j} 1/d_ij + lambda * sum_i 1/(1-|c_i|)
    # This avoids the extreme forces from 1/d^4
    
    def compute_forces(pts, lambda_bound):
        """Compute forces using KD-tree for efficient neighbor search"""
        n = len(pts)
        forces = np.zeros_like(pts)
        
        # Use KD-tree to find neighbors within a reasonable distance
        # For force computation, we need all pairs, but we can use a cutoff
        # The force decays as 1/d^3, so distant pairs contribute little
        # Use a cutoff of about 5*typical_spacing for efficiency
        tree = cKDTree(pts)
        
        # Get all pairs within a cutoff distance
        # For N=300, we can just use query_pairs which is O(N log N)
        pairs = tree.query_pairs(0.3)  # Cutoff distance - should be > 2*r_typical
        
        for i, j in pairs:
            diff = pts[i] - pts[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-12:
                continue
            dist = np.sqrt(dist_sq)
            # 1/d^3 weighting
            inv_d3 = 1.0 / (dist_sq * dist)
            force = inv_d3 * diff
            forces[i] += force
            forces[j] -= force
        
        # Boundary force: push points away from boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        radii_safe = np.where(radii > 1e-8, radii, 1.0)
        # Use a smoother boundary potential: 1/(1-r)^2
        margin = np.maximum(1.0 - radii_safe, 1e-6)
        bound_strength = lambda_bound / (margin * margin)
        bound_forces = bound_strength[:, np.newaxis] * pts / radii_safe[:, np.newaxis]
        
        return forces + bound_forces
    
    # Multi-stage optimization with different force potentials
    stages = [
        {'steps': 30000, 'step_size': 0.003, 'lambda': 2.0, 'eval_every': 25, 'max_no_improve': 100},
        {'steps': 20000, 'step_size': 0.0015, 'lambda': 4.0, 'eval_every': 20, 'max_no_improve': 70},
        {'steps': 15000, 'step_size': 0.0008, 'lambda': 6.0, 'eval_every': 15, 'max_no_improve': 50},
        {'steps': 10000, 'step_size': 0.0004, 'lambda': 8.0, 'eval_every': 10, 'max_no_improve': 30},
    ]
    
    for stage_idx, stage in enumerate(stages):
        if time.time() > deadline - 15.0:
            break
            
        step_size = stage['step_size']
        lambda_bound = stage['lambda']
        eval_every = stage['eval_every']
        max_no_improve = stage['max_no_improve']
        no_improve_count = 0
        initial_step_size = stage['step_size']
        
        for it in range(stage['steps']):
            if time.time() > deadline - 5.0:
                break
                
            # Compute current radius periodically
            if it % eval_every == 0:
                current_r = compute_radius(points)
                if current_r > best_r:
                    best_r = current_r
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Perturb to escape local minimum
                        perturb_scale = 0.008 * (1 + stage_idx * 0.3)
                        points += np.random.normal(0, perturb_scale, (N, 2))
                        # Ensure validity
                        radii = np.hypot(points[:, 0], points[:, 1])
                        outside = radii > 1.0
                        if np.any(outside):
                            points[outside] = points[outside] / radii[outside, np.newaxis]
                        
                        no_improve_count = 0
                        # Reset step size to initial for this stage
                        step_size = initial_step_size

            # Compute Forces
            total_forces = compute_forces(points, lambda_bound)
            
            # Adaptive step size based on gradient magnitude
            grad_mag = np.linalg.norm(total_forces, axis=1)
            grad_mag_safe = np.where(grad_mag > 1e-6, grad_mag, 1.0)
            adaptive_factor = 1.0 / (1.0 + 0.1 * grad_mag_safe)
            
            # Update with per-point adaptive step
            points += total_forces * (step_size * adaptive_factor)[:, np.newaxis]
            
            # Enforce hard constraint: inside unit circle
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]
            
            # Decay step size
            step_size *= 0.99999
        
        # Check if we should stop early
        if best_r > 0.0534:
            break

    # Final high-precision polish pass
    if time.time() < deadline - 3.0:
        polish_steps = 5000
        polish_step = 0.0001
        lambda_bound = 12.0
        
        for it in range(polish_steps):
            if time.time() > deadline - 1.0:
                break
            
            total_forces = compute_forces(points, lambda_bound)
            
            # Adaptive step for polish
            grad_mag = np.linalg.norm(total_forces, axis=1)
            grad_mag_safe = np.where(grad_mag > 1e-6, grad_mag, 1.0)
            adaptive_factor = 1.0 / (1.0 + 0.2 * grad_mag_safe)
            
            points += total_forces * (polish_step * adaptive_factor)[:, np.newaxis]
            
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]
            
            # Decay polish step
            polish_step *= 0.99995
        
        final_r = compute_radius(points)
        if final_r > best_r:
            best_r = final_r
            best_points = points.copy()
        
        # One more check with slightly different lambda to see if we can squeeze more
        if time.time() < deadline - 1.0:
            polish_steps2 = 2000
            polish_step2 = 0.00005
            lambda_bound2 = 15.0
            
            for it in range(polish_steps2):
                if time.time() > deadline - 0.5:
                    break
                
                total_forces = compute_forces(points, lambda_bound2)
                grad_mag = np.linalg.norm(total_forces, axis=1)
                grad_mag_safe = np.where(grad_mag > 1e-6, grad_mag, 1.0)
                adaptive_factor = 1.0 / (1.0 + 0.3 * grad_mag_safe)
                
                points += total_forces * (polish_step2 * adaptive_factor)[:, np.newaxis]
                
                new_radii = np.hypot(points[:, 0], points[:, 1])
                outside = new_radii > 1.0
                if np.any(outside):
                    points[outside] = points[outside] / new_radii[outside, np.newaxis]
            
            final_r2 = compute_radius(points)
            if final_r2 > best_r:
                best_r = final_r2
                best_points = points.copy()

    return best_points
