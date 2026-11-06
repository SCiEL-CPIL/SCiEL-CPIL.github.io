"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def generate_hex_lattice(n):
        """
        Generate n points in the unit disk using a hexagonal lattice.
        """
        # Estimate spacing
        s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        
        # Generate grid
        points = []
        max_idx = int(np.ceil(1.0 / s)) + 2
        
        for j in range(-max_idx, max_idx + 1):
            for i in range(-max_idx, max_idx + 1):
                x = s * (i + 0.5 * j)
                y = s * (np.sqrt(3) / 2.0 * j)
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) > N:
            # Select N points closest to center for better packing
            dists_to_center = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists_to_center)[:N]
            points = points[idx]
        elif len(points) < N:
            # Generate with smaller spacing
            s *= 0.9
            points = []
            max_idx = int(np.ceil(1.0 / s)) + 2
            for j in range(-max_idx, max_idx + 1):
                for i in range(-max_idx, max_idx + 1):
                    x = s * (i + 0.5 * j)
                    y = s * (np.sqrt(3) / 2.0 * j)
                    if np.hypot(x, y) <= 1.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) > N:
                dists_to_center = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists_to_center)[:N]
                points = points[idx]
            else:
                while len(points) < N:
                    angle = np.random.rand() * 2 * np.pi
                    r = np.sqrt(np.random.rand()) * 0.9
                    points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
        
        return points

    def compute_radius(pts):
        """
        Computes the maximum radius for the given points.
        """
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    # 1. Initialization: Hexagonal Lattice
    points = generate_hex_lattice(N)
    
    # Ensure exactly N points
    if len(points) != N:
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.9
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    # Project to unit disk
    radii = np.hypot(points[:, 0], points[:, 1])
    outside = radii > 1.0
    if np.any(outside):
        points[outside] = points[outside] / radii[outside, np.newaxis]

    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization: Force-Based Relaxation with 1/d^3 weighting
    # Use inverse cube weighting for better local optimization
    alpha = 3.0  # Exponent for distance weighting
    
    # Phase 1: Coarse optimization with larger steps
    step_size = 0.05
    decay_rate = 0.9995
    max_iter_1 = 5000
    
    for it in range(max_iter_1):
        if time.time() > deadline - 30.0:  # Reserve 30s for phase 2
            break
        
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        # Check actual radius every 200 iterations
        if it % 200 == 0:
            current_r = compute_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        
        # Compute forces with 1/d^alpha weighting
        # Pair forces: repulsive when too close
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        # Weight: strong repulsion for close pairs
        w_pair = 1.0 / (safe_dists ** alpha)
        np.fill_diagonal(w_pair, 0.0)
        
        # Boundary forces: repulsive when too close to boundary
        safe_boundary = np.where(boundary_dists > 1e-9, boundary_dists, 1.0)
        w_bound = 1.0 / (safe_boundary ** alpha)
        
        # Normalize weights to prevent explosion
        max_w_pair = np.max(w_pair)
        max_w_bound = np.max(w_bound)
        max_w = max(max_w_pair, max_w_bound)
        if max_w > 0:
            w_pair /= max_w
            w_bound /= max_w
        
        # Forces
        factor = w_pair / safe_dists
        np.fill_diagonal(factor, 0.0)
        
        F_pair = np.einsum('ij,ijk->ik', factor, diff)
        
        safe_radii = np.where(radii > 1e-9, radii, 1.0)
        unit_vecs = points / safe_radii[:, np.newaxis]
        F_bound = -w_bound[:, np.newaxis] * unit_vecs
        
        forces = F_pair + F_bound
        
        # Update
        points += forces * step_size
        
        # Project to unit disk
        radii = np.hypot(points[:, 0], points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / radii[outside, np.newaxis]
            
        # Decay step size
        step_size *= decay_rate
        
        # Perturbation to escape local minima
        if it % 100 == 0 and it > 0:
            pert_mag = 0.01
            num_perturb = N // 5
            idx = np.random.choice(N, num_perturb, replace=False)
            perturbations = np.random.randn(num_perturb, 2) * pert_mag
            points[idx] += perturbations
            radii = np.hypot(points[:, 0], points[:, 1])
            outside = radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / radii[outside, np.newaxis]
    
    # Phase 2: Fine-tuning with smaller steps
    # Start from best points found
    points = best_points.copy()
    step_size = 0.01
    decay_rate = 0.9998
    max_iter_2 = 10000
    
    for it in range(max_iter_2):
        if time.time() > deadline:
            break
        
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        # Check actual radius every 100 iterations
        if it % 100 == 0:
            current_r = compute_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        
        # Compute forces with 1/d^alpha weighting
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        w_pair = 1.0 / (safe_dists ** alpha)
        np.fill_diagonal(w_pair, 0.0)
        
        safe_boundary = np.where(boundary_dists > 1e-9, boundary_dists, 1.0)
        w_bound = 1.0 / (safe_boundary ** alpha)
        
        # Normalize
        max_w_pair = np.max(w_pair)
        max_w_bound = np.max(w_bound)
        max_w = max(max_w_pair, max_w_bound)
        if max_w > 0:
            w_pair /= max_w
            w_bound /= max_w
        
        # Forces
        factor = w_pair / safe_dists
        np.fill_diagonal(factor, 0.0)
        
        F_pair = np.einsum('ij,ijk->ik', factor, diff)
        
        safe_radii = np.where(radii > 1e-9, radii, 1.0)
        unit_vecs = points / safe_radii[:, np.newaxis]
        F_bound = -w_bound[:, np.newaxis] * unit_vecs
        
        forces = F_pair + F_bound
        
        # Update
        points += forces * step_size
        
        # Project
        radii = np.hypot(points[:, 0], points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / radii[outside, np.newaxis]
            
        # Decay
        step_size *= decay_rate
        
        # Small perturbations
        if it % 200 == 0 and it > 0:
            pert_mag = 0.002
            num_perturb = N // 10
            idx = np.random.choice(N, num_perturb, replace=False)
            perturbations = np.random.randn(num_perturb, 2) * pert_mag
            points[idx] += perturbations
            radii = np.hypot(points[:, 0], points[:, 1])
            outside = radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / radii[outside, np.newaxis]
    
    # Phase 3: Final high-precision polish
    points = best_points.copy()
    step_size = 0.002
    decay_rate = 0.9999
    max_iter_3 = 5000
    
    for it in range(max_iter_3):
        if time.time() > deadline:
            break
        
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        # Check radius every 50 iterations
        if it % 50 == 0:
            current_r = compute_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        
        # Compute forces
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        w_pair = 1.0 / (safe_dists ** alpha)
        np.fill_diagonal(w_pair, 0.0)
        
        safe_boundary = np.where(boundary_dists > 1e-9, boundary_dists, 1.0)
        w_bound = 1.0 / (safe_boundary ** alpha)
        
        max_w_pair = np.max(w_pair)
        max_w_bound = np.max(w_bound)
        max_w = max(max_w_pair, max_w_bound)
        if max_w > 0:
            w_pair /= max_w
            w_bound /= max_w
        
        factor = w_pair / safe_dists
        np.fill_diagonal(factor, 0.0)
        
        F_pair = np.einsum('ij,ijk->ik', factor, diff)
        
        safe_radii = np.where(radii > 1e-9, radii, 1.0)
        unit_vecs = points / safe_radii[:, np.newaxis]
        F_bound = -w_bound[:, np.newaxis] * unit_vecs
        
        forces = F_pair + F_bound
        
        points += forces * step_size
        
        radii = np.hypot(points[:, 0], points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / radii[outside, np.newaxis]
            
        step_size *= decay_rate
    
    # Final check
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = best_points.copy()

    return best_points
# EVOLVE-BLOCK-END
