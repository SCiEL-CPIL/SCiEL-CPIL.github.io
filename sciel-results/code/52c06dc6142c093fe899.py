"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial constraint
        radial_dists = np.linalg.norm(pts, axis=1)
        if np.any(radial_dists > 1.0):
            return -1.0
        min_radial = 1.0 - np.max(radial_dists)
        
        # Pairwise constraint
        # Vectorized distance calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Get upper triangle
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        min_pair = np.min(pair_dists) / 2.0
        
        return min(min_pair, min_radial)

    def physics_optimize(pts, max_iter=500, dt=0.001, damping=0.95):
        """
        Physics-based optimization using repulsive forces.
        """
        pts = pts.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsive forces
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1)
            dists = np.sqrt(dists_sq)
            
            # Avoid division by zero
            mask = dists > 1e-10
            dists_safe = np.where(mask, dists, 1.0)
            
            # Force magnitude: inverse square law
            force_mag = 1.0 / (dists_sq + 1e-10)
            force_mag[~mask] = 0.0
            
            # Force vector: direction from j to i
            force_vec = diff / dists_safe[:, :, np.newaxis]
            force_vec *= force_mag[:, :, np.newaxis]
            
            # Sum forces for each particle
            forces += np.sum(force_vec, axis=1)
            
            # Container wall repulsion
            radial_dists = np.linalg.norm(pts, axis=1)
            wall_force = np.zeros_like(pts)
            inside_mask = radial_dists < 1.0
            wall_force[inside_mask] = pts[inside_mask] / np.maximum(radial_dists[inside_mask, np.newaxis], 1e-10)
            # Scale wall force to be comparable to pairwise forces
            wall_force *= 5.0
            forces += wall_force
            
            # Update velocities and positions
            # Simple Euler integration with damping
            pts += forces * dt
            pts *= damping  # Damping to prevent oscillation
            
            # Project back to unit disk if necessary
            radial_dists = np.linalg.norm(pts, axis=1)
            outside_mask = radial_dists > 1.0
            if np.any(outside_mask):
                pts[outside_mask] /= radial_dists[outside_mask, np.newaxis]
        
        return pts

    def generate_hex_grid_perturbed(spacing=0.055):
        """
        Generate points on a hexagonal grid with random perturbation.
        """
        pts = []
        # Generate grid points
        k = int(1.0 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x**2 + y**2 <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        if len(pts) < N:
            # If not enough, try smaller spacing
            return generate_hex_grid_perturbed(spacing * 0.95)
        
        # Select N points closest to center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
        
        # Add small random perturbation to break symmetry
        perturbation = np.random.normal(0, spacing * 0.1, size=pts.shape)
        pts += perturbation
        
        # Project back to unit disk
        radial_dists = np.linalg.norm(pts, axis=1)
        outside_mask = radial_dists > 1.0
        if np.any(outside_mask):
            pts[outside_mask] /= radial_dists[outside_mask, np.newaxis]
        
        return pts

    def softmin_objective(params, tau=0.005):
        """
        Smooth objective function: -softmin of all constraints.
        """
        pts = params.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        # Radial distances
        radial_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        all_vals = np.concatenate([pair_dists, radial_dists])
        
        # Softmin
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        exp_terms = np.exp(-shifted / tau)
        softmin_val = min_val - tau * np.log(np.sum(exp_terms))
        
        return -softmin_val

    def softmin_gradient(params, tau=0.005):
        """
        Gradient of the softmin objective.
        """
        pts = params.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1.0)
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        # Radial distances
        radial_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        all_vals = np.concatenate([pair_dists, radial_dists])
        
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        exp_terms = np.exp(-shifted / tau)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        grad_z = -weights
        
        grad_pts = np.zeros_like(pts)
        
        i_idx, j_idx = iu
        w_pair = grad_z[:len(pair_dists)]
        
        vec = pts[i_idx] - pts[j_idx]
        norms = dists[iu]
        unit_vec = vec / norms[:, np.newaxis]
        
        grad_i = 0.5 * unit_vec * w_pair[:, np.newaxis]
        grad_j = -0.5 * unit_vec * w_pair[:, np.newaxis]
        
        np.add.at(grad_pts, i_idx, grad_i)
        np.add.at(grad_pts, j_idx, grad_j)
        
        w_radial = grad_z[len(pair_dists):]
        norms_rad = np.linalg.norm(pts, axis=1)
        norms_rad_safe = np.where(norms_rad > 1e-10, norms_rad, 1.0)
        unit_rad = pts / norms_rad_safe[:, np.newaxis]
        grad_pts += -w_radial[:, np.newaxis] * unit_rad
        
        return grad_pts.flatten()

    def optimize_from_initial(pts, max_iter=200):
        """Run L-BFGS-B optimization from initial points."""
        x0 = pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(
                softmin_objective, 
                x0, 
                method='L-BFGS-B', 
                jac=softmin_gradient,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < 0:
                new_pts = result.x.reshape(N, 2)
                score = compute_score(new_pts)
                if score > 0:
                    return new_pts, score
        except Exception:
            pass
        return None, -1.0

    # Strategy: Hex grid with perturbation + physics simulation + gradient polish
    best_pts = None
    best_score = -1.0
    
    # Multiple restarts with different perturbations
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        # Generate initial configuration
        init_pts = generate_hex_grid_perturbed(0.055)
        
        # Apply physics optimization
        phys_pts = physics_optimize(init_pts, max_iter=300)
        
        # Evaluate physics result
        phys_score = compute_score(phys_pts)
        if phys_score > best_score:
            best_score = phys_score
            best_pts = phys_pts
        
        # Apply gradient-based polish
        if time.time() < deadline - 10:
            opt_pts, opt_score = optimize_from_initial(phys_pts, max_iter=100)
            if opt_pts is not None and opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
    
    # Final polish if time remains
    if best_pts is not None and time.time() < deadline - 5:
        opt_pts, opt_score = optimize_from_initial(best_pts, max_iter=50)
        if opt_pts is not None and opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts

    # Fallback
    if best_pts is None:
        best_pts = np.zeros((N, 2))
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            best_pts[i, 0] = (col - n_per_row/2) * spacing
            best_pts[i, 1] = (row - n_per_row/2) * spacing

    return best_pts
# EVOLVE-BLOCK-END
