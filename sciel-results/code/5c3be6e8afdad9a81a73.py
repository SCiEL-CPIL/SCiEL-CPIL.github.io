"""Simulated annealing packing of 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import random

N = 50

def compute_radius(pos):
    """Compute the actual radius of the packing."""
    pos = pos.reshape(-1, 2)
    dists = np.sqrt(np.sum(pos**2, axis=1))
    if np.any(dists > 1.0):
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    min_boundary_dist = np.min(1.0 - dists)
    return min(min_pair_dist / 2.0, min_boundary_dist)

def optimize_scale(pos):
    """Find optimal uniform scale to maximize radius."""
    pos = pos.reshape(-1, 2)
    dists = np.sqrt(np.sum(pos**2, axis=1))
    max_dist = np.max(dists)
    if max_dist < 1e-10:
        return pos, compute_radius(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    r_pair = min_pair_dist / 2.0
    
    def radius_at_scale(s):
        if s * max_dist > 1.0:
            return 0.0
        r_p = s * r_pair
        r_b = 1.0 - s * max_dist
        return min(r_p, r_b)
    
    s_opt = 1.0 / (r_pair + max_dist)
    s_opt = np.clip(s_opt, 0.5, 1.5)
    
    r_final = radius_at_scale(s_opt)
    r_orig = radius_at_scale(1.0)
    
    if r_final > r_orig:
        return pos * s_opt, r_final
    else:
        return pos, r_orig

def compute_single_energy(pos, idx, sigma=0.05):
    """Compute the energy contribution of particle idx to the total system energy."""
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(pos[idx])
    e_bound = np.exp(-d_bound / sigma) if d_bound > 0 else np.exp(-1.0 / sigma) * 10 # High penalty if outside
    
    # Distances to other particles
    diffs = pos[idx] - pos
    dists = np.linalg.norm(diffs, axis=1)
    # Exclude self
    dists[idx] = np.inf
    # Pairwise repulsion: sum of exp(-d/sigma)
    e_pairs = np.sum(np.exp(-dists / sigma))
    
    return e_bound + e_pairs

def simulated_annealing(pos_initial, max_iter=200000, seed=42):
    """Run simulated annealing from given initial position with optimized energy calculation."""
    np.random.seed(seed)
    pos = pos_initial.copy()
    sigma = 0.05
    
    # Initial energy calculation
    current_energy = sum(compute_single_energy(pos, i, sigma) for i in range(N))
    # Note: The sum of single energies counts each pair twice. 
    # Let's define the total energy as 0.5 * sum(single_energies) + sum(boundary_energies)
    # Or just use the sum as a proxy, as the factor doesn't matter for SA acceptance.
    # However, for consistency, let's just use the sum.
    
    T = 1.0
    T_min = 1e-5
    cooling = 0.9999
    base_step = 0.1
    
    best_pos = pos.copy()
    best_energy = current_energy
    
    for it in range(max_iter):
        idx = np.random.randint(0, N)
        
        # Calculate old energy contribution of idx
        old_contrib = compute_single_energy(pos, idx, sigma)
        
        # Generate new position
        step_size = base_step * (1.0 + T)
        new_pos = pos.copy()
        new_pos[idx] += np.random.normal(0, step_size, 2)
        
        # Calculate new energy contribution of idx
        new_contrib = compute_single_energy(new_pos, idx, sigma)
        
        # Delta E
        delta_e = new_contrib - old_contrib
        
        # Acceptance criterion
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(T, 1e-10)):
            pos = new_pos
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = pos.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    pos = best_pos
    
    # Local refinement using L-BFGS-B on a smooth objective
    def smooth_objective(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        # Use the same exponential potential
        e = 0
        for i in range(N):
            e += compute_single_energy(pos, i, sigma)
        return e
    
    try:
        result = minimize(smooth_objective, pos.reshape(-1), method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < best_energy:
            pos = result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return pos

def local_refinement(pos, max_iter=200):
    """Directly optimize to maximize the minimum distance using gradient-free approach."""
    pos = pos.copy()
    
    def negative_min_radius(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        dists_to_center = np.sqrt(np.sum(pos**2, axis=1))
        if np.any(dists_to_center > 1.0):
            return 100.0 + np.sum(np.maximum(0, dists_to_center - 1.0))**2
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair = np.min(dist_mat)
        min_boundary = np.min(1.0 - dists_to_center)
        
        r = min(min_pair / 2.0, min_boundary)
        return -r
    
    # Use Nelder-Mead for local refinement
    try:
        result = minimize(negative_min_radius, pos.reshape(-1), method='Nelder-Mead',
                         options={'maxiter': max_iter * 50, 'xatol': 1e-10, 'fatol': 1e-12})
        if -result.fun > compute_radius(pos):
            pos = result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return pos

def construct():
    np.random.seed(42)
    random.seed(42)
    
    best_pos = None
    best_r = 0.0
    
    # Multiple restarts with different initial configurations
    for restart in range(5):
        seed = 42 + restart * 1000 + 7
        
        # Generate initial hexagonal lattice with varying parameters
        points = []
        
        if restart % 3 == 0:
            r_hex = 0.125
        elif restart % 3 == 1:
            r_hex = 0.13
        else:
            r_hex = 0.135
        
        sqrt3 = np.sqrt(3.0)
        
        for i in range(-12, 13):
            for j in range(-12, 13):
                x = i * r_hex + (j % 2) * r_hex / 2.0
                y = j * r_hex * sqrt3 / 2.0
                # Add small random perturbation
                x += np.random.uniform(-0.015, 0.015)
                y += np.random.uniform(-0.015, 0.015)
                if x*x + y*y <= 0.94:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) > N:
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)[:N]
            points = points[idx]
        elif len(points) < N:
            while len(points) < N:
                angle = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0, 0.75)
                points = np.vstack([points, [[r*np.cos(angle), r*np.sin(angle)]]])
        
        pos_initial = points.copy()
        
        # Run simulated annealing with more iterations
        pos = simulated_annealing(pos_initial, max_iter=200000, seed=seed)
        
        # Optimize scale
        pos, r = optimize_scale(pos)
        
        # Local refinement
        pos = local_refinement(pos, max_iter=100)
        
        # Re-optimize scale after refinement
        pos, r = optimize_scale(pos)
        
        # Final validation
        dists = np.sqrt(np.sum(pos**2, axis=1))
        if np.any(dists > 1.0):
            mask = dists > 1.0
            pos[mask] = pos[mask] / dists[mask, np.newaxis] * 0.9999
            r = compute_radius(pos)
        
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        
        # Early exit if we're close to known best
        if best_r > 0.1258:
            break
    
    if best_pos is None:
        # Fallback: return hexagonal lattice
        points = []
        r_hex = 0.13
        sqrt3 = np.sqrt(3.0)
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * r_hex + (j % 2) * r_hex / 2.0
                y = j * r_hex * sqrt3 / 2.0
                if x*x + y*y <= 0.95:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)[:N]
            points = points[idx]
        best_pos = points
    
    return best_pos
