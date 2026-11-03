"""Simulated annealing packing of 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import random
import time

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
    pos = pos.reshape(-1, 2).copy()
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
    
    # The optimal scale balances pair distance and boundary
    # s * r_pair = 1 - s * max_dist => s = 1/(r_pair + max_dist)
    s_opt = 1.0 / (r_pair + max_dist)
    s_opt = np.clip(s_opt, 0.3, 2.0)
    
    r_final = radius_at_scale(s_opt)
    r_orig = radius_at_scale(1.0)
    
    if r_final > r_orig:
        return pos * s_opt, r_final
    else:
        return pos, r_orig

def make_energy(pos, target_r):
    """Create a smooth energy function that penalizes violations relative to target radius."""
    pos = pos.reshape(-1, 2)
    
    def energy(pos_flat):
        p = pos_flat.reshape(-1, 2)
        dists_to_center = np.sqrt(np.sum(p**2, axis=1))
        
        # Boundary: circles must be inside unit circle
        # Penalty if any circle center is too far from origin
        boundary_viol = np.maximum(0, dists_to_center - (1.0 - target_r))
        boundary_pen = np.sum(boundary_viol**2) * 500
        
        # Pairwise: circles must not overlap
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, 1.0)
        
        # Penalty for pairs closer than 2*target_r
        close_pairs = np.maximum(0, 2.0 * target_r - dist_mat)
        pair_pen = np.sum(close_pairs**2) * 500
        
        # Extra strong penalty for very close pairs to break symmetry
        very_close = np.maximum(0, 1.5 * target_r - dist_mat)
        very_close_pen = np.sum(very_close**2) * 2000
        
        return boundary_pen + pair_pen + very_close_pen
    
    return energy

def simulated_annealing(pos_initial, max_iter=80000, seed=42, time_limit=15.0):
    """Run simulated annealing from given initial position."""
    np.random.seed(seed)
    pos = pos_initial.copy()
    
    # Start with a reasonable target radius
    initial_r = compute_radius(pos)
    if initial_r < 0.05:
        initial_r = 0.12
    
    target_r = initial_r * 1.05
    energy_fn = make_energy(pos, target_r)
    
    T = 5.0
    T_min = 1e-6
    cooling = 0.9998
    base_step = 0.08
    
    best_pos = pos.copy()
    best_energy = energy_fn(pos.reshape(-1))
    current_energy = best_energy
    
    start_time = time.time()
    
    for it in range(max_iter):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive step size based on temperature
        step_size = base_step * (0.5 + T / 5.0)
        
        new_pos = pos.copy()
        idx = np.random.randint(0, N)
        new_pos[idx] += np.random.normal(0, step_size, 2)
        
        new_energy = energy_fn(new_pos.reshape(-1))
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(T, 1e-10)):
            pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = pos.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    pos = best_pos
    
    # Local refinement: try to increase target_r and re-optimize
    for refinement_iter in range(3):
        current_r = compute_radius(pos)
        if current_r <= 0:
            break
        new_target = current_r * 1.01
        energy_fn = make_energy(pos, new_target)
        
        try:
            result = minimize(energy_fn, pos.reshape(-1), method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            new_pos = result.x.reshape(-1, 2)
            new_r = compute_radius(new_pos)
            if new_r > current_r:
                pos = new_pos
            else:
                break
        except Exception:
            break
    
    return pos

def local_refinement(pos, max_iter=150, time_limit=10.0):
    """Directly optimize to maximize the minimum radius."""
    pos = pos.copy()
    start_time = time.time()
    
    def negative_min_radius(pos_flat):
        p = pos_flat.reshape(-1, 2)
        dists_to_center = np.sqrt(np.sum(p**2, axis=1))
        
        # Heavy penalty for boundary violations
        if np.any(dists_to_center > 1.0):
            viol = np.maximum(0, dists_to_center - 1.0)
            return 100.0 + np.sum(viol**2)
        
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair = np.min(dist_mat)
        min_boundary = np.min(1.0 - dists_to_center)
        
        r = min(min_pair / 2.0, min_boundary)
        return -r
    
    current_r = compute_radius(pos)
    
    # Use Powell method which is often better for this type of problem
    try:
        result = minimize(negative_min_radius, pos.reshape(-1), method='Powell',
                         options={'maxiter': max_iter * 20, 'xatol': 1e-9, 'fatol': 1e-11})
        new_r = -result.fun
        if new_r > current_r:
            pos = result.x.reshape(-1, 2)
            current_r = new_r
    except Exception:
        pass
    
    # Try Nelder-Mead as well
    try:
        result = minimize(negative_min_radius, pos.reshape(-1), method='Nelder-Mead',
                         options={'maxiter': max_iter * 30, 'xatol': 1e-10, 'fatol': 1e-12})
        new_r = -result.fun
        if new_r > current_r:
            pos = result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return pos

def generate_hex_initial(seed, spacing):
    """Generate a hexagonal lattice initial configuration."""
    np.random.seed(seed)
    points = []
    sqrt3 = np.sqrt(3.0)
    
    for i in range(-15, 16):
        for j in range(-15, 16):
            x = i * spacing + (j % 2) * spacing / 2.0
            y = j * spacing * sqrt3 / 2.0
            # Add small random perturbation
            x += np.random.uniform(-0.01, 0.01)
            y += np.random.uniform(-0.01, 0.01)
            if x*x + y*y <= 0.93:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) > N:
        dists = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, 0.8)
            points = np.vstack([points, [[r*np.cos(angle), r*np.sin(angle)]]])
    
    return points

def construct():
    np.random.seed(42)
    random.seed(42)
    
    best_pos = None
    best_r = 0.0
    start_time = time.time()
    
    # Try multiple restarts with different initial configurations
    for restart in range(10):
        if time.time() - start_time > 100:
            break
            
        seed = 42 + restart * 7919 + 13
        
        # Vary the hexagonal spacing
        spacings = [0.24, 0.25, 0.26, 0.27, 0.28]
        spacing = spacings[restart % len(spacings)]
        
        pos_initial = generate_hex_initial(seed, spacing)
        
        # Run simulated annealing
        pos = simulated_annealing(pos_initial, max_iter=60000, seed=seed, time_limit=12.0)
        
        # Optimize scale
        pos, r = optimize_scale(pos)
        
        # Local refinement
        pos = local_refinement(pos, max_iter=80, time_limit=8.0)
        
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
        best_pos = generate_hex_initial(42, 0.25)
    
    return best_pos
