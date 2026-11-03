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
    
    # Optimal scale s such that s * r_pair = 1 - s * max_dist
    # s (r_pair + max_dist) = 1
    s_opt = 1.0 / (r_pair + max_dist)
    
    # Ensure s is reasonable
    s_opt = np.clip(s_opt, 0.1, 10.0)
    
    new_pos = pos * s_opt
    r_final = compute_radius(new_pos)
    
    # Check original
    r_orig = compute_radius(pos)
    
    if r_final > r_orig:
        return new_pos, r_final
    else:
        return pos, r_orig

def simulated_annealing(pos_initial, max_iter=50000, seed=42):
    """Run simulated annealing from given initial position."""
    np.random.seed(seed)
    pos = pos_initial.copy()
    
    def energy(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        dists_to_center = np.sqrt(np.sum(pos**2, axis=1))
        
        # Smooth boundary penalty: penalize if dist > 1 - r_target? 
        # Actually, we want to maximize r. 
        # Let's use a potential that encourages separation.
        # E = sum (max(0, 2r - d_ij)^2) + sum (max(0, r + d_i - 1)^2)
        # But r is not fixed here. 
        # Instead, let's minimize the "overlap" assuming a target radius.
        # Or just minimize the sum of inverse distances? No, that's singular.
        
        # Let's stick to a simple penalty for close pairs and boundary violations.
        # We want to maximize the minimum distance.
        # A good proxy is minimizing the sum of (target - dist)^2 for dist < target.
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, 1.0)
        
        # Target distance is roughly 2 * 0.125 = 0.25
        # Let's use a dynamic target or just a fixed one based on current min dist?
        # Fixed target 0.25 is okay.
        
        # Boundary: we want dists_to_center < 1 - r.
        # Let's penalize dists_to_center > 0.85 (assuming r ~ 0.15)
        
        boundary_violation = np.maximum(0, dists_to_center - 0.85)
        boundary_penalty = np.sum(boundary_violation**2) * 50
        
        # Pairwise: penalize if dist < 0.25
        pair_violation = np.maximum(0, 0.25 - dist_mat)
        pair_penalty = np.sum(pair_violation**2) * 50
        
        return boundary_penalty + pair_penalty

    T = 50.0
    T_min = 1e-6
    cooling = 0.9995
    base_step = 0.05
    
    best_pos = pos.copy()
    best_energy = energy(pos.reshape(-1))
    current_energy = best_energy
    
    for it in range(max_iter):
        # Adaptive step size
        step_size = base_step * (1.0 + T / 50.0)
        
        new_pos = pos.copy()
        idx = np.random.randint(0, N)
        new_pos[idx] += np.random.normal(0, step_size, 2)
        
        new_energy = energy(new_pos.reshape(-1))
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
    return pos

def local_refinement(pos, max_iter=200):
    """Refine using L-BFGS-B on a smooth objective approximating -min_dist."""
    pos = pos.copy()
    
    def smooth_min_dist_obj(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        dists_to_center = np.sqrt(np.sum(pos**2, axis=1))
        
        # Boundary terms: exp(-alpha * (1 - dist))
        # Pair terms: exp(-alpha * dist)
        # We want to maximize min(dist_pair/2, 1-dist_center)
        # This is equivalent to minimizing max(2/dist_pair - 1, 1/(1-dist_center) - 1)? No.
        
        # Let's use the log-sum-exp trick.
        # We want to minimize -min(d_i).
        # min(d_i) ~ -1/alpha * ln(sum(exp(-alpha * d_i)))
        # So we want to minimize 1/alpha * ln(sum(exp(-alpha * d_i)))
        
        # d_i are the "constraints":
        # For pairs: d_ij
        # For boundary: 1 - dist_i
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        # Get upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        pair_dists = dist_mat[iu]
        
        # Boundary distances
        bound_dists = 1.0 - dists_to_center
        
        # Combine all distances
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Clip to avoid overflow/underflow and handle negatives (if any)
        # If a dist is negative, it's a huge violation.
        # exp(-alpha * dist) will be huge if dist is negative.
        
        alpha = 50.0 # Higher alpha = sharper minimum
        
        # To prevent overflow, shift by min
        min_d = np.min(all_dists)
        if min_d < -10: # Huge violation
            return 1000.0
        
        # exp(-alpha * (d - min_d))
        # ln(sum(exp(-alpha * d))) = -alpha * min_d + ln(sum(exp(-alpha * (d - min_d))))
        # We want to minimize -min_d, which is proportional to ln(sum(exp(-alpha * d)))
        
        # Actually, if we minimize (1/alpha) * ln(sum(exp(-alpha * d))), 
        # for large alpha, this approximates min(d).
        # So we want to MINIMIZE this quantity? 
        # Wait. min(d) is the value we want to MAXIMIZE.
        # So we want to MINIMIZE -min(d).
        # -min(d) ~ 1/alpha * ln(sum(exp(-alpha * d))) ? No.
        # Let x = -d. max(x) = -min(d).
        # max(x) ~ 1/alpha * ln(sum(exp(alpha * x))) = 1/alpha * ln(sum(exp(-alpha * d)))
        # So we want to MINIMIZE 1/alpha * ln(sum(exp(-alpha * d)))
        
        shifted = -alpha * (all_dists - min_d)
        # If min_d is very negative, shifted is large positive -> exp overflows.
        # But we check min_d < -10.
        
        # Use stable logsumexp
        # logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
        z = -alpha * all_dists
        z_max = np.max(z)
        if z_max > 700:
             return 1000.0
        log_sum = z_max + np.log(np.sum(np.exp(z - z_max)))
        val = log_sum / alpha
        
        return val

    try:
        result = minimize(smooth_min_dist_obj, pos.reshape(-1), method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < smooth_min_dist_obj(pos.reshape(-1)):
             # Check if it actually improved the radius
             new_pos = result.x.reshape(-1, 2)
             if compute_radius(new_pos) > compute_radius(pos):
                 pos = new_pos
    except Exception:
        pass
    
    return pos

def construct():
    np.random.seed(42)
    random.seed(42)
    
    best_pos = None
    best_r = 0.0
    
    # Multiple restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        seed = 42 + restart * 1000 + 7
        
        # Generate initial hexagonal lattice
        points = []
        r_hex = 0.125
        sqrt3 = np.sqrt(3.0)
        
        for i in range(-12, 13):
            for j in range(-12, 13):
                x = i * r_hex + (j % 2) * r_hex / 2.0
                y = j * r_hex * sqrt3 / 2.0
                # Add small random perturbation
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                if x*x + y*y <= 0.9:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) > N:
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)[:N]
            points = points[idx]
        elif len(points) < N:
            while len(points) < N:
                angle = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0, 0.7)
                points = np.vstack([points, [[r*np.cos(angle), r*np.sin(angle)]]])
        
        pos_initial = points.copy()
        
        # Run simulated annealing
        pos = simulated_annealing(pos_initial, max_iter=50000, seed=seed)
        
        # Optimize scale
        pos, r = optimize_scale(pos)
        
        # Local refinement
        pos = local_refinement(pos)
        
        # Re-optimize scale
        pos, r = optimize_scale(pos)
        
        # Final validation and clipping
        dists = np.sqrt(np.sum(pos**2, axis=1))
        if np.any(dists > 1.0):
            mask = dists > 1.0
            pos[mask] = pos[mask] / dists[mask, np.newaxis] * 0.9999
            r = compute_radius(pos)
        
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
            
    if best_pos is None:
        # Fallback
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
