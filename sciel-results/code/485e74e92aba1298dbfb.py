import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # --- Step 1: Initialization ---
    # Hexagonal lattice is a strong baseline.
    def hex_lattice_init(n):
        # Target radius is ~0.053. Spacing ~0.106.
        spacing = 0.11
        while True:
            k = int(1.0 / spacing) + 2
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = i * spacing
                    y = j * spacing * np.sqrt(3)
                    if (j % 2) != 0:
                        x += spacing / 2
                    points.append([x, y])
            points = np.array(points)
            r = np.hypot(points[:, 0], points[:, 1])
            # Keep points that fit comfortably
            mask = r <= 1.0 - spacing
            points = points[mask]
            
            if len(points) >= n:
                # Select n closest to center
                dists = r[mask]
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            else:
                spacing *= 0.95
                if spacing < 0.05:
                    # Fallback: random
                    return np.random.rand(n, 2) * 2 - 1

    pts = hex_lattice_init(N)
    
    # Ensure validity
    r_init = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(r_init > 1.0):
        pts = pts / r_init[:, np.newaxis] * 0.999

    def compute_radius(pts):
        # Use KDTree for speed
        tree = cKDTree(pts)
        # Find all pairs with distance < some threshold? 
        # We need the MINIMUM distance.
        # cKDTree.query_pairs is good for finding close pairs.
        # But we need the global min.
        # Let's just compute min distance via query.
        # For each point, find 1 nearest neighbor.
        dists, _ = tree.query(pts, k=2) # k=2 to exclude self
        min_pair = np.min(dists[:, 1]) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    current_pts = best_pts.copy()
    
    # --- Step 2: Simulated Annealing with Soft Penalty ---
    # Objective: Minimize E = sum_{i<j} exp(-d_ij^2 / sigma^2) + sum_i exp(-(1-|p_i|)^2 / sigma_b^2)
    # Gradient of E w.r.t p_i is used for updates.
    # We use a temperature schedule to allow "bad" moves (increasing E) early on,
    # which helps escape local minima.
    
    # Parameters
    T_initial = 0.05
    T_final = 1e-6
    cooling_rate = 0.995
    
    # Soft penalty parameters
    # We want to penalize overlaps (d < 2r) and boundary violations (|p| > 1-r).
    # Since we don't know r, we use a soft repulsion that becomes very strong as d -> 0.
    # A common choice is 1/d^2 or exp(-d^2/sigma). 
    # Let's use a repulsive potential V(d) = 1/d^2 for d < cutoff, 0 otherwise.
    # And boundary potential V_b(r) = 1/(1-r)^2 for r > 1-cutoff_b.
    
    # To make it differentiable and smooth, we use:
    # V_rep(d) = 1 / (d^2 + eps)^2  (always active, decays fast)
    # V_bound(r) = 1 / (1-r+eps)^2  (if r > 1-cutoff)
    
    # However, computing gradients for all pairs is O(N^2).
    # We use cKDTree to find neighbors within a cutoff distance.
    # Cutoff for repulsion: ~0.3 (covers ~3x target diameter)
    # Cutoff for boundary: 0.1
    
    cutoff_rep = 0.3
    eps = 1e-4
    
    T = T_initial
    
    # We will run for a fixed number of iterations or until time runs out.
    # Each iteration:
    # 1. Compute forces (gradients of potential energy) using KDTree neighbors.
    # 2. Propose a move: p_new = p_old - alpha * grad + noise.
    # 3. Accept based on Metropolis criterion.
    
    # To speed up, we only update a subset of points or do full sweep?
    # Full sweep with gradient descent is fast if we use KDTree.
    # Let's do full sweep gradient descent with annealing noise.
    
    for iteration in range(5000):
        if time.time() > deadline - 10:
            break
            
        # Update Tree
        tree = cKDTree(current_pts)
        
        # Calculate Forces
        forces = np.zeros_like(current_pts)
        
        # 1. Repulsive forces from neighbors
        # Find all pairs within cutoff_rep
        pairs = tree.query_pairs(cutoff_rep)
        
        for i, j in pairs:
            d_vec = current_pts[i] - current_pts[j]
            d_sq = np.dot(d_vec, d_vec)
            d = np.sqrt(d_sq)
            if d < 1e-6:
                # Random nudge
                nudge = np.random.randn(2) * 0.01
                forces[i] += nudge
                forces[j] -= nudge
                continue
            
            # Potential V = 1/(d^2+eps)^2
            # Gradient dV/dd = -4 d / (d^2+eps)^3
            # Force on i is -grad V = +4 d / (d^2+eps)^3 * (d_vec/d) = 4 / (d^2+eps)^3 * d_vec
            # Wait, V(d) = (d^2+eps)^-2. dV/dd = -2(d^2+eps)^-3 * 2d = -4d(d^2+eps)^-3.
            # Force = -dV/dp = -dV/dd * dd/dp = -dV/dd * (d_vec/d) = 4d(d^2+eps)^-3 * d_vec/d = 4 d_vec / (d^2+eps)^3.
            
            factor = 4.0 / (d_sq + eps)**3
            f_vec = factor * d_vec
            
            forces[i] += f_vec
            forces[j] -= f_vec
        
        # 2. Boundary forces
        r_pts = np.hypot(current_pts[:, 0], current_pts[:, 1])
        # We want to push points inside if r > 1 - margin.
        # Let's use a soft potential for r > 0.95
        mask_bound = r_pts > 0.95
        if np.any(mask_bound):
            # Potential V_b = 1/(1-r+eps)^2 for r > 1-eps
            # Actually, let's just use a linear push back that increases as r -> 1.
            # V_b = (r - 0.95)^2 / (1-r+eps)^2 ?
            # Simpler: Force = -k * (r - 0.95) / (1-r+eps) * (p/r)
            # Let's use: F = - (r - 0.9) / (1 - r + eps) * p / r
            # This is a strong push when r is close to 1.
            
            weights = (r_pts[mask_bound] - 0.9) / (1.0 - r_pts[mask_bound] + eps)
            # Direction is -p / |p|
            dir_bound = -current_pts[mask_bound] / r_pts[mask_bound, np.newaxis]
            forces[mask_bound] += dir_bound * weights[:, np.newaxis]
            
        # 3. Centering force (weak) to keep points in center?
        # Not strictly necessary if boundary force is good, but helps symmetry.
        # F_center = -0.01 * p
        # forces -= 0.01 * current_pts
        
        # Normalize forces to avoid instability
        f_mags = np.linalg.norm(forces, axis=1)
        max_f = 1.0
        mask_f = f_mags > max_f
        if np.any(mask_f):
            forces[mask_f] = forces[mask_f] / f_mags[mask_f, np.newaxis] * max_f
        
        # Propose move
        # Step size (learning rate)
        alpha = 0.05
        # Noise
        noise = np.random.randn(N, 2) * T * 0.1
        
        new_pts = current_pts + alpha * forces + noise
        
        # Enforce hard boundary constraint (project back)
        r_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask_out = r_new > 1.0
        if np.any(mask_out):
            new_pts[mask_out] = new_pts[mask_out] / r_new[mask_out, np.newaxis] * 0.9999
            
        # Calculate "Energy" (Penalty)
        # We need a fast energy calculation for Metropolis.
        # E = sum_{pairs in cutoff} 1/(d^2+eps)^2 + sum_{bound} 1/(1-r+eps)^2
        # This is expensive if we do it fully every step.
        # Approximation: Only check if the move improves the local neighborhood?
        # Or just use the force magnitude as a proxy? No.
        # Let's compute energy for the new points using the SAME pairs?
        # The set of pairs might change.
        # For speed, let's only accept if the move doesn't increase the "worst" constraint significantly?
        # Actually, let's just use a simplified energy check:
        # Check min distance and min boundary for new_pts.
        # If min_dist_new > min_dist_old - tol OR min_bound_new > min_bound_old - tol, accept.
        # This is a greedy improvement with noise.
        
        # To make it true annealing, we need E.
        # Let's compute E only every 10 steps to save time, and use greedy otherwise?
        # Or, let's just do Greedy Hill Climbing with Noise (Simulated Annealing approximation).
        # If new_pts has a better "score" (radius), accept.
        # Else, accept with probability exp((r_new - r_old) / T).
        
        # Computing radius is O(N log N). 5000 iterations * O(N log N) might be too slow?
        # N=300. N log N ~ 2400. 5000 * 2400 = 12M ops. Should be fine in Python with NumPy?
        # Let's try.
        
        r_curr = compute_radius(current_pts)
        r_new_val = compute_radius(new_pts)
        
        # Metropolis Criterion
        delta_r = r_new_val - r_curr
        if delta_r > 0:
            current_pts = new_pts
            if r_new_val > best_radius:
                best_radius = r_new_val
                best_pts = new_pts.copy()
        else:
            # Accept bad move with probability exp(delta_r / T)
            # delta_r is negative.
            if np.random.rand() < np.exp(delta_r / T):
                current_pts = new_pts
        
        # Cool down
        T *= cooling_rate
        
    # Final Polish: Local Gradient Descent
    # Now that we are in a good basin, do precise local optimization.
    for iteration in range(100):
        if time.time() > deadline - 2:
            break
            
        tree = cKDTree(current_pts)
        forces = np.zeros_like(current_pts)
        
        pairs = tree.query_pairs(cutoff_rep)
        for i, j in pairs:
            d_vec = current_pts[i] - current_pts[j]
            d_sq = np.dot(d_vec, d_vec)
            d = np.sqrt(d_sq)
            if d < 1e-6: continue
            factor = 4.0 / (d_sq + eps)**3
            f_vec = factor * d_vec
            forces[i] += f_vec
            forces[j] -= f_vec
            
        r_pts = np.hypot(current_pts[:, 0], current_pts[:, 1])
        mask_bound = r_pts > 0.95
        if np.any(mask_bound):
            weights = (r_pts[mask_bound] - 0.9) / (1.0 - r_pts[mask_bound] + eps)
            dir_bound = -current_pts[mask_bound] / r_pts[mask_bound, np.newaxis]
            forces[mask_bound] += dir_bound * weights[:, np.newaxis]
            
        f_mags = np.linalg.norm(forces, axis=1)
        max_f = 1.0
        mask_f = f_mags > max_f
        if np.any(mask_f):
            forces[mask_f] = forces[mask_f] / f_mags[mask_f, np.newaxis] * max_f
            
        alpha = 0.01
        new_pts = current_pts + alpha * forces
        
        r_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask_out = r_new > 1.0
        if np.any(mask_out):
            new_pts[mask_out] = new_pts[mask_out] / r_new[mask_out, np.newaxis] * 0.9999
            
        r_curr = compute_radius(current_pts)
        r_new_val = compute_radius(new_pts)
        
        if r_new_val > r_curr:
            current_pts = new_pts
            if r_new_val > best_radius:
                best_radius = r_new_val
                best_pts = new_pts.copy()
        else:
            break # Stagnation in local polish
            
    # Final check
    r_final = compute_radius(current_pts)
    if r_final > best_radius:
        best_radius = r_final
        best_pts = current_pts.copy()

    return best_pts
