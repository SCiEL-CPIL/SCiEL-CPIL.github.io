import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit

    def elapsed():
        return time.time() - start_time

    # 1. Initialization: Hexagonal Lattice
    # Create a dense hexagonal lattice and select N points closest to the center
    # to provide a good initial configuration.
    def hex_init():
        # Estimate spacing. Area per circle ~ pi*r^2. 
        # For N=300 in unit circle, r ~ 0.05. Area ~ 0.008.
        # Hex spacing s: area per hex ~ s^2 * sqrt(3)/2. 
        # s ~ sqrt(2 * 0.008 * 2 / 1.732) ~ 0.13.
        # Let's just generate a large grid and filter.
        spacing = 0.1
        max_coord = 2.0
        xs = np.arange(-max_coord, max_coord + spacing, spacing)
        ys = np.arange(-max_coord, max_coord + spacing, spacing)
        
        # Hexagonal offset
        points = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                offset = (spacing * np.sqrt(3) / 2) * (i % 2)
                points.append((x, y + offset))
        
        points = np.array(points)
        
        # Keep points inside radius 1 - spacing/2 to ensure they are valid initially
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii < 1.0 - spacing / 2
        points = points[mask]
        
        # Select N points with smallest radii (closest to center)
        # This tends to fill the center first, creating a solid core.
        indices = np.argsort(radii[mask])[:N]
        # Note: radii was computed for all points, so we need to re-index or 
        # compute radii for the filtered set.
        
        # Let's redo cleanly
        radii_all = np.hypot(points[:, 0], points[:, 1])
        # We already filtered, so points are the valid ones.
        # We need N points from the filtered set.
        if len(points) < N:
             # Fallback: random
             return np.random.uniform(-0.9, 0.9, (N, 2))
             
        # Sort by distance to origin
        order = np.argsort(radii_all)
        p_init = points[order[:N]].copy()
        return p_init

    p = hex_init()
    
    # 2. Objective Function: Soft Min of Clearances
    # We want to maximize min( min_{i<j} d_ij/2, min_i (1 - |p_i|) )
    # Let c_k be the clearances. We maximize min(c_k).
    # Equivalent to minimizing -min(c_k).
    # Soft min: F(x) = -1/beta * log( sum( exp(-beta * c_k) ) )
    
    def compute_clearances(p):
        # Pairwise distances
        # Using broadcasting for N=300 is 300x300x2, which is ~ 720k floats. Manageable.
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Pair clearances: d/2
        c_pair = dist / 2.0
        
        # Boundary clearances
        r_norm = np.linalg.norm(p, axis=1)
        c_bound = 1.0 - r_norm
        
        # Combine
        # We only need upper triangle for pairs to avoid double counting in the sum,
        # but for the minimum, duplicates don't change the value.
        # For the soft min sum, duplicates change the weight. 
        # To be consistent, we use unique pairs.
        i_idx, j_idx = np.triu_indices(N, k=1)
        c_pairs = c_pair[i_idx, j_idx]
        
        all_c = np.concatenate([c_bound, c_pairs])
        return all_c, c_pair, c_bound, i_idx, j_idx

    def soft_min_obj_and_grad(p_flat, beta):
        p = p_flat.reshape(N, 2)
        all_c, c_pair, c_bound, i_idx, j_idx = compute_clearances(p)
        
        # Soft min calculation
        # To avoid overflow, shift by max
        max_c = np.max(all_c)
        exp_terms = np.exp(-beta * (all_c - max_c))
        sum_exp = np.sum(exp_terms)
        
        # Objective value: -1/beta * log(sum_exp) - max_c (shifted back)
        # Wait, the standard log-sum-exp for soft-min is:
        # -1/beta * log( sum( exp(-beta * c_i) ) )
        # Let c'_i = c_i - max_c. Then c_i = c'_i + max_c.
        # sum exp(-beta(c'_i + max_c)) = exp(-beta max_c) * sum exp(-beta c'_i)
        # log(sum) = -beta max_c + log(sum exp(-beta c'_i))
        # obj = -1/beta * (-beta max_c + log(...)) = max_c - 1/beta * log(sum exp(-beta c'_i))
        # This is a "soft max" of -c? No.
        # Let's stick to the formula:
        obj = -1.0 / beta * np.log(sum_exp) - max_c # Wait, sign check.
        # If beta -> inf, obj -> -max_c? No, we want to MINIMIZE a function that is -min(c).
        # So we want f = -min(c).
        # Soft min of c is approx min(c).
        # We want to maximize min(c) => minimize -min(c).
        # So objective function to minimize is F = -soft_min(c).
        # soft_min(c) = -1/beta * log(sum exp(-beta c)).
        # So F = 1/beta * log(sum exp(-beta c)).
        
        # Let's re-evaluate.
        # We want to MAXIMIZE the minimum clearance.
        # Let r_min = min(c).
        # We can minimize J(r) = -r_min.
        # Approximation: J = 1/beta * log( sum(exp(-beta * c)) )
        
        # Gradient of J w.r.t p:
        # dJ/dc_k = -exp(-beta c_k) / sum_exp
        # dJ/dp = sum_k (dJ/dc_k) * (dc_k/dp)
        
        # Let's use this formulation.
        # J = 1/beta * log(sum_exp) - max_c  (using the shifted sum_exp from above? No, sum_exp above was for shifted c)
        # Let S = sum(exp(-beta * (c - max_c)))
        # log(sum(exp(-beta c))) = log(exp(-beta max_c) * S) = -beta max_c + log(S)
        # J = 1/beta * (-beta max_c + log(S)) = -max_c + 1/beta * log(S)
        
        obj_val = -max_c + 1.0 / beta * np.log(sum_exp)
        
        # Weights for gradient
        # w_k = dJ/dc_k = -exp(-beta c_k) / sum_exp * (1) ?
        # d/dc_k log(sum exp(-beta c)) = -beta exp(-beta c_k) / sum_exp
        # dJ/dc_k = 1/beta * (-beta exp(-beta c_k) / sum_exp) = -exp(-beta c_k) / sum_exp
        
        weights = -exp_terms / sum_exp
        
        grad = np.zeros_like(p)
        
        # Boundary terms
        # c_i = 1 - |p_i|
        # d c_i / d p_i = - p_i / |p_i|
        # Contribution to grad: weights_bound[i] * (-p_i / |p_i|)
        
        r_norm = np.linalg.norm(p, axis=1)
        r_norm_safe = np.where(r_norm > 1e-8, r_norm, 1.0)
        grad_bound = -p / r_norm_safe[:, np.newaxis]
        grad_bound *= np.where(r_norm > 1e-8, 1.0, 0.0)[:, np.newaxis]
        
        grad += weights[:N] * grad_bound
        
        # Pair terms
        # c_ij = 0.5 * |p_i - p_j|
        # d c_ij / d p_i = 0.5 * (p_i - p_j) / |p_i - p_j|
        # d c_ij / d p_j = 0.5 * (p_j - p_i) / |p_i - p_j|
        
        diff_pairs = p[i_idx] - p[j_idx]
        dist_pairs = c_pair[i_idx, j_idx] # This is d/2? No, c_pair was dist/2.
        # In compute_clearances, c_pair = dist / 2.0.
        # So dist_pairs in code is actually the clearance c_ij.
        # The distance is 2 * c_ij.
        # unit_vec = (p_i - p_j) / |p_i - p_j| = diff_pairs / (2 * c_ij)
        
        dist_pairs_safe = np.where(dist_pairs > 1e-8, dist_pairs, 1.0)
        # d c_ij / d p_i = 0.5 * (p_i - p_j) / (2 * c_ij) * 2? 
        # Let d = |p_i - p_j|. c = d/2.
        # dc/dp_i = 1/2 * dd/dp_i = 1/2 * (p_i - p_j)/d = (p_i - p_j) / (4 c_ij)? 
        # No. d = |p_i - p_j|. c = 0.5 d.
        # dc/dp_i = 0.5 * (p_i - p_j)/d.
        # So unit direction is (p_i - p_j)/d.
        # Coefficient is 0.5.
        
        unit_vecs = diff_pairs / (2.0 * dist_pairs_safe)[:, np.newaxis] # This is (p_i - p_j) / d
        
        # grad_p_i += w_ij * 0.5 * unit_vec
        # grad_p_j += w_ij * 0.5 * (-unit_vec)
        
        w_pairs = weights[N:] * 0.5
        
        np.add.at(grad, i_idx, w_pairs * unit_vecs)
        np.add.at(grad, j_idx, -w_pairs * unit_vecs)
        
        return obj_val, grad

    def calc_score(p):
        all_c, _, _, _, _ = compute_clearances(p)
        return np.min(all_c)

    best_p = p.copy()
    best_score = calc_score(best_p)
    
    # 3. Optimization Loop
    # Beta schedule: start low (smooth), increase (sharp)
    betas = [10, 50, 200, 1000, 5000]
    
    for beta in betas:
        if elapsed() > deadline - 5:
            break
            
        p_flat = best_p.flatten()
        
        try:
            res = minimize(
                lambda x: soft_min_obj_and_grad(x, beta)[0],
                p_flat,
                jac=lambda x: soft_min_obj_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_p = res.x.reshape(N, 2)
            new_score = calc_score(new_p)
            
            if new_score > best_score:
                best_score = new_score
                best_p = new_p.copy()
                
        except Exception:
            pass

    # 4. Local Polish Phase
    # Identify active constraints and push them apart slightly to increase min radius.
    # This is a heuristic refinement.
    
    current_p = best_p.copy()
    current_score = best_score
    max_polish_time = 20.0
    
    polish_start = time.time()
    
    for iter in range(50):
        if time.time() - polish_start > max_polish_time:
            break
        if elapsed() > deadline - 2:
            break
            
        all_c, c_pair, c_bound, i_idx, j_idx = compute_clearances(current_p)
        r_min = np.min(all_c)
        
        # Identify "tight" constraints: clearance < r_min + delta
        delta = 0.0001 + 0.001 * (1 - iter / 50.0) # Decreasing delta
        
        tight_pairs = c_pair[i_idx, j_idx] < (r_min + delta)
        tight_boundary = c_bound < (r_min + delta)
        
        if not np.any(tight_pairs) and not np.any(tight_boundary):
            break
            
        forces = np.zeros_like(current_p)
        
        # Pair forces: push apart
        if np.any(tight_pairs):
            ii = i_idx[tight_pairs]
            jj = j_idx[tight_pairs]
            dists = c_pair[ii, jj] # clearances
            # Force magnitude proportional to how tight? Or constant?
            # Let's use a constant push for simplicity, scaled by step size.
            diff = current_p[ii] - current_p[jj]
            dists_actual = np.linalg.norm(diff, axis=1)
            # Normalize
            norms = np.where(dists_actual > 1e-8, dists_actual, 1.0)
            unit_vecs = diff / norms[:, np.newaxis]
            
            # Add force to i, subtract from j
            # Use small step
            step = 0.001
            np.add.at(forces, ii, step * unit_vecs)
            np.add.at(forces, jj, -step * unit_vecs)
            
        # Boundary forces: push inwards
        if np.any(tight_boundary):
            tight_indices = np.where(tight_boundary)[0]
            radii = np.linalg.norm(current_p[tight_indices], axis=1)
            norms = np.where(radii > 1e-8, radii, 1.0)
            unit_vecs = -current_p[tight_indices] / norms[:, np.newaxis] # Inward
            
            step = 0.001
            forces[tight_indices] += step * unit_vecs
            
        # Apply forces
        new_p = current_p + forces
        
        # Project back to ensure no point is outside (safety)
        radii_new = np.linalg.norm(new_p, axis=1)
        out_mask = radii_new > 1.0
        if np.any(out_mask):
            new_p[out_mask] *= (1.0 / radii_new[out_mask])[:, np.newaxis]
            
        new_score = calc_score(new_p)
        
        if new_score > current_score:
            current_score = new_score
            current_p = new_p
        else:
            # If it got worse, stop or reduce step? 
            # For simplicity, if it doesn't improve, we might be at a local max for this heuristic.
            # We can break or continue with smaller step.
            pass 

    if current_score > best_score:
        best_score = current_score
        best_p = current_p

    return best_p
