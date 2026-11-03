"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the score for a given set of points."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check container constraint
    dists_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_origin > 1.0):
        return 0.0
    
    # Compute pairwise distances
    # To save memory/time, we can do it in chunks or just use broadcasting for N=50
    # N=50 is small enough for O(N^2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_boundary_dist = 1.0 - np.max(dists_origin)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _init_hexagonal():
    """Initialize points on a hexagonal lattice."""
    # Approximate radius for N=50 is ~0.125
    # Area of circle = pi. Area of N circles = N * pi * r^2.
    # r ~ sqrt(1/N) ~ 0.14.
    # Let's use a spacing that allows ~50 points.
    spacing = 0.25 
    points = []
    # Generate grid
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if np.sqrt(x*x + y*y) < 0.8: # Keep well inside initially
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        # Fallback to random if lattice fails (should not happen)
        return np.random.uniform(-0.5, 0.5, (N, 2))
    
    # Select N points closest to origin to start compact
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:N]
    return points[idx]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    points = _init_hexagonal()
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Langevin Dynamics Parameters
    dt = 0.01
    gamma = 0.5 # Friction
    kT = 0.05   # Initial Temperature (Noise)
    beta = 10.0 # Initial inverse temperature for smooth min
    
    # Annealing schedule
    # We want to increase beta to sharpen the objective
    # and decrease kT to reduce noise
    
    num_steps = 20000
    step = 0
    
    while step < num_steps:
        if time.time() > deadline:
            break
        
        # Compute smooth objective gradient
        # Objective: Minimize E = (1/beta) * log( sum_{i<j} exp(-beta * d_ij) + sum_i exp(-beta * (1-|c_i|)) )
        # Actually, we want to MAXIMIZE the min distance.
        # Let d_min be the limiting distance.
        # We can use a soft-min: S = -1/beta * log( sum exp(-beta * d) )
        # We want to maximize S. So minimize -S = 1/beta * log( sum exp(-beta * d) )
        
        # Distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        dists_origin = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - dists_origin
        
        # We combine pairwise and boundary distances into a single list of "constraints"
        # We want the minimum of all these to be large.
        # Let X be the vector of all d_ij and (1-|c_i|).
        # We want to maximize min(X).
        # Surrogate: F(X) = -1/beta * log( sum_{x in X} exp(-beta * x) )
        # Gradient of F w.r.t points:
        # dF/dx_k = -1/beta * ( -beta * exp(-beta * x_k) * sum_exp(-beta * x) ) / sum_exp(-beta * x)
        # = exp(-beta * x_k) / sum_exp(-beta * x)
        # So the force on a point is proportional to the "weight" of its constraints.
        
        # To avoid overflow in exp, subtract max
        all_dists = np.concatenate([dists[np.triu_indices(N, 1)], boundary_dists])
        max_d = np.max(all_dists)
        exps = np.exp(-beta * (all_dists - max_d))
        sum_exps = np.sum(exps)
        weights = exps / sum_exps
        
        # Reconstruct gradients
        # Gradient of the energy E = 1/beta * log(sum exp) w.r.t positions
        # Note: The derivative of log-sum-exp is the softmax weights.
        # dE/dx_k = weights[k]
        # We want to move points to INCREASE x_k (distances).
        # So we move in the direction of +gradient of x_k.
        
        forces = np.zeros_like(points)
        
        # Pairwise part
        # weights[0 : N*(N-1)/2] correspond to pairs (i, j) with i < j
        num_pairs = N * (N - 1) // 2
        pair_weights = weights[:num_pairs]
        
        # Map pair indices to i, j
        # This is slow in pure Python. Vectorize or use precomputed indices.
        # For N=50, num_pairs = 1225. Loop is okay but vectorized is better.
        # Let's precompute indices outside the loop? No, points change.
        # But indices (i,j) are fixed.
        
        # Precompute i, j indices for pairs
        i_idx, j_idx = np.triu_indices(N, 1)
        
        # Vectorized force accumulation for pairs
        # Force on i is +w * (r_i - r_j) / d_ij
        # Force on j is -w * (r_i - r_j) / d_ij
        # Wait, d/dx_i (d_ij) = (x_i - x_j) / d_ij.
        # We want to maximize d_ij. So we move x_i in direction (x_i - x_j) / d_ij.
        
        # d_ij matrix for pairs
        pair_diff = points[i_idx] - points[j_idx]
        pair_dists = dists[i_idx, j_idx]
        # Avoid division by zero
        pair_dists_safe = np.where(pair_dists < 1e-8, 1e-8, pair_dists)
        
        # Unit vectors
        unit_vecs = pair_diff / pair_dists_safe[:, np.newaxis]
        
        # Weighted unit vectors
        weighted_vecs = pair_weights[:, np.newaxis] * unit_vecs
        
        # Accumulate forces
        np.add.at(forces, i_idx, weighted_vecs)
        np.add.at(forces, j_idx, -weighted_vecs)
        
        # Boundary part
        # weights[num_pairs : ] correspond to points i
        boundary_weights = weights[num_pairs:]
        # Force on i is +w * (c_i / |c_i|) ? 
        # We want to maximize (1 - |c_i|).
        # d(1-|c_i|)/d c_i = - c_i / |c_i|.
        # So we want to move in direction - c_i / |c_i| (towards origin).
        # Wait, the gradient of the objective w.r.t c_i is -c_i/|c_i|.
        # We are minimizing E. So we move in direction - grad E.
        # grad E for boundary term is proportional to weight * (grad of (1-|c_i|))?
        # Let's check signs.
        # E = 1/beta log sum exp(-beta x).
        # dE/dx = exp(-beta x) / sum. (Positive)
        # x = 1 - |c|. dx/dc = -c/|c|.
        # dE/dc = (dE/dx) * (dx/dc) = w * (-c/|c|).
        # We want to minimize E. So step = -dt * dE/dc = dt * w * c/|c|.
        # This moves the point OUTWARDS?
        # If c is small, |c| is small. c/|c| is direction away from origin.
        # This would push points away from origin, increasing |c|, decreasing x.
        # This is WRONG. We want to INCREASE x (distance to boundary).
        # x = 1 - |c|. To increase x, we must decrease |c|.
        # So we must move TOWARDS origin.
        # Direction: -c/|c|.
        
        # Let's re-evaluate the gradient direction.
        # We want to MAXIMIZE min(x).
        # We are MINIMIZING E(x) which approximates -min(x).
        # So we should move in direction - grad E.
        # For pair: x = d. dx/dc_i = (c_i-c_j)/d.
        # dE/dc_i = w * (c_i-c_j)/d.
        # Step = -dt * w * (c_i-c_j)/d.
        # This moves c_i towards c_j?
        # If c_i is to the right of c_j, c_i-c_j is positive. Step is negative. Moves left. Towards c_j.
        # This REDUCES distance.
        # We want to INCREASE distance.
        # So we should move in direction + (c_i-c_j)/d.
        # This implies we should MAXIMIZE E? No.
        
        # Let's look at the function again.
        # We want to maximize M = min(x).
        # Softmax approximation for min: min(x) ~ -1/beta log sum exp(-beta x).
        # Let G(x) = -1/beta log sum exp(-beta x). We want to MAXIMIZE G.
        # So we do Gradient Ascent on G.
        # dG/dx = exp(-beta x) / sum exp(-beta x) = w. (Positive)
        # So we move in direction + grad G.
        # For pair: dG/dc_i = w * (c_i-c_j)/d.
        # Step = +dt * w * (c_i-c_j)/d.
        # This moves c_i away from c_j. Correct.
        
        # For boundary: x = 1 - |c|.
        # dG/dc = w * d(1-|c|)/dc = w * (-c/|c|).
        # Step = +dt * w * (-c/|c|).
        # This moves c towards origin. Correct.
        
        # So the forces calculated above for pairs were correct (add.at with +w*unit for i, -w*unit for j).
        # Wait, my previous accumulation:
        # np.add.at(forces, i_idx, weighted_vecs) -> forces[i] += w * (c_i-c_j)/d
        # np.add.at(forces, j_idx, -weighted_vecs) -> forces[j] += w * (c_j-c_i)/d
        # This is correct for Gradient Ascent on G.
        
        # Now boundary forces:
        dists_origin_safe = np.where(dists_origin < 1e-8, 1e-8, dists_origin)
        unit_vectors_origin = points / dists_origin_safe[:, np.newaxis]
        # Force = - w * unit_vector (towards origin)
        forces += -boundary_weights[:, np.newaxis] * unit_vectors_origin
        
        # Langevin Update
        # v = (1-gamma) v + dt * F + noise
        # We are doing position-based Langevin (no explicit velocity storage, effectively implicit velocity)
        # x_new = x + dt * F + sqrt(2*kT*dt) * noise
        
        noise = np.random.normal(0, 1, size=points.shape)
        noise_scale = np.sqrt(2 * kT * dt)
        
        points = points + dt * forces + noise_scale * noise
        
        # Hard constraint: Keep inside circle
        # If a point goes outside, project it back.
        # This is a non-smooth operation but necessary for stability.
        new_dists = np.linalg.norm(points, axis=1)
        mask = new_dists > 1.0
        if np.any(mask):
            points[mask] = points[mask] / new_dists[mask, np.newaxis]
            
        # Annealing
        # Increase beta (sharpen min), decrease kT (reduce noise)
        beta *= 1.01
        kT *= 0.99
        
        # Occasionally, if stuck, we might want to restart or perturb, 
        # but annealing usually handles this.
        
        # Compute score periodically
        if step % 100 == 0:
            current_score = _compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                
        step += 1

    # Final polish: Local search on best_points
    # Try to improve best_points with a few iterations of gradient ascent with very low noise
    p = best_points.copy()
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        dists_origin = np.linalg.norm(p, axis=1)
        boundary_dists = 1.0 - dists_origin
        all_dists = np.concatenate([dists[np.triu_indices(N, 1)], boundary_dists])
        
        # Use a very high beta for final polish
        beta_final = 100.0
        max_d = np.max(all_dists)
        exps = np.exp(-beta_final * (all_dists - max_d))
        sum_exps = np.sum(exps)
        weights = exps / sum_exps
        
        forces = np.zeros_like(p)
        i_idx, j_idx = np.triu_indices(N, 1)
        pair_weights = weights[:num_pairs]
        pair_diff = p[i_idx] - p[j_idx]
        pair_dists = dists[i_idx, j_idx]
        pair_dists_safe = np.where(pair_dists < 1e-8, 1e-8, pair_dists)
        unit_vecs = pair_diff / pair_dists_safe[:, np.newaxis]
        weighted_vecs = pair_weights[:, np.newaxis] * unit_vecs
        np.add.at(forces, i_idx, weighted_vecs)
        np.add.at(forces, j_idx, -weighted_vecs)
        
        boundary_weights = weights[num_pairs:]
        dists_origin_safe = np.where(dists_origin < 1e-8, 1e-8, dists_origin)
        unit_vectors_origin = p / dists_origin_safe[:, np.newaxis]
        forces += -boundary_weights[:, np.newaxis] * unit_vectors_origin
        
        # Small step
        p = p + 0.001 * forces
        
        # Project
        new_dists = np.linalg.norm(p, axis=1)
        mask = new_dists > 1.0
        if np.any(mask):
            p[mask] = p[mask] / new_dists[mask, np.newaxis]
            
        # Check for overlaps? If overlaps occur, we might have to back off.
        # The soft potential should prevent this, but hard projection might cause issues.
        # Let's just take the best score found.
        
        final_score = _compute_score(p)
        if final_score > best_score:
            best_score = final_score
            best_points = p.copy()

    return best_points

# EVOLVE-BLOCK-END
