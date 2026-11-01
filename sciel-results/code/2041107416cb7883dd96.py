"""Improved program for packing 300 circles in a unit circle using regularized soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    # Estimate spacing. r ~ 1/sqrt(N). Diameter ~ 2r.
    # Hex lattice spacing s = 2r.
    # Area of hexagon per point ~ s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 ~ pi.
    # s ~ sqrt(2*pi / (N*sqrt(3))).
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    # Hex lattice coordinates:
    # x = s * (i + 0.5 * (j % 2))
    # y = s * j * sqrt(3) / 2
    max_k = int(1.0 / (s * 0.5)) + 3
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        # If we didn't generate enough, shrink spacing slightly and retry
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 4
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        # Sort by distance to origin to pick the most central ones
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: random points
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _ring_init(n):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
    points = []
    # Estimate ring structure
    # Number of rings ~ sqrt(n)
    num_rings = int(np.sqrt(n * 2 / np.sqrt(3)))
    
    # Distribute points among rings
    # Ring k has roughly proportional to circumference
    ring_counts = []
    remaining = n
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            # Approximate count based on circumference
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n)))
            count = max(1, count)
        ring_counts.append(count)
    
    # Adjust to match total
    diff = n - sum(ring_counts)
    if diff > 0:
        # Add extra to outer rings
        for i in range(len(ring_counts)-1, -1, -1):
            add = min(diff, max(1, ring_counts[i]//3))
            ring_counts[i] += add
            diff -= add
            if diff == 0:
                break
    elif diff < 0:
        # Remove from outer rings
        for i in range(len(ring_counts)-1, -1, -1):
            rem = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= rem
            diff += rem
            if diff == 0:
                break
    
    # Remove empty rings
    ring_counts = [c for c in ring_counts if c > 0]
    num_rings = len(ring_counts)
    
    # Radius of each ring
    # Total area ~ pi
    # Ring k area ~ 2*pi*r_k*dr
    # For hex packing, dr ~ s = 2r_circle
    # r_k ~ k * s
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    for k, count in enumerate(ring_counts):
        if k == 0:
            points.append([0.0, 0.0])
        else:
            r_k = k * s * 0.95  # Slightly less than full hex spacing
            angles = np.linspace(0, 2*np.pi, count, endpoint=False) + np.pi/count  # Offset
            for angle in angles:
                points.append([r_k * np.cos(angle), r_k * np.sin(angle)])
    
    points = np.array(points[:n])
    
    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization - try multiple initializations and keep the best
    np.random.seed(42)
    
    inits = [
        _hex_lattice_init(N),
        _ring_init(N),
    ]
    
    # Also add a random initialization
    random_points = np.random.rand(N, 2) * 2 - 1
    norms = np.linalg.norm(random_points, axis=1)
    mask = norms > 1.0
    random_points[mask] = random_points[mask] / norms[mask, np.newaxis] * 0.95
    inits.append(random_points)
    
    points = inits[0]
    best_points = points.copy()
    best_score = _compute_score(points)
    
    for init in inits[1:]:
        score = _compute_score(init)
        if score > best_score:
            best_score = score
            best_points = init
            points = init
    
    # 2. Optimization using L-BFGS-B with regularized soft-min objective
    # Objective: minimize -soft_min(constraints) + lambda * RepulsiveEnergy
    # RepulsiveEnergy = sum_{i<j} 1 / (d_ij^2 + epsilon)
    # This term encourages spreading out, acting as a regularizer.
    
    def objective_and_grad(x, beta, lam):
        p = x.reshape(-1, 2)
        
        # Calculate distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        
        # We want the radius r, so the "distance" for pairs is d/2.
        # Let's define the "constraint distances" c_k.
        # For pairs (i,j): c_{ij} = dists[i,j] / 2.0
        # For boundary i: c_i = 1.0 - norms[i]
        
        # We want to maximize min(c_k).
        # Minimize -min(c_k).
        # Smooth approximation: -soft_min(c)
        
        # Flatten distances for vectorization
        # Pair distances: upper triangle
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu] / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Combine all constraint distances
        # Note: pair_dists has N*(N-1)/2 elements, bound_dists has N elements.
        # We need to compute soft_min and its gradient.
        
        # To avoid overflow in exp, we use the log-sum-exp trick.
        # Let z = -beta * c.
        # soft_min(c) = -1/beta * log(sum(exp(-beta*c)))
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        # Concatenate for sum
        # We can compute log_sum_exp separately or combined.
        # Combined is easier for gradient weighting.
        
        # Max values for stability
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        # exp terms
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        # log_sum_exp = max_z + log(sum_exp)
        log_sum_exp = max_z + np.log(sum_exp)
        
        # Objective value: -soft_min = 1/beta * log_sum_exp
        val_softmin = log_sum_exp / beta
        
        # Gradient of soft-min part
        # Weights
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        # Pair contributions to soft-min gradient
        # c_{ij} = 0.5 * |p_i - p_j|
        # grad_{p_i} c_{ij} = 0.5 * (p_i - p_j) / |p_i - p_j|
        # grad_{p_j} c_{ij} = 0.5 * (p_j - p_i) / |p_i - p_j|
        
        # Vectorized pair gradient
        # For each pair (i,j) with weight w_k:
        # grad[p_i] += w_k * 0.5 * (p_i - p_j) / dists[i,j]
        # grad[p_j] -= w_k * 0.5 * (p_j - p_i) / dists[i,j]
        
        # dists[i,j] = 2 * pair_dists[k]
        # So 0.5 / dists[i,j] = 0.5 / (2 * pair_dists[k]) = 0.25 / pair_dists[k]
        
        # Let's compute the vector directions
        # diff[i,j] = p_i - p_j
        # We have iu indices.
        i_idx, j_idx = iu
        
        # Avoid division by zero
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        
        # Coefficient for each pair
        # w_k * 0.25 / pair_dists[k]
        coeff_softmin = w_pairs * 0.25 / safe_pair_dists
        
        # Vector from j to i
        vecs = p[i_idx] - p[j_idx]
        
        # Accumulate gradients for soft-min part
        # grad[p_i] += coeff * vecs
        # grad[p_j] -= coeff * vecs
        
        np.add.at(grad, i_idx, coeff_softmin[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -coeff_softmin[:, np.newaxis] * vecs)
        
        # Boundary contributions to soft-min gradient
        # c_i = 1 - |p_i|
        # grad_{p_i} c_i = - p_i / |p_i|
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = -p / safe_norms[:, np.newaxis]
        
        # grad[p_i] += w_bounds[i] * bound_grad_vecs[i]
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        # The objective is -soft_min + lambda * Repulsive
        # grad(-soft_min) = -grad(soft_min)
        # So we negate the soft-min gradient part
        grad_softmin_negated = -grad
        
        # Now add the repulsive part
        # RepulsiveEnergy = sum_{i<j} 1 / (d_ij^2 + epsilon)
        # Let epsilon = 1e-4 to avoid singularity
        epsilon = 1e-4
        pair_dists_sq = dists_sq[iu]
        repulsive_term = 1.0 / (pair_dists_sq + epsilon)
        
        val_repulsive = np.sum(repulsive_term)
        
        # Gradient of repulsive term
        # d/dp_i (1 / (d_ij^2 + eps)) = -1 / (d_ij^2 + eps)^2 * d/dp_i (d_ij^2)
        # d/dp_i (d_ij^2) = 2 * (p_i - p_j)
        # So grad_{p_i} = -2 * (p_i - p_j) / (d_ij^2 + eps)^2
        # grad_{p_j} = -2 * (p_j - p_i) / (d_ij^2 + eps)^2 = 2 * (p_i - p_j) / (d_ij^2 + eps)^2
        
        repulsive_coeff = -2.0 * lam / (pair_dists_sq + epsilon)**2
        
        # Accumulate repulsive gradients
        # grad[p_i] += repulsive_coeff * (p_i - p_j)
        # grad[p_j] -= repulsive_coeff * (p_i - p_j)
        
        np.add.at(grad, i_idx, repulsive_coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -repulsive_coeff[:, np.newaxis] * vecs)
        
        # Total gradient is grad_softmin_negated + grad_repulsive
        # Wait, my `grad` variable currently holds the gradient of soft_min (before negation).
        # I calculated `grad_softmin_negated` but didn't use it in the accumulation above.
        # Let's restructure:
        # 1. Calculate soft-min gradient contribution
        # 2. Calculate repulsive gradient contribution
        # 3. Sum them up with correct signs.
        
        # Let's redo the gradient accumulation cleanly.
        grad_total = np.zeros_like(p)
        
        # Soft-min part: we want grad of (-soft_min).
        # grad(soft_min) = sum(w_k * grad(c_k)).
        # So grad(-soft_min) = -sum(w_k * grad(c_k)).
        
        # Pair part of soft-min
        np.add.at(grad_total, i_idx, -coeff_softmin[:, np.newaxis] * vecs)
        np.add.at(grad_total, j_idx, coeff_softmin[:, np.newaxis] * vecs)
        
        # Boundary part of soft-min
        grad_total += -w_bounds[:, np.newaxis] * bound_grad_vecs
        
        # Repulsive part: we want grad of (lambda * Repulsive).
        # grad(Repulsive) for pair (i,j):
        # d/dp_i (1/(d^2+eps)) = -2(p_i-p_j)/(d^2+eps)^2
        # d/dp_j (1/(d^2+eps)) = 2(p_i-p_j)/(d^2+eps)^2
        # So we add lambda * these.
        
        np.add.at(grad_total, i_idx, repulsive_coeff[:, np.newaxis] * vecs)
        np.add.at(grad_total, j_idx, -repulsive_coeff[:, np.newaxis] * vecs)
        
        val_total = val_softmin + lam * val_repulsive
        
        return val_total, grad_total

    # Optimization loop with increasing beta and decreasing lambda
    # Start with high lambda (spread out) and low beta (smooth)
    # End with low lambda (focus on packing) and high beta (sharp)
    
    # (beta, lambda) pairs
    schedule = [
        (10.0, 1.0),
        (50.0, 0.5),
        (100.0, 0.2),
        (500.0, 0.1),
        (1000.0, 0.05),
        (3000.0, 0.01),
        (5000.0, 0.0),
        (10000.0, 0.0)
    ]
    
    current_points = points.copy()
    
    for beta, lam in schedule:
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta, lam)[0],
                current_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta, lam)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'gtol': 1e-9,
                    'ftol': 1e-11,
                    'maxls': 25,
                    'disp': False
                }
            )
            
            if time.time() > deadline - 8:
                break
                
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
            else:
                # If it didn't improve, keep current best for next step
                current_points = best_points
                
        except Exception:
            pass

    # Final polish with very high beta if time permits
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: objective_and_grad(x, 5000.0, 0.0)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, 5000.0, 0.0)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-11,
                    'ftol': 1e-13,
                    'maxls': 15,
                    'disp': False
                }
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
        except Exception:
            pass

    # Second polish pass with even higher beta and tighter tolerance
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: objective_and_grad(x, 10000.0, 0.0)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, 10000.0, 0.0)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 150,
                    'gtol': 1e-12,
                    'ftol': 1e-14,
                    'maxls': 20,
                    'disp': False
                }
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
        except Exception:
            pass

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        # Recalculate score
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
