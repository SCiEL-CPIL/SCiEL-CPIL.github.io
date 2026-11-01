"""
Combined solution for 50 equal circles in a unit circle.
Strategy:
1. Diverse initializations (Hex, Ring, Random).
2. Vectorized Soft-Min L-BFGS-B optimization with increasing beta.
3. Vectorized Force-Directed Polish for final precision.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin for 120s limit

# Pre-compute indices for pairwise distances
_I_IDX, _J_IDX = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_I_IDX)

def _compute_score(centers):
    """Compute the packing score for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if len(centers) != N:
        return 0.0
    
    # Boundary check
    norms = np.linalg.norm(centers, axis=1)
    if np.any(norms > 1.0 + 1e-9):
        return 0.0
    min_bound = np.min(1.0 - norms)
    
    # Pairwise check
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_bound)

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    best_pts = None
    best_score = -1.0

    for spacing in [0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]:
        k = int(1.0 / spacing) + 4
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])

        mask = radius <= 0.95
        points = points[mask]
        radius = radius[mask]

        if len(points) >= n:
            idx = np.argsort(radius, kind="stable")[:n]
            pts = points[idx]
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts

    if best_pts is not None:
        return best_pts
    
    return _get_random_initialization(n, seed=42)

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = np.sqrt(rng.uniform(0, 0.9)) # Uniform in area
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _get_ring_initialization(n):
    """Generate ring-based initialization."""
    centers = []
    centers.append([0.0, 0.0])
    
    # Approximate distribution: 1, 6, 12, 12, 19? Total 50.
    rings = [
        (6, 0.25),
        (12, 0.5),
        (12, 0.75),
        (19, 0.9)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
        if count >= n:
            break
            
    return np.array(centers[:n])

def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise/2 and boundary).
    """
    pts = flat.reshape(N, 2)

    # Boundary distances: 1 - ||p_i||
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin

    # Pairwise distances: ||p_i - p_j|| / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # Extract upper triangle
    pair_dists = dist_mat[_I_IDX, _J_IDX] / 2.0

    all_dists = np.concatenate([boundary_dists, pair_dists])

    # Soft min: -min(x) approx by logsumexp(-beta * x) / beta
    # To avoid overflow, we shift by max
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    
    if sum_exp == 0:
        return 0.0, np.zeros_like(flat)

    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta

    # Weights for gradient
    weights = np.exp(y_shifted) / sum_exp

    grad_pts = np.zeros_like(pts)

    # Gradient contribution from boundary constraints
    # d/dp (1 - ||p||) = -p / ||p||
    # d/dp (soft_min) = sum( w_i * grad_i )
    # Here obj is min, so grad of obj is sum(w * grad(dist))
    # We want to MINIMIZE obj. So we return gradient of obj.
    # grad(obj) = sum( w_i * grad(x_i) )
    
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    # grad(boundary_dist) = -pts / dists
    boundary_grad = (-w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad

    # Gradient contribution from pair constraints
    # x_k = ||p_i - p_j|| / 2
    # d/dp_i x_k = (p_i - p_j) / (2 * ||p_i - p_j||)
    # d/dp_j x_k = (p_j - p_i) / (2 * ||p_i - p_j||)
    
    w_pairs = weights[N:]
    i_idx = _I_IDX
    j_idx = _J_IDX
    d_pairs = dist_mat[i_idx, j_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    
    vecs = pts[i_idx] - pts[j_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    # Gradient w.r.t p_i is 0.5 * w * unit_vec
    # Gradient w.r.t p_j is -0.5 * w * unit_vec
    grad_contrib_i = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_j = -0.5 * w_pairs[:, np.newaxis] * unit_vecs

    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, j_idx, grad_contrib_j)

    grad_flat = grad_pts.flatten()

    return obj_val, grad_flat

def _force_directed_polish(pts, r_target, steps=100, lr=0.005):
    """
    Vectorized force-directed method to fix violations and polish.
    Pushes overlapping circles apart and boundary circles inward.
    """
    pts = pts.copy()
    
    for step in range(steps):
        if time.time() > DEADLINE - 2.0:
            break
            
        dists_origin = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - dists_origin
        
        # Pairwise forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        
        force = np.zeros_like(pts)
        
        d_pairs = dist_mat[_I_IDX, _J_IDX]
        
        # Mask pairs that are too close (< 2*r_target)
        # We want to push them apart.
        mask_pair = (d_pairs < 2 * r_target) & (d_pairs > 1e-10)
        if np.any(mask_pair):
            ii = _I_IDX[mask_pair]
            jj = _J_IDX[mask_pair]
            d = d_pairs[mask_pair]
            vecs = pts[ii] - pts[jj]
            units = vecs / d[:, np.newaxis]
            # Strength proportional to violation
            strengths = (2 * r_target - d) * 0.5
            np.add.at(force, ii, units * strengths[:, np.newaxis])
            np.add.at(force, jj, -units * strengths[:, np.newaxis])
            
        # Boundary forces
        mask_bnd = boundary_dists < r_target
        if np.any(mask_bnd):
            bi = np.where(mask_bnd)[0]
            safe_dists = np.where(dists_origin[bi] < 1e-8, 1.0, dists_origin[bi])
            unit_out = pts[bi] / safe_dists[:, np.newaxis]
            strengths = r_target - boundary_dists[bi]
            force[bi] -= unit_out * strengths[:, np.newaxis]
            
        pts += lr * force
        
        # Project back into unit disk (hard constraint)
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0 - r_target
        if np.any(mask):
            pts[mask] = pts[mask] * (1.0 - r_target) / dists[mask, np.newaxis]
            
    return pts

def _optimize_single(init_pts, start_time):
    """
    Optimize a single initialization using L-BFGS-B with soft-min schedule.
    """
    current = init_pts.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    # Schedule of betas: increasing sharpness
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    for beta in betas:
        if time.time() - start_time > DEADLINE - 10.0:
            break
            
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            new_pts = res.x.reshape(N, 2)
            
            # Ensure validity
            if not np.all(np.isfinite(new_pts)):
                continue
                
            # Clip to unit disk
            norms = np.linalg.norm(new_pts, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis]
                
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_pts.copy()
                current = new_pts.copy()
            else:
                # If no improvement, keep current to avoid degradation
                pass
                
        except Exception:
            pass
            
    # Final polish with force-directed method
    if time.time() - start_time < DEADLINE - 5.0:
        # Estimate current r
        r_est = best_score
        polished = _force_directed_polish(best_centers, r_est, steps=50, lr=0.005)
        if np.all(np.isfinite(polished)):
            pol_score = _compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_centers = polished.copy()
                
    return best_centers, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=777),
    ]
    
    # Optimize each initialization
    for init in initializations:
        if time.time() - start_time > DEADLINE - 20.0:
            break
            
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        optimized, score = _optimize_single(init, start_time)
        
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
            
    if best_centers is None:
        return initializations[0]
        
    # Global refinement: Try to improve the best found so far
    # Use a higher beta schedule on the best solution
    if time.time() - start_time < DEADLINE - 10.0:
        refined, refined_score = _optimize_single(best_centers, start_time)
        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined.copy()
            
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 2.0:
        final_polished = _force_directed_polish(best_centers, best_score, steps=100, lr=0.002)
        if np.all(np.isfinite(final_polished)):
            final_score = _compute_score(final_polished)
            if final_score > best_score:
                best_score = final_score
                best_centers = final_polished.copy()
                
    return best_centers
