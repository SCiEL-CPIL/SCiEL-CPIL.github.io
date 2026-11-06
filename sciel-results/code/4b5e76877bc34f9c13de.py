"""Improved packing via soft-min gradient ascent + Nelder-Mead polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 120s limit


def _exact_radius(centers):
    """Compute the exact minimum radius (score) for a set of centers."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    if n == 0:
        return 0.0
    # Boundary distances
    boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = float(np.min(boundary))
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    return min(min_boundary, min_pair / 2.0)


def _soft_min_radius_and_grad(centers, beta):
    """
    Compute a smooth surrogate for the minimum radius using log-sum-exp,
    and its gradient. Higher beta -> closer to true min.
    
    Returns (surrogate_value, gradient) where surrogate approximates min radius.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    
    # Boundary terms: distance to each face
    # For each center i, dist to boundary = min over 6 faces
    # We'll compute soft-min over all boundary distances and pairwise distances
    
    # Collect all "constraint distances" that define the radius:
    # r_i_boundary = 0.5 - |c_ik| for each coordinate k
    # r_ij = |c_i - c_j| / 2 for each pair
    
    # For efficiency, we compute the log-sum-exp of -distances (so soft-min = -soft-max of -dist)
    # Actually: soft_min(d_1,...,d_m) = -logsumexp(-beta * d_i) / beta
    
    # Let's compute all distances
    # Boundary distances for each sphere: 6 values each
    abs_c = np.abs(centers)
    # Shape: (n, 3) -> for each sphere, dist to each face
    # Face distances: 0.5 - abs_c[:,k] for k in 0,1,2
    boundary_dists = np.concatenate([
        0.5 - abs_c[:, 0:1],
        0.5 - abs_c[:, 1:2],
        0.5 - abs_c[:, 2:3]
    ], axis=1)  # shape (n, 6)
    
    # Pairwise distances / 2
    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            pair_dists.append(d)
    pair_dists = np.array(pair_dists)
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists.flatten(), pair_dists])
    
    # Soft-min: -logsumexp(-beta * d) / beta
    beta_d = beta * all_dists
    # Clip for numerical stability
    beta_d_clipped = np.clip(beta_d, -500, 500)
    max_val = np.max(beta_d_clipped)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_d_clipped - max_val)))
    soft_min_val = -logsumexp / beta
    
    # Gradient with respect to centers
    # d(soft_min)/d(centers) = -1/beta * d(logsumexp)/d(centers)
    # d(logsumexp)/d(d_k) = exp(beta*d_k - logsumexp) / beta ... wait let me redo
    
    # soft_min = -(1/beta) * logsumexp(-beta * d)
    # Let L = logsumexp(-beta * d)
    # d(soft_min)/d(centers) = -(1/beta) * dL/d(centers)
    # dL/d(d_k) = exp(-beta*d_k - L) * (-beta) = -beta * softmax_k
    # where softmax_k = exp(-beta*d_k - L)
    # dL/d(centers) = sum_k dL/d(d_k) * d(d_k)/d(centers)
    
    # Compute softmax weights
    neg_beta_d = -beta * all_dists
    neg_beta_d_clipped = np.clip(neg_beta_d, -500, 500)
    max_nb = np.max(neg_beta_d_clipped)
    exp_vals = np.exp(neg_beta_d_clipped - max_nb)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax over all distances
    
    # Now compute gradient
    grad = np.zeros_like(centers)
    
    # Boundary contributions
    # boundary_dists[i, 0] = 0.5 - abs(centers[i,0])
    # d/d(centers[i,0]) = -sign(centers[i,0])
    # boundary_dists[i, 1] = 0.5 - abs(centers[i,1])
    # boundary_dists[i, 2] = 0.5 - abs(centers[i,2])
    # boundary_dists[i, 3] = 0.5 - abs(centers[i,0])  (other side)
    # etc.
    
    # For sphere i, coordinate k:
    # Two boundary distances: 0.5 - centers[i,k] and 0.5 + centers[i,k]
    # d(0.5 - c)/dc = -1, d(0.5 + c)/dc = 1
    
    # boundary_dists column 0: 0.5 - abs(c[:,0]) -> this is min(0.5-c0, 0.5+c0) but we used abs
    # Actually I defined it as 0.5 - abs_c, which is the distance to the nearest face on that axis
    # But that's not differentiable at 0. Let me use the two separate face distances instead.
    
    # Let me restructure: use 6 separate face distances per sphere
    # Face -x: 0.5 + centers[i,0], d/dc = 1
    # Face +x: 0.5 - centers[i,0], d/dc = -1
    # Similarly for y, z
    
    # I need to recompute. Let me just do it more carefully.
    
    # Recompute boundary distances as 6 separate values
    bx_neg = 0.5 + centers[:, 0]  # d/dx = 1
    bx_pos = 0.5 - centers[:, 0]  # d/dx = -1
    by_neg = 0.5 + centers[:, 1]  # d/dy = 1
    by_pos = 0.5 - centers[:, 1]  # d/dy = -1
    bz_neg = 0.5 + centers[:, 2]  # d/dz = 1
    bz_pos = 0.5 - centers[:, 2]  # d/dz = -1
    
    # Rebuild all_dists with these
    all_dists2 = np.concatenate([
        bx_neg, bx_pos, by_neg, by_pos, bz_neg, bz_pos, pair_dists
    ])
    
    # Recompute weights
    neg_beta_d2 = -beta * all_dists2
    neg_beta_d2_clipped = np.clip(neg_beta_d2, -500, 500)
    max_nb2 = np.max(neg_beta_d2_clipped)
    exp_vals2 = np.exp(neg_beta_d2_clipped - max_nb2)
    sum_exp2 = np.sum(exp_vals2)
    weights2 = exp_vals2 / sum_exp2
    
    # Soft min value
    L2 = max_nb2 + np.log(sum_exp2)
    soft_min_val2 = -L2 / beta
    
    # Gradient: d(soft_min)/d(centers) = -(1/beta) * sum_k w_k * (-beta) * d(d_k)/d(centers)
    # = sum_k w_k * d(d_k)/d(centers)
    
    grad = np.zeros_like(centers)
    
    # Boundary contributions
    # bx_neg: weight w[0*n+i], d/dc = +1 for x
    grad[:, 0] += weights2[0*n:1*n] * 1.0
    grad[:, 0] += weights2[1*n:2*n] * (-1.0)
    grad[:, 1] += weights2[2*n:3*n] * 1.0
    grad[:, 1] += weights2[3*n:4*n] * (-1.0)
    grad[:, 2] += weights2[4*n:5*n] * 1.0
    grad[:, 2] += weights2[5*n:6*n] * (-1.0)
    
    # Pairwise contributions
    # pair_dists[i] = |c_a - c_b| / 2
    # d/d(c_a) = (c_a - c_b) / |c_a - c_b| * (1/2) * 2 = (c_a - c_b)/|c_a - c_b|
    # Wait: d(|c_a-c_b|/2)/d(c_a) = (c_a-c_b)/|c_a-c_b| * (1/2) * 1 ... 
    # Actually d(|v|)/d(v) = v/|v|, so d(|c_a-c_b|/2)/d(c_a) = (c_a-c_b)/|c_a-c_b| * 1/2 * 2? No.
    # f = |c_a - c_b| / 2
    # df/d(c_a) = (c_a - c_b) / |c_a - c_b| * (1/2)
    # df/d(c_b) = -(c_a - c_b) / |c_a - c_b| * (1/2)
    
    idx = 6 * n
    for i in range(n):
        for j in range(i + 1, n):
            w = weights2[idx]
            idx += 1
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                unit = diff / dist
                grad[i] += w * unit * 0.5
                grad[j] += w * (-unit) * 0.5
    
    return soft_min_val2, grad


def _init_bcc():
    """BCC-like initialization for 50 spheres."""
    # Use a 3x3x3 grid = 27 points, plus add more
    # Better: use a structured grid with some perturbation
    # Let's try 4x4x4 = 64, take 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_random(seed=42):
    """Random initialization within the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(N, 3))


def construct():
    start_time = time.time()
    
    best_centers = None
    best_score = -np.inf
    
    # Multiple initializations
    inits = []
    inits.append(('bcc', _init_bcc()))
    for seed in [42, 123, 777, 2024, 999]:
        inits.append((f'rand_{seed}', _init_random(seed)))
    
    for init_name, centers in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        # Soft-min gradient ascent phase
        beta_start = 5.0
        beta_end = 50.0
        n_steps = 300
        lr = 0.01
        
        c = centers.copy()
        
        for step in range(n_steps):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
            
            # Animate beta from low to high
            progress = step / max(n_steps - 1, 1)
            beta = beta_start + (beta_end - beta_start) * progress
            
            val, grad = _soft_min_radius_and_grad(c, beta)
            
            # Gradient ascent (we want to maximize radius)
            c = c + lr * grad
            
            # Project back into cube with small margin
            c = np.clip(c, -0.499, 0.499)
            
            # Periodically check exact score
            if step % 50 == 0 or step == n_steps - 1:
                exact = _exact_radius(c)
                if exact > best_score:
                    best_score = exact
                    best_centers = c.copy()
        
        # Nelder-Mead polish on the best from this init
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 8:
            continue
        
        def neg_radius(x):
            return -_exact_radius(x.reshape(N, 3))
        
        try:
            res = minimize(neg_radius, c.flatten(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            polished = res.x.reshape(N, 3)
            polished = np.clip(polished, -0.499, 0.499)
            exact = _exact_radius(polished)
            if exact > best_score:
                best_score = exact
                best_centers = polished.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 2:
        if best_centers is not None:
            return best_centers
        return _init_bcc()
    
    # Final polish if we have time
    if best_centers is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 3:
            def neg_radius(x):
                return -_exact_radius(x.reshape(N, 3))
            try:
                res = minimize(neg_radius, best_centers.flatten(), method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                polished = res.x.reshape(N, 3)
                polished = np.clip(polished, -0.499, 0.499)
                exact = _exact_radius(polished)
                if exact > best_score:
                    best_score = exact
                    best_centers = polished.copy()
            except Exception:
                pass
    
    if best_centers is None:
        best_centers = _init_bcc()
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
