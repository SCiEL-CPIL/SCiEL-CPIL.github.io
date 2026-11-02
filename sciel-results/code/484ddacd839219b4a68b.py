"""Packing 100 equal spheres in a unit cube: multi-strategy with coordinate-descent minmax."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the packing radius for centres in cube [-0.5, 0.5]^3."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    bound = np.minimum.reduce([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 2],
        centers[:, 2] + 0.5,
    ])
    min_bound = np.min(bound)
    return 0.5 * min(min_pair, min_bound)


def _soft_min_objective(flat, beta):
    """Negative soft-min of pairwise and boundary distances."""
    C = flat.reshape(N, 3)
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    idx = np.triu_indices(N, k=1)
    pair_dists = dists[idx]
    bounds = np.concatenate([
        0.5 - C[:, 0], C[:, 0] + 0.5,
        0.5 - C[:, 1], C[:, 1] + 0.5,
        0.5 - C[:, 2], C[:, 2] + 0.5,
    ])
    all_dists = np.concatenate([pair_dists, bounds])
    neg_beta_d = -beta * all_dists
    max_val = np.max(neg_beta_d)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_d - max_val)))
    soft_min = -log_sum_exp / beta
    return -soft_min


def _soft_min_grad(flat, beta):
    """Gradient of -soft_min_objective."""
    C = flat.reshape(N, 3)
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    idx = np.triu_indices(N, k=1)
    pair_dists = dists[idx]
    bounds = np.concatenate([
        0.5 - C[:, 0], C[:, 0] + 0.5,
        0.5 - C[:, 1], C[:, 1] + 0.5,
        0.5 - C[:, 2], C[:, 2] + 0.5,
    ])
    all_dists = np.concatenate([pair_dists, bounds])
    neg_beta_d = -beta * all_dists
    max_val = np.max(neg_beta_d)
    exps = np.exp(neg_beta_d - max_val)
    sum_exps = np.sum(exps)
    weights = exps / sum_exps

    grad_C = np.zeros_like(C)
    for k, (i, j) in enumerate(zip(idx[0], idx[1])):
        d = pair_dists[k]
        if d < 1e-12:
            continue
        w = weights[k]
        diff_ij = C[i] - C[j]
        grad_C[i] -= w * diff_ij / d
        grad_C[j] += w * diff_ij / d

    base = N * (N - 1) // 2
    for i in range(N):
        w0 = weights[base + 6 * i + 0]
        w1 = weights[base + 6 * i + 1]
        w2 = weights[base + 6 * i + 2]
        w3 = weights[base + 6 * i + 3]
        w4 = weights[base + 6 * i + 4]
        w5 = weights[base + 6 * i + 5]
        grad_C[i, 0] = w0 - w1
        grad_C[i, 1] = w2 - w3
        grad_C[i, 2] = w4 - w5

    return grad_C.ravel()


def _build_initial_grid(n_points):
    """Build a jittered grid initial configuration."""
    m = int(np.ceil(n_points ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:n_points]
    return grid


def _build_bcc_init(n_points):
    """Build a BCC-like initial configuration."""
    m = int(np.ceil((n_points / 2) ** (1.0 / 3.0)))
    pts = []
    for iz in range(m):
        for iy in range(m):
            for ix in range(m):
                x = (ix + 0.5) / m - 0.5
                y = (iy + 0.5) / m - 0.5
                z = (iz + 0.5) / m - 0.5
                pts.append([x, y, z])
                if (ix + iy + iz) % 2 == 0:
                    cx = (ix + 0.25) / m - 0.5
                    cy = (iy + 0.25) / m - 0.5
                    cz = (iz + 0.25) / m - 0.5
                    pts.append([cx, cy, cz])
    pts = np.array(pts[:n_points])
    return pts


def _build_halton_init(n_points):
    """Build a Halton sequence initial configuration."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    pts = []
    for i in range(1, n_points + 1):
        pts.append([halton(i, 2), halton(i, 3), halton(i, 5)])
    pts = np.array(pts) - 0.5
    return pts


def _coordinate_descent_minmax(C, max_iter=200, tol=1e-12):
    """
    Coordinate-descent minmax refinement:
    Directly push apart the binding pair and inward wall-violating points.
    """
    C = C.copy()
    for it in range(max_iter):
        # Compute current radius
        D = cdist(C, C)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D)
        pair_i, pair_j = np.unravel_index(np.argmin(D), D.shape)
        
        bound = np.minimum.reduce([
            0.5 - C[:, 0], C[:, 0] + 0.5,
            0.5 - C[:, 1], C[:, 1] + 0.5,
            0.5 - C[:, 2], C[:, 2] + 0.5,
        ])
        min_bound = np.min(bound)
        bound_idx = np.argmin(bound)
        
        r = 0.5 * min(min_pair, min_bound)
        
        # Determine which constraint is binding
        if min_pair < min_bound - 1e-14:
            # Binding: pair distance
            diff = C[pair_i] - C[pair_j]
            d = np.linalg.norm(diff)
            if d < 1e-14:
                # Coincident, push randomly
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                step = 1e-6
                C[pair_i] += step * direction
                C[pair_j] -= step * direction
            else:
                direction = diff / d
                # Push apart
                step = 1e-5
                C[pair_i] += step * direction
                C[pair_j] -= step * direction
        else:
            # Binding: boundary
            # Determine which wall
            b = bound[bound_idx]
            # Figure out which of the 6 boundaries is the min
            c = C[bound_idx]
            dists_to_walls = np.array([
                0.5 - c[0], c[0] + 0.5,
                0.5 - c[1], c[1] + 0.5,
                0.5 - c[2], c[2] + 0.5,
            ])
            wall_idx = np.argmin(dists_to_walls)
            # Push inward
            if wall_idx == 0:   # x = -0.5
                C[bound_idx, 0] += 1e-5
            elif wall_idx == 1: # x = +0.5
                C[bound_idx, 0] -= 1e-5
            elif wall_idx == 2: # y = -0.5
                C[bound_idx, 1] += 1e-5
            elif wall_idx == 3: # y = +0.5
                C[bound_idx, 1] -= 1e-5
            elif wall_idx == 4: # z = -0.5
                C[bound_idx, 2] += 1e-5
            elif wall_idx == 5: # z = +0.5
                C[bound_idx, 2] -= 1e-5
        
        # Project to cube
        C = np.clip(C, -0.5, 0.5)
        
        # Check convergence
        if it > 10:
            D2 = cdist(C, C)
            np.fill_diagonal(D2, np.inf)
            min_pair2 = np.min(D2)
            bound2 = np.minimum.reduce([
                0.5 - C[:, 0], C[:, 0] + 0.5,
                0.5 - C[:, 1], C[:, 1] + 0.5,
                0.5 - C[:, 2], C[:, 2] + 0.5,
            ])
            min_bound2 = np.min(bound2)
            r2 = 0.5 * min(min_pair2, min_bound2)
            if abs(r2 - r) < tol:
                break
    
    return C


def _anneal(C, betas, max_iter_per_beta=300, deadline=None):
    """Anneal through increasing beta values."""
    C = C.copy()
    for beta in betas:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        result = minimize(
            _soft_min_objective,
            C.ravel(),
            jac=_soft_min_grad,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter_per_beta, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        C = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
    return C


def _optimize_from_initial(C_init, deadline):
    """Optimize from a given initial configuration."""
    # Annealing schedule
    betas = [30, 80, 200, 500, 1000, 3000]
    C = _anneal(C_init, betas, max_iter_per_beta=300, deadline=deadline)
    
    # Coordinate-descent minmax polish
    C = _coordinate_descent_minmax(C, max_iter=100, tol=1e-12)
    
    # High-precision soft-min polish
    if time.time() < deadline - 3.0:
        for beta_polish in [5000, 10000, 20000]:
            if time.time() > deadline - 2.0:
                break
            try:
                result = minimize(
                    _soft_min_objective,
                    C.ravel(),
                    jac=_soft_min_grad,
                    args=(beta_polish,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                C = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
            except Exception:
                break
        
        # Final coordinate-descent
        C = _coordinate_descent_minmax(C, max_iter=50, tol=1e-13)
    
    return C


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit
    
    best_C = None
    best_r = 0.0
    
    np.random.seed(123)
    
    # Multiple initial configurations
    inits = []
    
    # Jittered grid
    grid = _build_initial_grid(N)
    grid_jitter = grid + np.random.randn(N, 3) * 0.01
    grid_jitter = np.clip(grid_jitter, -0.5, 0.5)
    inits.append(grid_jitter)
    
    # BCC-like
    bcc = _build_bcc_init(N)
    bcc_jitter = bcc + np.random.randn(N, 3) * 0.005
    bcc_jitter = np.clip(bcc_jitter, -0.5, 0.5)
    inits.append(bcc_jitter)
    
    # Halton sequence
    halton = _build_halton_init(N)
    inits.append(halton)
    
    # Random
    rand = np.random.randn(N, 3) * 0.4
    rand = np.clip(rand, -0.5, 0.5)
    inits.append(rand)
    
    # Optimize from each initial configuration
    for i, C_init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        C_opt = _optimize_from_initial(C_init, deadline)
        r = _compute_radius(C_opt)
        
        if r > best_r:
            best_r = r
            best_C = C_opt.copy()
    
    # If we still have time, try more random restarts
    if best_C is not None and time.time() < deadline - 15.0:
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            C_rand = np.random.randn(N, 3) * 0.35
            C_rand = np.clip(C_rand, -0.5, 0.5)
            C_opt = _optimize_from_initial(C_rand, deadline)
            r = _compute_radius(C_opt)
            if r > best_r:
                best_r = r
                best_C = C_opt.copy()
    
    # Final validation
    if best_C is None:
        best_C = _build_initial_grid(N)
    
    # Ensure strict validity
    best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_C

# EVOLVE-BLOCK-END
