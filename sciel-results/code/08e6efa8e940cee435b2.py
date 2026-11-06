"""Combined approach: lattice initializations + epigraph formulation for sphere packing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _exact_radius(C):
    """Compute the exact packing radius."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    if n < 2:
        return np.inf
    
    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distance
    min_bnd = np.min(0.5 - np.abs(C))
    
    return min(min_pair, min_bnd)


def _epigraph_objective_and_grad(x, beta=10.0):
    """
    Epigraph formulation: x = [C_1, ..., C_N, t]
    We want to maximize t such that:
      - t <= d_ij / 2 for all i < j
      - t <= 0.5 - |c_ik| for all i, k
    
    We minimize -t + (1/beta) * sum(exp(-beta * (constraint - t)))
    where constraints are the pairwise and boundary distances.
    
    This is a smooth surrogate for the nonsmooth min.
    """
    C = x[:3*N].reshape(N, 3)
    t = x[3*N]
    
    # Pairwise terms
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    dists = np.sqrt(dists_sq + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    triu_idx = np.triu_indices(N, k=1)
    pair_dists = dists[triu_idx]
    pair_constraints = pair_dists / 2.0
    
    # Boundary terms
    boundary_terms = 0.5 - np.abs(C)
    boundary_flat = boundary_terms.flatten()
    
    all_constraints = np.concatenate([pair_constraints, boundary_flat])
    
    # Soft penalty: (1/beta) * sum(exp(-beta * (c_i - t)))
    # This penalizes when c_i < t
    diff_from_t = all_constraints - t
    # For numerical stability
    max_diff = np.max(diff_from_t)
    if max_diff > 50.0:
        # All constraints are well above t, penalty is negligible
        penalty = 0.0
        grad_penalty_C = np.zeros_like(C)
        grad_penalty_t = 0.0
    else:
        exp_vals = np.exp(-beta * (diff_from_t - max_diff))
        sum_exp = np.sum(exp_vals)
        penalty = (1.0 / beta) * np.log(sum_exp) + max_diff
        
        # Gradient with respect to C and t
        # d(penalty)/d(c_i) = -exp(-beta*(c_i - t - max_diff)) / sum_exp * beta / beta = -exp_vals[i]/sum_exp
        # Wait, let me re-derive.
        # penalty = (1/beta) * log(sum_j exp(-beta*(c_j - t)))
        # Let S = sum_j exp(-beta*(c_j - t))
        # penalty = (1/beta) * log(S)
        # d(penalty)/d(c_j) = (1/beta) * (1/S) * d(S)/d(c_j) = (1/beta) * (1/S) * (-beta) * exp(-beta*(c_j - t)) = -exp(-beta*(c_j - t))/S
        # d(penalty)/d(t) = (1/beta) * (1/S) * d(S)/d(t) = (1/beta) * (1/S) * (beta) * sum_j exp(-beta*(c_j - t)) = (1/beta) * S/S = 1
        
        weights = exp_vals / sum_exp
        
        # Gradient of penalty w.r.t. C
        grad_penalty_C = np.zeros_like(C)
        i_idx = triu_idx[0]
        j_idx = triu_idx[1]
        
        d_vecs = C[i_idx] - C[j_idx]
        d_norms = pair_dists
        d_norms_safe = np.where(d_norms < 1e-15, 1e-15, d_norms)
        
        # d(c_pair)/dC_i = (C_i - C_j) / d_ij
        # d(c_pair)/dC_j = -(C_i - C_j) / d_ij
        # c_pair = d_ij / 2, so d(c_pair)/dC_i = (C_i - C_j) / (2 * d_ij)
        
        pair_weights = weights[:len(pair_constraints)]
        
        grad_pair_i = pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
        grad_pair_j = -pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
        
        # penalty gradient: -weights * d(constraint)/dC
        np.add.at(grad_penalty_C, i_idx, -grad_pair_i)
        np.add.at(grad_penalty_C, j_idx, -grad_pair_j)
        
        # Boundary gradient
        bnd_weights = weights[len(pair_constraints):].reshape(N, 3)
        signs = np.sign(C)
        signs[np.abs(C) < 1e-15] = 1.0
        # d(boundary)/dC = -sign(C)
        # penalty gradient: -weights * d(constraint)/dC = -bnd_weights * (-signs) = bnd_weights * signs
        grad_penalty_C += bnd_weights * signs
        
        grad_penalty_t = 1.0
    
    # Objective: -t + penalty
    obj = -t + penalty
    
    # Gradient: [-grad_penalty_C, -1 + grad_penalty_t]
    grad_C = -grad_penalty_C.flatten()
    grad_t = -1.0 + grad_penalty_t
    
    grad = np.concatenate([grad_C, [grad_t]])
    
    return obj, grad


def _bcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    n_cells = max(1, int(round(n_spheres / 2.0)))
    m = int(np.ceil(n_cells ** (1.0 / 3.0)))
    while m ** 3 < n_cells:
        m += 1
        
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    points = points[:n_spheres]
    
    span = m - 1 if m > 1 else 1.0
    scale = 0.9 / span if span > 0 else 1.0
    points = points * scale - scale * (m - 1) / 2.0
    
    points += rng.uniform(-0.02, 0.02, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points


def _fcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    n_cells = max(1, int(round(n_spheres / 4.0)))
    m = int(np.ceil(n_cells ** (1.0 / 3.0)))
    while m ** 3 < n_cells:
        m += 1
        
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz])
                points.append([ix + 0.5, iy, iz + 0.5])
                points.append([ix, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    points = points[:n_spheres]
    
    span = m - 1 if m > 1 else 1.0
    scale = 0.9 / span if span > 0 else 1.0
    points = points * scale - scale * (m - 1) / 2.0
    
    points += rng.uniform(-0.02, 0.02, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points


def _cubic_grid_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n_spheres].copy()
    grid += rng.uniform(-0.01, 0.01, size=grid.shape)
    return grid


def _jittered_grid_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n_spheres].copy()
    # Larger jitter
    grid += rng.uniform(-0.05, 0.05, size=grid.shape)
    grid = np.clip(grid, -0.45, 0.45)
    return grid


def _random_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(n_spheres, 3))


def _optimize_from_init(C_start, deadline, max_iter=500):
    """Optimize using epigraph formulation from a starting configuration."""
    t_start = time.time()
    
    # Initial t estimate
    r_init = _exact_radius(C_start)
    x0 = np.concatenate([C_start.flatten(), [r_init]])
    
    bounds = [(-0.499, 0.499)] * (3 * N) + [(-0.5, 0.5)]
    
    # Coarse-to-fine beta schedule
    betas = [5.0, 20.0, 50.0, 100.0, 200.0]
    x_current = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _epigraph_objective_and_grad,
                x_current,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'maxfun': max_iter * 3,
                         'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_new = result.x
            C_new = x_new[:3*N].reshape(N, 3)
            t_new = x_new[3*N]
            
            # Validate
            if np.all(np.isfinite(C_new)) and np.all(C_new >= -0.5) and np.all(C_new <= 0.5):
                r_new = _exact_radius(C_new)
                r_cur = _exact_radius(x_current[:3*N].reshape(N, 3))
                if r_new >= r_cur - 1e-10:
                    x_current = x_new
        except Exception:
            pass
    
    C_opt = x_current[:3*N].reshape(N, 3)
    C_opt = np.clip(C_opt, -0.499, 0.499)
    return C_opt


def _polish(C, deadline):
    """Final high-precision polish pass."""
    t_start = time.time()
    r = _exact_radius(C)
    x0 = np.concatenate([C.flatten(), [r]])
    
    bounds = [(-0.499, 0.499)] * (3 * N) + [(-0.5, 0.5)]
    
    betas = [200.0, 500.0, 1000.0]
    x_current = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(
                _epigraph_objective_and_grad,
                x_current,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 900,
                         'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_new = result.x
            C_new = x_new[:3*N].reshape(N, 3)
            
            if np.all(np.isfinite(C_new)) and np.all(C_new >= -0.5) and np.all(C_new <= 0.5):
                r_new = _exact_radius(C_new)
                r_cur = _exact_radius(x_current[:3*N].reshape(N, 3))
                if r_new >= r_cur - 1e-12:
                    x_current = x_new
        except Exception:
            pass
    
    C_opt = x_current[:3*N].reshape(N, 3)
    C_opt = np.clip(C_opt, -0.499, 0.499)
    return C_opt


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Leave 10s margin for the 150s kill
    
    # Generate multiple initializations
    initializations = [
        _bcc_lattice_init(N, seed=42),
        _fcc_lattice_init(N, seed=42),
        _cubic_grid_init(N, seed=42),
        _jittered_grid_init(N, seed=42),
        _bcc_lattice_init(N, seed=123),
        _fcc_lattice_init(N, seed=123),
        _cubic_grid_init(N, seed=123),
        _jittered_grid_init(N, seed=123),
        _random_init(N, seed=7),
        _random_init(N, seed=99),
    ]
    
    # Evaluate initializations and pick the best few
    init_radii = [_exact_radius(init) for init in initializations]
    sorted_indices = np.argsort(init_radii)[::-1]
    
    best_C = None
    best_r = -np.inf
    
    # Try optimizing from the top 4 initializations
    for idx in sorted_indices[:4]:
        if time.time() > deadline - 30:
            break
        
        init = initializations[idx]
        C_opt = _optimize_from_init(init, deadline)
        r_opt = _exact_radius(C_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_C = C_opt.copy()
    
    # Final polish
    if best_C is not None and time.time() < deadline - 5:
        best_C = _polish(best_C, deadline)
        best_r = _exact_radius(best_C)
    
    # Final validation: ensure points are strictly inside the cube
    if best_C is not None:
        best_C = np.clip(best_C, -0.499, 0.499)
    else:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_C = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_C
# EVOLVE-BLOCK-END
