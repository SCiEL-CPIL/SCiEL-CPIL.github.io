"""Combined approach: lattice initializations + two-stage optimization for sphere packing."""
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


def _repulsive_energy_and_grad(C, power=6.0):
    """Compute repulsive energy E = sum_{i<j} 1/d_{ij}^power and its gradient."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    
    inv_d_p = 1.0 / (pair_dists ** power)
    energy = np.sum(inv_d_p)
    
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    d_vecs = C[i_idx] - C[j_idx]
    d_norms = pair_dists
    
    coeff = -power * (d_norms ** (-(power + 2)))
    
    grad_C = np.zeros_like(C)
    np.add.at(grad_C, i_idx, coeff[:, np.newaxis] * d_vecs)
    np.add.at(grad_C, j_idx, -coeff[:, np.newaxis] * d_vecs)
    
    return energy, grad_C.flatten()


def _soft_min_radius_and_grad(C, beta):
    """Compute the negative soft-min radius and its gradient."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    # Pairwise terms
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    pair_terms = pair_dists / 2.0
    
    # Boundary terms
    boundary_terms = 0.5 - np.abs(C)
    boundary_terms_flat = boundary_terms.flatten()
    
    all_terms = np.concatenate([pair_terms, boundary_terms_flat])
    
    # Soft min calculation
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        weights = np.zeros_like(all_terms)
        min_indices = np.where(all_terms == min_val)[0]
        weights[min_indices] = 1.0 / len(min_indices)
    else:
        weights = exp_vals / sum_exp
    
    soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
    
    grad_C = np.zeros_like(C)
    
    pair_weights = weights[:len(pair_terms)]
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    
    d_vecs = C[i_idx] - C[j_idx]
    d_norms = np.linalg.norm(d_vecs, axis=1)
    d_norms_safe = np.where(d_norms < 1e-15, 1e-15, d_norms)
    
    grad_pair_i = pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    grad_pair_j = -pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    
    np.add.at(grad_C, i_idx, grad_pair_i)
    np.add.at(grad_C, j_idx, grad_pair_j)
    
    # Boundary gradient
    bnd_weights = weights[len(pair_terms):].reshape(n, 3)
    signs = np.sign(C)
    signs[np.abs(C) < 1e-15] = 1.0
    grad_C += -bnd_weights * signs
    
    return -soft_min_val, -grad_C.flatten()


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


def _optimize_from_start(C_start, deadline, max_repulsive_iter=200, max_softmin_iter=300):
    """Run two-stage optimization from a starting configuration."""
    t_start = time.time()
    
    # Stage 1: Repulsive energy minimization
    try:
        result_rep = minimize(
            _repulsive_energy_and_grad,
            C_start.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=[(-0.49, 0.49)] * 3 * N,
            options={'maxiter': max_repulsive_iter, 'maxfun': max_repulsive_iter * 2,
                     'ftol': 1e-10, 'gtol': 1e-8}
        )
        C_rep = result_rep.x.reshape(N, 3)
        C_rep = np.clip(C_rep, -0.49, 0.49)
    except Exception:
        C_rep = C_start.copy()
    
    # Check time
    if time.time() > deadline - 30:
        return C_rep
    
    # Stage 2: Coarse-to-fine soft-min optimization
    betas = [5.0, 20.0, 100.0, 500.0]
    C_current = C_rep.copy()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        try:
            result_soft = minimize(
                _soft_min_radius_and_grad,
                C_current.flatten(),
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * 3 * N,
                options={'maxiter': max_softmin_iter, 'maxfun': max_softmin_iter * 2,
                         'ftol': 1e-12, 'gtol': 1e-10}
            )
            C_new = result_soft.x.reshape(N, 3)
            C_new = np.clip(C_new, -0.49, 0.49)
            
            # Only accept if it improves
            if _exact_radius(C_new) >= _exact_radius(C_current) - 1e-10:
                C_current = C_new
        except Exception:
            pass
    
    return C_current


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Leave 10s margin for the 150s kill
    
    # Generate multiple initializations
    initializations = [
        _bcc_lattice_init(N, seed=42),
        _fcc_lattice_init(N, seed=42),
        _cubic_grid_init(N, seed=42),
        _bcc_lattice_init(N, seed=123),
        _fcc_lattice_init(N, seed=123),
        _cubic_grid_init(N, seed=123),
    ]
    
    rng = np.random.RandomState(7)
    initializations.append(rng.uniform(-0.4, 0.4, size=(N, 3)))
    
    # Evaluate initializations and pick the best
    best_r = -np.inf
    best_init = None
    for init in initializations:
        r = _exact_radius(init)
        if r > best_r:
            best_r = r
            best_init = init.copy()
    
    # Optimize from the best initialization
    best_C = _optimize_from_start(best_init, deadline)
    best_r = _exact_radius(best_C)
    
    # Try a few more optimizations if time permits
    for init in initializations[1:3]:  # Try 2 more
        if time.time() > deadline - 20:
            break
        
        C_opt = _optimize_from_start(init, deadline)
        r_opt = _exact_radius(C_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_C = C_opt
    
    # Final validation: ensure points are strictly inside the cube
    best_C = np.clip(best_C, -0.499, 0.499)
    
    return best_C
# EVOLVE-BLOCK-END
