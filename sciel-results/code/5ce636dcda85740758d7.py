"""
Optimized 50-sphere packing in a unit cube.
Uses vectorized soft-min radius optimization with L-BFGS-B.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50

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

def _soft_min_radius_and_grad(C, beta):
    """
    Compute the soft-min radius and its gradient.
    Returns (value, gradient) where value is -soft_min_radius.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    # Pairwise terms
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    # We only need upper triangle for pairs
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    pair_terms = pair_dists / 2.0
    
    # Boundary terms
    boundary_terms = 0.5 - np.abs(C)
    boundary_terms_flat = boundary_terms.flatten()
    
    # Combine all terms
    all_terms = np.concatenate([pair_terms, boundary_terms_flat])
    
    # Soft min calculation
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient calculation
    # d(soft_min)/d(term_k) = weights[k]
    # We need d(soft_min)/d(C)
    
    grad_C = np.zeros_like(C)
    
    # Pairwise gradient
    # term_k = |c_i - c_j| / 2
    # d term_k / d c_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # d term_k / d c_j = (c_j - c_i) / (2 * |c_i - c_j|)
    
    pair_weights = weights[:len(pair_terms)]
    
    # Vectorized gradient for pairs
    # triu_idx[0] contains i, triu_idx[1] contains j
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    
    # d_vec = c_i - c_j
    d_vecs = C[i_idx] - C[j_idx]
    d_norms = np.linalg.norm(d_vecs, axis=1)
    
    # Avoid division by zero
    d_norms_safe = np.where(d_norms < 1e-15, 1e-15, d_norms)
    
    # grad_term_k = d_vecs / (2 * d_norms)
    # Contribution to c_i: pair_weights * grad_term_k
    # Contribution to c_j: -pair_weights * grad_term_k
    
    # Use np.add.at for accumulation
    grad_pair_i = pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    grad_pair_j = -pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    
    np.add.at(grad_C, i_idx, grad_pair_i)
    np.add.at(grad_C, j_idx, grad_pair_j)
    
    # Boundary gradient
    # term = 0.5 - |c_k|
    # d term / d c_k = -sign(c_k)
    bnd_weights = weights[len(pair_terms):].reshape(n, 3)
    signs = np.sign(C)
    signs[np.abs(C) < 1e-15] = 1.0
    grad_C += -bnd_weights * signs
    
    # The objective is -soft_min_val, so gradient is -grad_soft_min
    # Wait, we want to MAXIMIZE soft_min_radius.
    # minimize(-soft_min_radius)
    # grad(-soft_min) = -grad(soft_min)
    # The weights computed above are for soft_min.
    # So we return -grad_C as the gradient of the objective.
    
    return -soft_min_val, -grad_C.flatten()

def _bcc_lattice_init(n_spheres=50, seed=42):
    """Generate n_spheres centers on a distorted BCC superlattice."""
    rng = np.random.RandomState(seed)
    n_cells = max(1, int(round(n_spheres / 2.0)))
    m = int(np.ceil(n_cells ** (1.0 / 3.0)))
    while m ** 3 < n_cells:
        m += 1
        
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner
                points.append([ix, iy, iz])
                # Body center
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    points = points[:n_spheres]
    
    span = m - 1 if m > 1 else 1.0
    scale = 0.9 / span if span > 0 else 1.0
    points = points * scale - scale * (m - 1) / 2.0
    
    # Add small noise to break symmetry
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points

def _fcc_lattice_init(n_spheres=50, seed=42):
    """Generate n_spheres centers on a distorted FCC superlattice."""
    rng = np.random.RandomState(seed)
    # FCC has 4 atoms per unit cell
    n_cells = max(1, int(round(n_spheres / 4.0)))
    m = int(np.ceil(n_cells ** (1.0 / 3.0)))
    while m ** 3 < n_cells:
        m += 1
        
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corners (1/8 of atom if inside, but we just list them and clip later)
                # Simpler: just list the 4 basis positions
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz])
                points.append([ix + 0.5, iy, iz + 0.5])
                points.append([ix, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    points = points[:n_spheres]
    
    span = m - 1 if m > 1 else 1.0
    scale = 0.9 / span if span > 0 else 1.0
    points = points * scale - scale * (m - 1) / 2.0
    
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points

def _cubic_grid_init(n_spheres=50):
    """Simple cubic grid initialization."""
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_spheres].copy()

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Leave 5s margin for the 150s kill

    # Initializations
    initializations = [
        _bcc_lattice_init(N, seed=42),
        _fcc_lattice_init(N, seed=42),
        _cubic_grid_init(N),
        _bcc_lattice_init(N, seed=123),
        _fcc_lattice_init(N, seed=123),
    ]
    
    # Add random initialization
    rng = np.random.RandomState(7)
    initializations.append(rng.uniform(-0.4, 0.4, size=(N, 3)))
    
    best_C = None
    best_r = -np.inf
    
    # Evaluate initializations
    for init in initializations:
        r = _exact_radius(init)
        if r > best_r:
            best_r = r
            best_C = init.copy()

    # Optimization strategy:
    # 1. Coarse optimization with low beta to escape local minima
    # 2. Fine optimization with high beta to push radius up
    
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        def objective(x):
            C = x.reshape(-1, 3)
            val, _ = _soft_min_radius_and_grad(C, beta)
            return val
            
        def gradient(x):
            C = x.reshape(-1, 3)
            _, grad = _soft_min_radius_and_grad(C, beta)
            return grad

        x0 = best_C.flatten()
        
        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            C_new = res.x.reshape(-1, 3)
            r_new = _exact_radius(C_new)
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 5.0:
        beta_final = 1000.0
        
        def objective_final(x):
            C = x.reshape(-1, 3)
            val, _ = _soft_min_radius_and_grad(C, beta_final)
            return val
            
        def gradient_final(x):
            C = x.reshape(-1, 3)
            _, grad = _soft_min_radius_and_grad(C, beta_final)
            return grad

        try:
            res = minimize(
                objective_final,
                best_C.flatten(),
                jac=gradient_final,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            C_new = res.x.reshape(-1, 3)
            r_new = _exact_radius(C_new)
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass

    # Ensure validity: clip to cube
    best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_C
