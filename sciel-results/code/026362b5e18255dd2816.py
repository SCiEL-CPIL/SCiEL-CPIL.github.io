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

def _repulsive_energy_and_grad(C, power=6.0):
    """
    Compute repulsive energy E = sum_{i<j} 1/d_{ij}^power and its gradient.
    This helps spread points apart quickly.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    
    # Energy: sum(1/d^p)
    # Gradient: d/dx (1/d^p) = -p * d^{-(p+2)} * (x_i - x_j)
    
    # To avoid overflow/underflow with high power, we can scale
    # But for 50 points, distances are ~0.1, so 1/d^6 is ~1e6, manageable.
    
    inv_d_p = 1.0 / (pair_dists ** power)
    energy = np.sum(inv_d_p)
    
    # Gradient calculation
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    d_vecs = C[i_idx] - C[j_idx]
    d_norms = pair_dists
    
    # grad_E / d c_i = sum_j -p * d^{-(p+2)} * (c_i - c_j)
    # grad_E / d c_j = sum_i -p * d^{-(p+2)} * (c_j - c_i) = sum_i p * d^{-(p+2)} * (c_i - c_j)
    
    coeff = -power * (d_norms ** (-(power + 2)))
    
    grad_C = np.zeros_like(C)
    
    # Contribution to i
    np.add.at(grad_C, i_idx, coeff[:, np.newaxis] * d_vecs)
    # Contribution to j
    np.add.at(grad_C, j_idx, -coeff[:, np.newaxis] * d_vecs)
    
    return energy, grad_C.flatten()

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
    # Clip to prevent underflow/overflow in exp, though log-sum-exp is stable
    # If shifted is very negative, exp is 0. If very positive (shouldn't happen as min_val is min), exp is large.
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        # Fallback to hard min
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
    
    # d term_k / d c_i = (c_i - c_j) / (2 * d)
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

def _jittered_grid_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = np.arange(m) / m
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n_spheres].copy()
    # Jitter to break symmetry
    grid += rng.uniform(-0.1, 0.1, size=grid.shape)
    grid = (grid + 0.5) / m - 0.5 # Normalize to [-0.5, 0.5] roughly
    # Actually, let's just scale the original grid points to fit and jitter
    span = m
    scale = 0.9 / span
    grid = (np.arange(n_spheres) % m + 0.5) * scale # x
    # This is getting complicated, let's stick to simple grid + jitter
    return _cubic_grid_init(n_spheres, seed)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Leave 10s margin for the 150s kill

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
    
    best_C = None
    best_r = -np.inf
    
    # Evaluate initializations
    for init in initializations:
        r = _exact_radius(init)
        if r > best_r:
            best_r = r
            best_C = init.copy()

    # Optimization Strategy:
    # 1. Repulsive pre-optimization
    # 2. Coarse-to-fine soft-min
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for init in initializations:
        if time.time() > deadline - 30.0:
            break
            
        C_curr = init.copy()
        
        # Step 1: Repulsive minimization to spread points
        def rep_obj(x):
            C = x.reshape(-1, 3)
            val, _ = _repulsive_energy_and_grad(C, power=6.0)
            return val
            
        def rep_grad(x):
            C = x.reshape(-1, 3)
            _, grad = _repulsive_energy_and_grad(C, power=6.0)
            return grad

        try:
            res = minimize(
                rep_obj,
                C_curr.flatten(),
                jac=rep_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            C_new = res.x.reshape(-1, 3)
            C_curr = C_new
        except Exception:
            pass

        # Step 2: Soft-min schedule
        for beta in betas:
            if time.time() > deadline - 20.0:
                break
                
            def soft_obj(x, b=beta):
                C = x.reshape(-1, 3)
                val, _ = _soft_min_radius_and_grad(C, b)
                return val
                
            def soft_grad(x, b=beta):
                C = x.reshape(-1, 3)
                _, grad = _soft_min_radius_and_grad(C, b)
                return grad

            try:
                res = minimize(
                    soft_obj,
                    C_curr.flatten(),
                    jac=soft_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                C_new = res.x.reshape(-1, 3)
                r_new = _exact_radius(C_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
                
                # Always update C_curr to the new position for next beta step
                # even if it didn't beat global best, it might be better for this init
                # But if it got worse, we might want to stick to previous?
                # Usually L-BFGS-B monotonically decreases the objective (increases radius).
                C_curr = C_new
            except Exception:
                pass

    # Final high-precision polish on global best
    if time.time() < deadline - 5.0 and best_C is not None:
        beta_final = 2000.0
        
        def final_obj(x):
            C = x.reshape(-1, 3)
            val, _ = _soft_min_radius_and_grad(C, beta_final)
            return val
            
        def final_grad(x):
            C = x.reshape(-1, 3)
            _, grad = _soft_min_radius_and_grad(C, beta_final)
            return grad

        try:
            res = minimize(
                final_obj,
                best_C.flatten(),
                jac=final_grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            C_new = res.x.reshape(-1, 3)
            r_new = _exact_radius(C_new)
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass

    # Ensure validity: clip to cube
    if best_C is not None:
        best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    else:
        best_C = _cubic_grid_init(N)
        best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
        
    return best_C
