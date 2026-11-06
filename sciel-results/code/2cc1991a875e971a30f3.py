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

def _repulsive_energy_and_grad(C, power=4.0):
    """
    Compute repulsive energy E = sum_{i<j} 1/d_{ij}^power and its gradient.
    Used to spread points apart initially.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1) + 1e-12
    np.fill_diagonal(dists_sq, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[triu_idx]
    
    # E = sum (d^2)^(-power/2)
    # dE/dd^2 = -power/2 * (d^2)^(-power/2 - 1)
    # dE/dx_i = dE/dd^2 * d(d^2)/dx_i
    # d(d^2)/dx_i = 2 * (x_i - x_j)
    
    p = power
    inv_d_sq_p = np.power(pair_dists_sq, -p/2.0)
    energy = np.sum(inv_d_sq_p)
    
    # Gradient
    # dE/dx_i = sum_j -p * (d^2)^(-p/2 - 1) * (x_i - x_j)
    coeff = -p * np.power(pair_dists_sq, -p/2.0 - 1.0)
    
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    
    d_vecs = C[i_idx] - C[j_idx]
    
    grad_C = np.zeros_like(C)
    
    # grad for i
    grad_i = coeff[:, np.newaxis] * d_vecs
    np.add.at(grad_C, i_idx, grad_i)
    
    # grad for j
    grad_j = -coeff[:, np.newaxis] * d_vecs
    np.add.at(grad_C, j_idx, grad_j)
    
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
    
    # Soft min calculation using log-sum-exp for numerical stability
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    
    # Clip to prevent underflow/overflow
    shifted = np.clip(shifted, -700, 700)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0 or not np.isfinite(sum_exp):
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
    # BCC: 2 points per unit cell
    # For 50 spheres, use a 3x3x3 grid of cells (27 cells, 54 points) and take 50
    # Or 2x3x3 = 18 cells (36 points) + ... 
    # Let's use a grid that generates exactly or slightly more than n_spheres
    # 50 is close to 54 (3x3x3 BCC)
    m = 3
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    # Shuffle to randomize which 50 are picked, but keep symmetry? 
    # Better to just take the first 50 after shuffling indices
    idx = rng.permutation(len(points))
    points = points[idx[:n_spheres]]
    
    # Center and scale
    # BCC in a 3x3x3 cell grid spans 0 to 3.
    # We want it in [-0.5, 0.5]
    # Center is 1.5
    points = points - 1.5
    # Scale to fit in [-0.45, 0.45] to allow movement
    # Max coord is 1.5. So scale by 0.45/1.5 = 0.3
    points *= 0.3
    
    # Add small noise
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    return points

def _fcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # FCC: 4 points per unit cell
    # 50 / 4 = 12.5. Use 3x3x2 = 18 cells? 18*4=72 points.
    m_x, m_y, m_z = 3, 3, 2
    points = []
    for ix in range(m_x):
        for iy in range(m_y):
            for iz in range(m_z):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz])
                points.append([ix + 0.5, iy, iz + 0.5])
                points.append([ix, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    idx = rng.permutation(len(points))
    points = points[idx[:n_spheres]]
    
    # Center
    cx, cy, cz = (m_x-1)/2.0, (m_y-1)/2.0, (m_z-1)/2.0
    points = points - np.array([cx, cy, cz])
    
    # Scale
    # Max extent approx 1.5
    points *= 0.3
    
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    return points

def _random_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # Random points in a slightly smaller cube to avoid immediate boundary issues
    return rng.uniform(-0.4, 0.4, size=(n_spheres, 3))

def _grid_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # 5x5x4 grid = 100 points
    m_x, m_y, m_z = 5, 5, 4
    points = []
    for ix in range(m_x):
        for iy in range(m_y):
            for iz in range(m_z):
                points.append([ix, iy, iz])
    points = np.array(points)
    idx = rng.permutation(len(points))
    points = points[idx[:n_spheres]]
    
    cx, cy, cz = (m_x-1)/2.0, (m_y-1)/2.0, (m_z-1)/2.0
    points = points - np.array([cx, cy, cz])
    
    # Scale
    # Max extent 2.0
    points *= 0.22 # 0.45/2.0
    
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    return points

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # 5s margin
    
    initializations = [
        _bcc_lattice_init(N, seed=42),
        _fcc_lattice_init(N, seed=42),
        _grid_init(N, seed=42),
        _random_init(N, seed=7),
        _bcc_lattice_init(N, seed=123),
        _fcc_lattice_init(N, seed=123),
        _grid_init(N, seed=123),
        _random_init(N, seed=456),
    ]
    
    best_C = None
    best_r = -np.inf
    
    # Evaluate initializations
    for init in initializations:
        r = _exact_radius(init)
        if r > best_r:
            best_r = r
            best_C = init.copy()
    
    # Optimization Strategy:
    # 1. Repulsive relaxation (optional, might be slow)
    # 2. Soft-min schedule
    
    # We will run the optimization for each initialization
    # But we can also just take the best init and run multiple times?
    # Better to run on all, but with time limits.
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    for init in initializations:
        if time.time() > deadline - 20.0:
            break
            
        C_curr = init.copy()
        
        # Optional: Repulsive relaxation
        # Skip for now to save time, soft-min with low beta is similar
        
        # Step 1: Soft-min schedule
        for beta in betas:
            if time.time() > deadline - 10.0:
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
                # Use L-BFGS-B
                res = minimize(
                    soft_obj,
                    C_curr.flatten(),
                    jac=soft_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                C_new = res.x.reshape(-1, 3)
                
                # Check if the new solution is better
                r_new = _exact_radius(C_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
                
                # Update current for next beta step
                C_curr = C_new
            except Exception:
                pass
    
    # Final high-precision polish on global best
    if time.time() < deadline - 5.0 and best_C is not None:
        beta_final = 10000.0
        
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
        
        # Another round of hard-min polishing (subgradient)
        if time.time() < deadline - 2.0:
            # Use a very high beta soft-min as a proxy for hard min
            # Or implement a simple subgradient step
            # Let's just do another high beta pass
            beta_hard = 50000.0
            def hard_obj(x):
                C = x.reshape(-1, 3)
                val, _ = _soft_min_radius_and_grad(C, beta_hard)
                return val
            def hard_grad(x):
                C = x.reshape(-1, 3)
                _, grad = _soft_min_radius_and_grad(C, beta_hard)
                return grad

            try:
                res = minimize(
                    hard_obj,
                    best_C.flatten(),
                    jac=hard_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
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
        best_C = _random_init(N)
        best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
        
    return best_C
