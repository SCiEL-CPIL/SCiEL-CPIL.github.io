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
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    min_bnd = np.min(0.5 - np.abs(C))
    return min(min_pair, min_bnd)

def _repulsive_energy_and_grad(C, power=2.0):
    """
    Compute repulsive energy E = sum_{i<j} 1/d_{ij}^power and its gradient.
    This creates a smooth landscape that pushes points apart.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1) + 1e-15
    np.fill_diagonal(dists_sq, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[triu_idx]
    
    p = power
    # E = sum (d^2)^(-p/2)
    inv_d_sq_p = np.power(pair_dists_sq, -p/2.0)
    energy = np.sum(inv_d_sq_p)
    
    # Gradient: dE/dx_i = -p * (d^2)^(-p/2 - 1) * (x_i - x_j)
    coeff = -p * np.power(pair_dists_sq, -p/2.0 - 1.0)
    
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    d_vecs = C[i_idx] - C[j_idx]
    
    grad_C = np.zeros_like(C)
    grad_i = coeff[:, np.newaxis] * d_vecs
    np.add.at(grad_C, i_idx, grad_i)
    grad_j = -coeff[:, np.newaxis] * d_vecs
    np.add.at(grad_C, j_idx, grad_j)
    
    return energy, grad_C.flatten()

def _soft_min_radius_and_grad(C, beta):
    """
    Compute the negative soft-min radius and its gradient.
    soft_min = min_val - (1/beta) * log(sum(exp(-beta*(x - min_val))))
    We minimize -soft_min to maximize the radius.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    # Pairwise distances / 2
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    pair_terms = pair_dists / 2.0
    
    # Boundary distances
    boundary_terms = 0.5 - np.abs(C)
    boundary_terms_flat = boundary_terms.flatten()
    
    all_terms = np.concatenate([pair_terms, boundary_terms_flat])
    
    # Soft min using log-sum-exp for stability
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
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
    
    grad_pair_i = pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    grad_pair_j = -pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    
    np.add.at(grad_C, i_idx, grad_pair_i)
    np.add.at(grad_C, j_idx, grad_pair_j)
    
    bnd_weights = weights[len(pair_terms):].reshape(n, 3)
    signs = np.sign(C)
    signs[np.abs(C) < 1e-15] = 1.0
    grad_C += -bnd_weights * signs
    
    return -soft_min_val, -grad_C.flatten()

def _bcc_init(seed=42):
    """BCC lattice initialization."""
    rng = np.random.RandomState(seed)
    m = 3
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
    points = np.array(points)
    idx = rng.permutation(len(points))
    points = points[idx[:N]]
    points = points - 1.5
    points *= 0.28
    points += rng.uniform(-0.005, 0.005, size=points.shape)
    return points

def _fcc_init(seed=42):
    """FCC lattice initialization."""
    rng = np.random.RandomState(seed)
    mx, my, mz = 3, 3, 2
    points = []
    for ix in range(mx):
        for iy in range(my):
            for iz in range(mz):
                points.append([ix, iy, iz])
                points.append([ix+0.5, iy+0.5, iz])
                points.append([ix+0.5, iy, iz+0.5])
                points.append([ix, iy+0.5, iz+0.5])
    points = np.array(points)
    idx = rng.permutation(len(points))
    points = points[idx[:N]]
    cx, cy, cz = (mx-1)/2.0, (my-1)/2.0, (mz-1)/2.0
    points = points - np.array([cx, cy, cz])
    points *= 0.28
    points += rng.uniform(-0.005, 0.005, size=points.shape)
    return points

def _grid_init(seed=42):
    """Grid initialization."""
    rng = np.random.RandomState(seed)
    mx, my, mz = 5, 5, 4
    points = []
    for ix in range(mx):
        for iy in range(my):
            for iz in range(mz):
                points.append([ix, iy, iz])
    points = np.array(points)
    idx = rng.permutation(len(points))
    points = points[idx[:N]]
    cx, cy, cz = (mx-1)/2.0, (my-1)/2.0, (mz-1)/2.0
    points = points - np.array([cx, cy, cz])
    points *= 0.22
    points += rng.uniform(-0.005, 0.005, size=points.shape)
    return points

def _random_init(seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.42, 0.42, size=(N, 3))

def _coordinate_descent(C, max_iter=200, tol=1e-10, deadline=None):
    """
    Coordinate descent: for each point, find the optimal position along each axis
    given all other points fixed. This directly optimizes the hard min.
    """
    C = C.copy()
    n = len(C)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # For each axis, find the best position
            for axis in range(3):
                # Compute the constraint function: min of distances to all other points
                # and boundaries, as a function of C[i, axis]
                # We want to maximize the minimum.
                
                # Collect all "constraints" as functions of x = C[i, axis]
                # For each other point j: distance in the other two dims is fixed
                # dist_ij^2 = (x - C[j, axis])^2 + fixed_ij
                # We want to maximize min_j sqrt((x - C[j, axis])^2 + fixed_ij) / 2
                # and min(0.5 - |x|)
                
                # Instead of full optimization, do a local search around current position
                x0 = C[i, axis]
                
                # Evaluate the current radius contribution
                def radius_at(x):
                    C[i, axis] = x
                    # Compute min distance to all others
                    diffs = C[i] - C
                    dists = np.sqrt(np.sum(diffs**2, axis=1))
                    dists[i] = np.inf
                    min_pair = np.min(dists) / 2.0
                    min_bnd = np.min(0.5 - np.abs(C[i]))
                    return min(min_pair, min_bnd)
                
                # Try a few perturbation sizes
                best_x = x0
                best_r = radius_at(x0)
                
                for step in [0.001, 0.0005, 0.0002, 0.0001]:
                    for direction in [-1, 1]:
                        x_try = x0 + direction * step
                        if abs(x_try) > 0.5:
                            continue
                        r_try = radius_at(x_try)
                        if r_try > best_r + tol:
                            best_r = r_try
                            best_x = x_try
                
                if abs(best_x - x0) > tol:
                    C[i, axis] = best_x
                    improved = True
        
        if not improved:
            break
    
    return C

def _local_search(C, deadline):
    """
    Local search using coordinate descent and small random perturbations.
    """
    C = C.copy()
    rng = np.random.RandomState(12345)
    current_r = _exact_radius(C)
    
    for it in range(500):
        if time.time() > deadline:
            break
        
        # Coordinate descent
        C = _coordinate_descent(C, max_iter=50, tol=1e-12, deadline=deadline)
        r = _exact_radius(C)
        if r > current_r:
            current_r = r
            continue
        
        # Try small random perturbations
        for _ in range(10):
            if time.time() > deadline:
                break
            C_try = C.copy()
            idx = rng.randint(N)
            C_try[idx] += rng.normal(0, 0.002, size=3)
            C_try = np.clip(C_try, -0.4999, 0.4999)
            r_try = _exact_radius(C_try)
            if r_try > current_r:
                C = C_try
                current_r = r_try
    
    return C, current_r

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s margin
    
    # Generate multiple initializations
    inits = [
        _bcc_init(42), _bcc_init(123), _bcc_init(789),
        _fcc_init(42), _fcc_init(123), _fcc_init(789),
        _grid_init(42), _grid_init(123), _grid_init(789),
        _random_init(42), _random_init(123), _random_init(789),
    ]
    
    best_C = None
    best_r = -np.inf
    
    # Evaluate initializations
    for init in inits:
        r = _exact_radius(init)
        if r > best_r:
            best_r = r
            best_C = init.copy()
    
    # Phase 1: Repulsive energy optimization to spread points
    # Phase 2: Soft-min radius optimization
    # Phase 3: Local search / coordinate descent
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        C_curr = init.copy()
        
        # Step 1: Optimize repulsive energy (smooth landscape)
        def rep_obj(x):
            C = x.reshape(-1, 3)
            val, _ = _repulsive_energy_and_grad(C, power=2.0)
            return val
        
        def rep_grad(x):
            C = x.reshape(-1, 3)
            _, grad = _repulsive_energy_and_grad(C, power=2.0)
            return grad
        
        try:
            res = minimize(
                rep_obj, C_curr.flatten(), jac=rep_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            C_curr = res.x.reshape(-1, 3)
        except Exception:
            pass
        
        # Step 2: Soft-min radius optimization with increasing beta
        betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        
        for beta in betas:
            if time.time() > deadline - 15.0:
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
                    soft_obj, C_curr.flatten(), jac=soft_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 40}
                )
                C_new = res.x.reshape(-1, 3)
                r_new = _exact_radius(C_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
                
                C_curr = C_new
            except Exception:
                pass
        
        # Step 3: Local search / coordinate descent on the best from this init
        if time.time() < deadline - 5.0:
            C_ls, r_ls = _local_search(C_curr, deadline=deadline - 2.0)
            if r_ls > best_r:
                best_r = r_ls
                best_C = C_ls.copy()
    
    # Final polish on global best
    if best_C is not None and time.time() < deadline - 3.0:
        C_final, r_final = _local_search(best_C, deadline=deadline - 1.0)
        if r_final > best_r:
            best_r = r_final
            best_C = C_final.copy()
    
    # Ensure validity
    if best_C is None:
        best_C = _random_init(42)
    best_C = np.clip(best_C, -0.5 + 1e-14, 0.5 - 1e-14)
    
    return best_C
