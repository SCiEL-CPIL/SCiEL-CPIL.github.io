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
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    pair_terms = pair_dists / 2.0
    
    # Boundary terms
    boundary_terms = 0.5 - np.abs(C)
    boundary_terms_flat = boundary_terms.flatten()
    
    all_terms = np.concatenate([pair_terms, boundary_terms_flat])
    
    # Soft min calculation using log-sum-exp trick
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

def _repulsive_energy_and_grad(C, p=4, wall_softness=0.1):
    """
    Compute repulsive energy and gradient.
    E = sum_{i<j} 1/d^p + sum_i wall_penalty
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    E = 0.0
    grad = np.zeros_like(C)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i+1, n):
            d_vec = C[i] - C[j]
            d_sq = np.dot(d_vec, d_vec)
            if d_sq < 1e-12:
                d_sq = 1e-12
                d_vec = np.array([1e-6, 0, 0]) # Arbitrary direction if coincident
            d = np.sqrt(d_sq)
            
            # E_pair = d^(-p)
            # dE/dC_i = -p * d^(-p-2) * d_vec
            # dE/dC_j = p * d^(-p-2) * d_vec
            
            coeff = p * d_sq ** (-(p/2.0) - 1.0)
            E += d_sq ** (-p/2.0)
            
            grad[i] -= coeff * d_vec
            grad[j] += coeff * d_vec
            
    # Wall repulsion (soft constraint)
    # We want to push spheres away from walls.
    # Let dist_to_wall = 0.5 - |x|
    # Penalty if dist_to_wall < wall_softness
    # Use a smooth potential: V(w) = 1/(w + eps) if w < threshold
    # Or simply a quadratic penalty for being too close.
    # Let's use: if x > 0.5 - softness, push left.
    # Simpler: minimize sum (max(0, |x| - (0.5 - margin))^2)
    # But we want to maximize the min distance to wall.
    # So we want to keep |x| < 0.5 - r.
    # Let's add a repulsive force from the walls.
    
    margin = 0.05 # Start with a safety margin, refine later
    for i in range(n):
        for k in range(3):
            x = C[i, k]
            # Distance to nearest wall in dimension k
            dist_wall = 0.5 - abs(x)
            
            if dist_wall < margin:
                # Push away from wall
                # Force direction is -sign(x)
                # Magnitude proportional to how close we are
                # Let's use a smooth 1/dist^2 repulsion from the wall plane
                # dist_wall is the distance to the boundary plane.
                # If dist_wall is small, force is large.
                force_mag = 1.0 / (dist_wall + 1e-6)**2
                sign = 1.0 if x > 0 else -1.0
                # Gradient of energy w.r.t C[i,k] is -force_mag * sign?
                # Energy V = 1/(0.5 - |x|)^2 ? No, that diverges.
                # Let's use V = 1/(dist_wall + eps)^2
                # dV/dx = -2 * (dist_wall + eps)^-3 * d(dist_wall)/dx
                # d(dist_wall)/dx = -sign(x)
                # dV/dx = 2 * sign(x) / (dist_wall + eps)^3
                # We want to minimize V, so gradient is dV/dx.
                # Wait, if x > 0, dist_wall = 0.5 - x. d(dist_wall)/dx = -1.
                # V = (0.5 - x)^-2. dV/dx = -2(0.5-x)^-3 * (-1) = 2(0.5-x)^-3.
                # This is positive for x < 0.5. So increasing x increases V.
                # So we want to decrease x. Gradient is positive, so we move in -grad direction. Correct.
                
                # Let's just use a simple quadratic penalty for simplicity and stability
                # V = (margin - dist_wall)^2 if dist_wall < margin
                # dV/d(dist_wall) = -2(margin - dist_wall)
                # d(dist_wall)/dx = -sign(x)
                # dV/dx = 2(margin - dist_wall) * sign(x)
                
                excess = margin - dist_wall
                if excess > 0:
                    dV_dx = 2.0 * excess * sign
                    grad[i, k] += dV_dx
                    E += 0.5 * excess * excess
            else:
                # No penalty, but we can add a tiny bias to keep them centered?
                # No, let the pairwise repulsion handle distribution.
                pass
                
    return E, grad

def _bcc_lattice_init(n_spheres=50, seed=42, jitter=0.02):
    rng = np.random.RandomState(seed)
    # BCC: 2 points per unit cell
    # 50 spheres -> 25 cells. 
    # 3x3x3 = 27 cells -> 54 points.
    m = 3
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    # We have 54 points. Pick 50.
    # To maximize radius, we should probably remove the 4 "worst" points or just random.
    # Random is fine for initialization.
    indices = rng.choice(len(points), size=n_spheres, replace=False)
    points = points[indices]
    
    spans = points.max(axis=0) - points.min(axis=0)
    # Scale to fit in [-0.45, 0.45] initially
    scale = 0.9 / np.max(spans)
    points = points * scale
    points -= points.mean(axis=0)
    
    points += rng.uniform(-jitter, jitter, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points

def _fcc_lattice_init(n_spheres=50, seed=42, jitter=0.02):
    rng = np.random.RandomState(seed)
    # FCC: 4 points per unit cell
    m_x, m_y, m_z = 2, 3, 3
    points = []
    for ix in range(m_x):
        for iy in range(m_y):
            for iz in range(m_z):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz])
                points.append([ix + 0.5, iy, iz + 0.5])
                points.append([ix, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    indices = rng.choice(len(points), size=n_spheres, replace=False)
    points = points[indices]
    
    spans = points.max(axis=0) - points.min(axis=0)
    scale = 0.9 / np.max(spans)
    points = points * scale
    points -= points.mean(axis=0)
    
    points += rng.uniform(-jitter, jitter, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points

def _random_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(n_spheres, 3))

def _optimize_from_init(C_init, deadline):
    """
    Optimize starting from C_init.
    1. Minimize repulsive energy.
    2. Polish with soft-min radius maximization.
    """
    C_curr = C_init.copy()
    best_C = C_init.copy()
    best_r = _exact_radius(C_init)
    
    # Phase 1: Repulsive Energy Minimization
    # This helps separate spheres that might be too close in the lattice init.
    try:
        def energy_obj(x):
            C = x.reshape(-1, 3)
            E, _ = _repulsive_energy_and_grad(C, p=4, wall_softness=0.05)
            return E
            
        def energy_grad(x):
            C = x.reshape(-1, 3)
            _, g = _repulsive_energy_and_grad(C, p=4, wall_softness=0.05)
            return g.flatten()

        res = minimize(
            energy_obj,
            C_curr.flatten(),
            jac=energy_grad,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        C_new = res.x.reshape(-1, 3)
        C_new = np.clip(C_new, -0.5 + 1e-15, 0.5 - 1e-15)
        
        r_new = _exact_radius(C_new)
        if r_new > best_r:
            best_r = r_new
            best_C = C_new.copy()
        C_curr = C_new
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        return best_C, best_r
        
    # Phase 2: Soft-Min Radius Maximization (Polishing)
    # Use increasing beta to sharpen the minimum
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 2.0:
            break
            
        def obj(x, b=beta):
            C = x.reshape(-1, 3)
            val, _ = _soft_min_radius_and_grad(C, b)
            return val
            
        def grad(x, b=beta):
            C = x.reshape(-1, 3)
            _, g = _soft_min_radius_and_grad(C, b)
            return g

        try:
            res = minimize(
                obj,
                C_curr.flatten(),
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            C_new = res.x.reshape(-1, 3)
            C_new = np.clip(C_new, -0.5 + 1e-15, 0.5 - 1e-15)
            
            r_new = _exact_radius(C_new)
            
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
            
            C_curr = C_new
        except Exception:
            pass
            
    return best_C, best_r

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Leave 10s margin for safety

    # Generate multiple initializations
    initializations = []
    
    # BCC Lattices with different jitters
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(_bcc_lattice_init(N, seed=seed, jitter=0.01))
        initializations.append(_bcc_lattice_init(N, seed=seed, jitter=0.05))
        
    # FCC Lattices
    for seed in [10, 11]:
        initializations.append(_fcc_lattice_init(N, seed=seed, jitter=0.02))
        
    # Random
    for seed in [30, 31]:
        initializations.append(_random_init(N, seed=seed))
        
    # Evaluate initializations and pick the best few
    init_radii = []
    for init in initializations:
        r = _exact_radius(init)
        init_radii.append(r)
    
    sorted_indices = np.argsort(init_radii)[::-1]
    
    # We have ~14 initializations.
    # Let's pick the top 6 to optimize fully.
    top_k = 6
    
    best_C = None
    best_r = -np.inf
    
    # Allocate time per initialization
    time_per_init = (deadline - t_start - 5.0) / top_k
    
    for idx in sorted_indices[:top_k]:
        t_init_start = time.time()
        init_deadline = t_init_start + time_per_init
        
        C_init = initializations[idx]
        C_opt, r_opt = _optimize_from_init(C_init, init_deadline)
        
        if r_opt > best_r:
            best_r = r_opt
            best_C = C_opt
            
        # Check if we are running out of time
        if time.time() > deadline - 10.0:
            break
            
    # Final high-precision polish on global best
    if time.time() < deadline - 5.0 and best_C is not None:
        C_curr = best_C.copy()
        best_r_polish = _exact_radius(C_curr)
        
        # High beta steps
        for beta in [2000.0, 5000.0]:
            if time.time() > deadline - 2.0:
                break
                
            def obj(x, b=beta):
                C = x.reshape(-1, 3)
                val, _ = _soft_min_radius_and_grad(C, b)
                return val
                
            def grad(x, b=beta):
                C = x.reshape(-1, 3)
                _, g = _soft_min_radius_and_grad(C, b)
                return g

            try:
                res = minimize(
                    obj,
                    C_curr.flatten(),
                    jac=grad,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 100
                    }
                )
                C_new = res.x.reshape(-1, 3)
                C_new = np.clip(C_new, -0.5 + 1e-15, 0.5 - 1e-15)
                r_new = _exact_radius(C_new)
                
                if r_new > best_r_polish:
                    best_r_polish = r_new
                    C_curr = C_new
            except Exception:
                pass
        
        if best_r_polish > best_r:
            best_r = best_r_polish
            best_C = C_curr
    
    # Ensure validity
    if best_C is None:
        best_C = _bcc_lattice_init(N, seed=1)
    
    best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_C
