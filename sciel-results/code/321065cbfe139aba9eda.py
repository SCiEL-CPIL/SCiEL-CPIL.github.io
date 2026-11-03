import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Leave 5s margin from 150s

def _score(c):
    """Compute the radius for a valid configuration."""
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5 - 1e-9) or np.any(c > 0.5 + 1e-9):
        return 0.0
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = c[i_idx] - c[j_idx]
    pair_dists = 0.5 * np.linalg.norm(diffs, axis=1)
    min_pair = np.min(pair_dists)
    return min(min_wall, min_pair)

def _soft_min_radius(c, beta):
    """
    Compute soft-min radius and gradient.
    Returns (negative_soft_min_radius, negative_gradient) for minimization.
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(c)  # (N, 3)
    
    # Pair distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = c[i_idx] - c[j_idx]
    norms = np.linalg.norm(diffs, axis=1)
    norms_safe = np.where(norms < 1e-12, 1e-12, norms)
    pair_dists = 0.5 * norms
    
    # Combine all distances
    all_dists = np.concatenate([wall_dists.flatten(), pair_dists])
    
    # Soft min via log-sum-exp
    m = np.min(all_dists)
    dm = all_dists - m
    weights = np.exp(-beta * dm)
    S = np.sum(weights)
    w = weights / S
    
    # Compute gradient
    grad = np.zeros(3 * N)
    
    # Wall gradient: d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
    w_wall = w[:3*N].reshape(N, 3)
    grad_wall = -np.sign(c) * w_wall
    grad += grad_wall.flatten()
    
    # Pair gradient
    w_pair = w[3*N:]
    unit_diffs = diffs / norms_safe[:, np.newaxis]
    
    # Accumulate gradients for pairs
    np.add.at(grad, i_idx * 3, 0.5 * w_pair * unit_diffs[:, 0])
    np.add.at(grad, i_idx * 3 + 1, 0.5 * w_pair * unit_diffs[:, 1])
    np.add.at(grad, i_idx * 3 + 2, 0.5 * w_pair * unit_diffs[:, 2])
    np.add.at(grad, j_idx * 3, -0.5 * w_pair * unit_diffs[:, 0])
    np.add.at(grad, j_idx * 3 + 1, -0.5 * w_pair * unit_diffs[:, 1])
    np.add.at(grad, j_idx * 3 + 2, -0.5 * w_pair * unit_diffs[:, 2])
    
    r_soft = m - (1.0 / beta) * np.log(S)
    return -r_soft, -grad

def _optimize_from_init(x0, beta_schedule, deadline):
    """Run optimization with a given initial point and beta schedule."""
    best_x = x0.copy()
    best_score = _score(x0)
    
    for beta in beta_schedule:
        if time.time() > deadline - 5:
            break
        
        def obj(x):
            return _soft_min_radius(x.reshape(N, 3), beta)[0]
        
        def jac(x):
            return _soft_min_radius(x.reshape(N, 3), beta)[1]
        
        try:
            res = minimize(obj, best_x, jac=jac, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (3 * N),
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            new_score = _score(res.x)
            if new_score > best_score:
                best_x = res.x
                best_score = new_score
        except Exception:
            pass
    
    return best_x, best_score

def _grid_init(perturb=0.0):
    """Generate a grid-based initial configuration."""
    # 5x5x4 grid
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    centers = []
    for z in zs:
        for y in ys:
            for x in xs:
                centers.append([x, y, z])
    centers = np.array(centers)
    
    if perturb > 0:
        rng = np.random.default_rng(42)
        centers += rng.normal(0, perturb, size=centers.shape)
        # Clip to stay in bounds
        centers = np.clip(centers, -0.49, 0.49)
    
    return centers.flatten()

def _random_init(seed):
    """Generate a random initial configuration."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centers.flatten()

def construct():
    start_time = time.time()
    
    # Start with grid initialization
    best_x = _grid_init(perturb=0.0)
    best_score = _score(best_x)
    
    # Beta schedule for first optimization
    beta_schedule = [20, 100, 500, 2000, 5000]
    best_x, best_score = _optimize_from_init(best_x, beta_schedule, DEADLINE)
    
    # Try several random restarts if time permits
    for trial in range(5):
        if time.time() > DEADLINE - 10:
            break
        
        x0 = _random_init(seed=100 + trial)
        x_opt, score_opt = _optimize_from_init(x0, [50, 200, 1000], DEADLINE)
        
        if score_opt > best_score:
            best_x = x_opt
            best_score = score_opt
    
    # Final polish with very high beta if time permits
    if time.time() < DEADLINE - 5:
        beta = 10000
        def obj_final(x):
            return _soft_min_radius(x.reshape(N, 3), beta)[0]
        def jac_final(x):
            return _soft_min_radius(x.reshape(N, 3), beta)[1]
        
        try:
            res = minimize(obj_final, best_x, jac=jac_final, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (3 * N),
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            new_score = _score(res.x)
            if new_score > best_score:
                best_x = res.x
                best_score = new_score
        except Exception:
            pass
    
    return best_x.reshape(N, 3)
