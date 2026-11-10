"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145  # 5s safety margin from 150s limit

def _hex_lattice_init(spacing, offset_x=0.0, offset_y=0.0):
    """Generate points on a hexagonal lattice inside the unit circle."""
    points = []
    k = int(1.0 / spacing) + 3
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing + offset_x
            y = j * spacing * np.sqrt(3.0) / 2.0 + offset_y
            if x*x + y*y <= 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii, kind="stable")[:N]
    return points[indices]

def _ring_init(n_rings=5):
    """
    Generate initial positions using concentric rings.
    """
    points = []
    points.append([0.0, 0.0])
    remaining = N - 1
    points_per_ring = []
    for ring in range(1, n_rings + 1):
        if ring == n_rings:
            points_per_ring.append(remaining)
        else:
            count = max(1, int(remaining * (2 * ring) / (n_rings * (n_rings + 1))))
            points_per_ring.append(count)
            remaining -= count
    diff = (N - 1) - sum(points_per_ring)
    if diff != 0:
        points_per_ring[-1] += diff
    for ring, count in enumerate(points_per_ring, 1):
        if count <= 0:
            continue
        r = ring / (n_rings + 0.5)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:N])

def _random_init(seed=42):
    """Generate random initial positions."""
    rng = np.random.RandomState(seed)
    points = []
    points.append([0.0, 0.0])
    for _ in range(N - 1):
        while True:
            r = rng.uniform(0, 0.9)
            theta = rng.uniform(0, 2 * np.pi)
            p = [r * np.cos(theta), r * np.sin(theta)]
            points.append(p)
            break
    return np.array(points[:N])

def _calc_score_fast(C):
    """Faster score calculation using only upper triangle."""
    C = np.asarray(C).reshape(-1, 2)
    if C.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(C)):
        return 0.0
    norms = np.linalg.norm(C, axis=1)
    r_bound = 1.0 - np.max(norms)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_pairs = np.linalg.norm(C[i_idx] - C[j_idx], axis=1)
    r_pair = np.min(d_pairs) / 2.0
    return min(r_bound, r_pair)

def _soft_min_objective_and_grad(x, beta):
    """
    Smooth surrogate for maximizing min(r_bound, r_pair) using soft-min.
    We minimize -softmin where softmin approximates the minimum of constraint values.
    """
    C = x.reshape(N, 2)
    norms = np.linalg.norm(C, axis=1)
    g_boundary = 1.0 - norms  # shape (N,)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    h_pairs = dists[i_idx, j_idx] / 2.0  # shape (N*(N-1)/2,)
    
    all_vals = np.concatenate([g_boundary, h_pairs])
    
    # Soft-min: softmin(v) = -logsumexp(-beta * v) / beta
    neg_beta_v = -beta * all_vals
    max_val = np.max(neg_beta_v)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_v - max_val)))
    softmin_val = -log_sum_exp / beta
    
    obj = -softmin_val
    
    # Gradient calculation
    exp_vals = np.exp(neg_beta_v - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(C)
    
    # Boundary part
    safe_norms = np.where(norms > 1e-10, norms, 1e-10)
    grad_boundary = -C / safe_norms[:, np.newaxis]
    w_boundary = weights[:N]
    grad += (w_boundary[:, np.newaxis] * grad_boundary).sum(axis=0)
    
    # Pairwise part
    w_pairs = weights[N:]
    valid_d = dists[i_idx, j_idx]
    safe_d = np.where(valid_d > 1e-10, valid_d, 1e-10)
    
    d_vec = C[i_idx] - C[j_idx]
    factor_pairs = w_pairs / (2.0 * safe_d)
    
    grad_i = factor_pairs[:, np.newaxis] * d_vec
    grad_j = -factor_pairs[:, np.newaxis] * d_vec
    
    np.add.at(grad, i_idx, grad_i)
    np.add.at(grad, j_idx, grad_j)
    
    return obj, grad.flatten()

def _epigraph_objective_and_grad(x, beta=200.0, lam=10.0):
    """
    Epigraph formulation: maximize t subject to all constraints >= t.
    x = [c_0, c_1, ..., c_{N-1}, t]
    """
    C = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    norms = np.linalg.norm(C, axis=1)
    g_boundary = 1.0 - norms - t  # shape (N,)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = C[i_idx] - C[j_idx]
    d_pairs = np.linalg.norm(diff, axis=1)
    h_pairs = d_pairs / 2.0 - t  # shape (N*(N-1)/2,)
    
    all_violations = np.concatenate([g_boundary, h_pairs])
    
    # Soft-min of violations
    neg_beta_v = -beta * all_violations
    max_val = np.max(neg_beta_v)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_v - max_val)))
    softmin_val = -log_sum_exp / beta
    
    obj = -t - lam * softmin_val
    
    grad = np.zeros_like(x)
    grad[2*N] = -1.0
    
    exp_vals = np.exp(neg_beta_v - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    safe_norms = np.where(norms > 1e-10, norms, 1e-10)
    grad_boundary_c = -C / safe_norms[:, np.newaxis]
    grad_boundary_t = -np.ones(N)
    
    w_boundary = weights[:N]
    grad[:2*N] += (w_boundary[:, np.newaxis] * grad_boundary_c).sum(axis=0)
    grad[2*N] += lam * (w_boundary * grad_boundary_t).sum()
    
    safe_d = np.where(d_pairs > 1e-10, d_pairs, 1e-10)
    factor_pairs = weights[N:] / (2.0 * safe_d)
    
    grad_i = factor_pairs[:, np.newaxis] * diff
    grad_j = -factor_pairs[:, np.newaxis] * diff
    
    np.add.at(grad[:2*N].reshape(N, 2), i_idx, grad_i)
    np.add.at(grad[:2*N].reshape(N, 2), j_idx, grad_j)
    grad[2*N] += lam * (weights[N:] * (-1.0)).sum()
    
    return obj, grad

def _polish(C_start, max_iter=2000):
    """
    Final high-precision polish pass using epigraph formulation and soft-min.
    """
    C = C_start.copy()
    best_score = _calc_score_fast(C)
    
    # Phase 1: Epigraph optimization
    if time.time() - START_TIME < TIME_LIMIT - 10:
        x0 = np.concatenate([C.flatten(), [best_score]])
        
        def fun_ep(x):
            return _epigraph_objective_and_grad(x)[0]
        
        def jac_ep(x):
            return _epigraph_objective_and_grad(x)[1]
        
        try:
            res = minimize(
                fun_ep, 
                x0, 
                jac=jac_ep, 
                method='L-BFGS-B', 
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            C_new = res.x[:2*N].reshape(N, 2)
            score_new = _calc_score_fast(C_new)
            if score_new > best_score:
                C = C_new
                best_score = score_new
        except Exception:
            pass
    
    # Phase 2: Soft-min polish with increasing beta
    beta_values = [100.0, 300.0, 600.0, 1200.0]
    
    for beta in beta_values:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        def fun(x):
            return _soft_min_objective_and_grad(x, beta)[0]
        
        def jac(x):
            return _soft_min_objective_and_grad(x, beta)[1]
        
        try:
            res = minimize(
                fun, 
                C.flatten(), 
                jac=jac, 
                method='L-BFGS-B', 
                options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            C_new = res.x.reshape(N, 2)
            score_new = _calc_score_fast(C_new)
            if score_new > best_score:
                C = C_new
                best_score = score_new
        except Exception:
            pass
    
    return C

def construct():
    # Generate multiple initial configurations
    inits = []
    
    # Hex lattice with various spacings and offsets
    spacings = [0.24, 0.25, 0.26, 0.27]
    offsets = [(0,0), (0.01, 0.01), (-0.01, 0.01), (0.01, -0.01)]
    for s in spacings:
        for ox, oy in offsets:
            pts = _hex_lattice_init(s, ox, oy)
            if pts is not None and len(pts) == N:
                inits.append(pts)
    
    # Ring inits
    inits.append(_ring_init(5))
    inits.append(_ring_init(4))
    
    # Random init
    inits.append(_random_init(42))
    
    # Evaluate all initial configs and pick the best
    best_C = None
    best_score = -np.inf
    for init in inits:
        s = _calc_score_fast(init)
        if s > best_score:
            best_score = s
            best_C = init.copy()
    
    # Phase 1: Direct optimization using epigraph formulation
    if time.time() - START_TIME < TIME_LIMIT - 40:
        x0 = np.concatenate([best_C.flatten(), [best_score]])
        
        def fun(x):
            return _epigraph_objective_and_grad(x, beta=100.0, lam=5.0)[0]
        
        def jac(x):
            return _epigraph_objective_and_grad(x, beta=100.0, lam=5.0)[1]
        
        try:
            res = minimize(
                fun, 
                x0, 
                jac=jac, 
                method='L-BFGS-B', 
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            C_new = res.x[:2*N].reshape(N, 2)
            score_new = _calc_score_fast(C_new)
            if score_new > best_score:
                best_score = score_new
                best_C = C_new.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start refinement from best config with adaptive perturbation
    if time.time() - START_TIME < TIME_LIMIT - 15:
        rng = np.random.RandomState(123)
        perturb_scales = [0.01, 0.005, 0.002, 0.001]
        
        for attempt, scale in enumerate(perturb_scales):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            C_perturbed = best_C + rng.normal(0, scale, size=best_C.shape)
            
            x0 = np.concatenate([C_perturbed.flatten(), [best_score]])
            
            def fun(x):
                return _epigraph_objective_and_grad(x, beta=200.0, lam=10.0)[0]
            
            def jac(x):
                return _epigraph_objective_and_grad(x, beta=200.0, lam=10.0)[1]
            
            try:
                res = minimize(
                    fun, 
                    x0, 
                    jac=jac, 
                    method='L-BFGS-B', 
                    options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                C_new = res.x[:2*N].reshape(N, 2)
                score_new = _calc_score_fast(C_new)
                if score_new > best_score:
                    best_score = score_new
                    best_C = C_new.copy()
            except Exception:
                pass
    
    # Phase 3: Final polish with epigraph + soft-min
    if time.time() - START_TIME < TIME_LIMIT - 5:
        C_polished = _polish(best_C, max_iter=2000)
        score_polished = _calc_score_fast(C_polished)
        if score_polished > best_score:
            best_score = score_polished
            best_C = C_polished
    
    return best_C

# EVOLVE-BLOCK-END
