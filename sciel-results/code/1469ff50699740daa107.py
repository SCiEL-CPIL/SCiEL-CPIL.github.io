import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of equilateral triangle with side 1

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
n_triples = len(_IDX_TRIPLES)  # 165

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _compute_dets_and_grads(bary):
    """
    Compute all determinants and their gradients with respect to barycentric coords.
    Returns: dets (n_triples,), grad_dets (n_triples, N, 2)
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    
    # Gradient of det with respect to barycentric coords of each point
    # det = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
    # where w = 1-u-v
    
    # For each triple (i,j,k), compute gradient w.r.t. (u_i, v_i, u_j, v_j, u_k, v_k)
    # Then map to the full gradient vector
    
    grad_dets = np.zeros((n_triples, N, 2))
    
    for t in range(n_triples):
        i, j, k = i_idx[t], j_idx[t], k_idx[t]
        
        ui, vi, wi = u[i], v[i], w[i]
        uj, vj, wj = u[j], v[j], w[j]
        uk, vk, wk = u[k], v[k], w[k]
        
        # det = ui*(vj*wk - wj*vk) - vi*(uj*wk - wj*uk) + wi*(uj*vk - vj*uk)
        
        # d(det)/d(ui) = (vj*wk - wj*vk)
        # d(det)/d(vi) = -(uj*wk - wj*uk)
        # d(det)/d(wi) = (uj*vk - vj*uk)
        
        # d(det)/d(uj) = -vi*wk + wi*vk
        # d(det)/d(vj) = ui*wk - wi*uk
        # d(det)/d(wj) = -ui*vk + vi*uk
        
        # d(det)/d(uk) = -vi*wj + wi*vj
        # d(det)/d(vk) = ui*wj - wi*uj
        # d(det)/d(wk) = -ui*vj + vi*uj
        
        # Since w = 1-u-v, dw/du = -1, dw/dv = -1
        # So d(det)/d(ui) in terms of (ui, vi):
        # = (vj*wk - wj*vk) + (-ui*vk + vi*uk)*(-1)
        # Wait, let me be more careful.
        
        # Actually, the determinant is a function of the barycentric coordinates.
        # Let me compute the gradient with respect to the original (u,v) coordinates.
        
        # det = ui*(vj*wk - wj*vk) - vi*(uj*wk - wj*uk) + wi*(uj*vk - vj*uk)
        # where wi = 1-ui-vi, wj = 1-uj-vj, wk = 1-uk-vk
        
        # This is getting complex. Let me use a simpler approach:
        # The area of triangle (Pi, Pj, Pk) in Cartesian coordinates is:
        # 0.5 * |det([Pj-Pi, Pk-Pi])|
        # where Pi, Pj, Pk are Cartesian points.
        
        # In barycentric coordinates, the determinant we computed is related to the area.
        # The gradient can be computed numerically or analytically.
        
        # For now, let's use a numerical gradient for the smooth objective.
        pass
    
    return det, grad_dets

def _objective_smooth(x, beta, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords (u,v,w < 0)
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    excess_u = np.maximum(0, -u)
    penalty += penalty_lambda * np.sum(excess_u**2)
    excess_v = np.maximum(0, -v)
    penalty += penalty_lambda * np.sum(excess_v**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 20
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _local_optimize_nm(x0, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, maxiter=200, gtol=1e-9):
    """Run L-BFGS-B local optimization from x0."""
    try:
        res = minimize(
            _objective_smooth, x0, method='L-BFGS-B',
            args=(50.0,),
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _simulated_annealing(x0, deadline, n_iter=5000, T_start=0.05, T_end=0.001):
    """
    Simulated annealing to explore the landscape.
    x0: initial point (2N,)
    deadline: absolute time deadline
    """
    x = _project_to_bary_simplex(x0.reshape(N, 2)).flatten()
    current_score = _min_area_ratio_from_bary(x.reshape(N, 2)) / (2 * TRI_AREA)
    best_x = x.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Generate a random perturbation
        step_size = 0.05 * (1.0 + 0.5 * np.exp(-it / 1000.0))
        dx = rng.normal(0, step_size, size=2 * N)
        x_new = x + dx
        x_new = _project_to_bary_simplex(x_new.reshape(N, 2)).flatten()
        
        new_score = _min_area_ratio_from_bary(x_new.reshape(N, 2)) / (2 * TRI_AREA)
        
        if new_score > current_score:
            x = x_new
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_x = x_new.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                accept_prob = np.exp(delta / T)
                if rng.random() < accept_prob:
                    x = x_new
                    current_score = new_score
    
    return best_x, best_score

def score(points):
    """Compute the raw score: min triangle area / TRI_AREA."""
    bary = np.zeros((N, 2))
    for i in range(N):
        p = points[i]
        # Convert Cartesian to barycentric
        w = 2 * p[1] / np.sqrt(3)
        v = p[0] - w * 0.5
        u = 1 - v - w
        bary[i] = [u, v]
    
    dets = _all_dets_from_bary(bary)
    min_area = np.min(np.abs(dets))
    return min_area / (2 * TRI_AREA)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(15):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 4. Structured Starts
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    rng3 = np.random.RandomState(123)
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary3 += rng3.normal(0, 0.02, size=bary3.shape)
    inits.append(_project_to_bary_simplex(bary3).flatten())
    
    rng4 = np.random.RandomState(456)
    bary4 = []
    bary4.append([0.0, 0.0])
    bary4.append([1.0, 0.0])
    bary4.append([0.0, 1.0])
    bary4.append([1/3, 1/3])
    for _ in range(7):
        u = rng4.uniform(0.1, 0.9)
        v = rng4.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary4.append([u, v])
        else:
            bary4.append([u * 0.8, v * 0.8])
    inits.append(_project_to_bary_simplex(np.array(bary4[:N])).flatten())
    
    # Optimize each initial point with gradient-based methods
    for init in inits:
        if time_left() < 5.0:
            break
        
        x0 = _project_to_bary_simplex(init.reshape(N, 2)).flatten()
        
        for opt_func in [_local_optimize_nm, _local_optimize_lbfgs, _local_optimize_slsqp]:
            if time_left() < 5.0:
                break
            
            try:
                x_opt = opt_func(x0)
                x_opt = _project_to_bary_simplex(x_opt.reshape(N, 2)).flatten()
                
                bary_opt = x_opt.reshape(N, 2)
                s = _min_area_ratio_from_bary(bary_opt) / (2 * TRI_AREA)
                
                if s > best_score:
                    best_score = s
                    best_x = x_opt.copy()
                    
                x0 = x_opt
            except Exception:
                pass
    
    # Run simulated annealing from the best solution found
    if best_x is not None and time_left() > 10.0:
        sa_x, sa_score = _simulated_annealing(best_x, deadline - 2.0, n_iter=3000)
        if sa_score > best_score:
            best_score = sa_score
            best_x = sa_x
    
    # Final polish with Nelder-Mead
    if best_x is not None and time_left() > 5.0:
        try:
            x_polished = _local_optimize_nm(best_x, maxiter=200)
            x_polished = _project_to_bary_simplex(x_polished.reshape(N, 2)).flatten()
            bary_polished = x_polished.reshape(N, 2)
            s_polished = _min_area_ratio_from_bary(bary_polished) / (2 * TRI_AREA)
            if s_polished > best_score:
                best_score = s_polished
                best_x = x_polished
        except Exception:
            pass
    
    # Convert to Cartesian
    if best_x is not None:
        bary_best = best_x.reshape(N, 2)
        pts_best = _bary_to_cart(bary_best[:, 0], bary_best[:, 1])
        return pts_best
    
    # Fallback
    pts = []
    for _ in range(N):
        u, v = np.random.random(2)
        while u + v > 1:
            u, v = np.random.random(2)
        pts.append([u + 0.5*v, np.sqrt(3)/2*v])
    return np.array(pts)
