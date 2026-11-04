"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples (C(11,3) = 165)
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = _IDX_TRIPLES.shape[0]

def _bary_to_cart(bary):
    """Convert barycentric (u, v) to Cartesian. w = 1-u-v. P = u*V0 + v*V1 + w*V2."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack([x, y])

def _all_dets_from_bary(bary):
    """Compute all 165 determinants (twice the area in barycentric coords, scaled by TRI_AREA)."""
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

def _min_area_ratio(bary):
    """Returns the minimum |det| (proportional to area / TRI_AREA)."""
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _score_bary(bary):
    """Score: minimum triangle area (in actual units)."""
    bary = _clamp_bary(bary)
    return _min_area_ratio(bary) * TRI_AREA

def _clamp_bary(bary):
    """Project to valid barycentric simplex: u>=0, v>=0, u+v<=1."""
    u = np.maximum(0.0, bary[:, 0])
    v = np.maximum(0.0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack([u, v])

def _softmin_objective(x, beta=120.0):
    """Smooth surrogate: negative soft-min of |dets| using log-sum-exp, with floor penalty."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_a - log_sum / beta
    
    # Penalize any area that is extremely small (near-degenerate)
    floor_penalty = 1000.0 * np.sum(np.maximum(0.0, 1e-6 - areas) ** 2)
    
    return -soft_min + floor_penalty

def _softmin_grad(x, beta=120.0):
    """Gradient of the soft-min objective w.r.t. barycentric coordinates."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
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
    
    # det = Pi[0]*(Pj[1]*Pk[2]-Pj[2]*Pk[1]) - Pi[1]*(Pj[0]*Pk[2]-Pj[2]*Pk[0]) + Pi[2]*(Pj[0]*Pk[1]-Pj[1]*Pk[0])
    dets = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    areas = np.abs(dets)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    # Weights for soft-min: w_i = exp(-beta * (a_i - min_a)) / sum(exp(-beta*(a_j - min_a)))
    exp_vals = np.exp(-beta * shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    # d(soft_min)/d(area_i) = -w_i (since soft_min = min_a - log_sum/beta, and d log_sum/d area_i = -beta * w_i)
    # So d(-soft_min)/d(area_i) = w_i
    # d(area_i)/d(det_i) = sign(det_i)
    # d(det_i)/d(bary) involves the cross product structure
    
    # Gradient of det w.r.t. each point's barycentric coords
    # det = Pi x Pj · Pk (scalar triple product in 3D barycentric space)
    # d(det)/d(Pi) = Pj × Pk
    # d(det)/d(Pj) = Pk × Pi
    # d(det)/d(Pk) = Pi × Pj
    
    grad_det = np.zeros((N, 3))
    for t in range(N_TRIPLES):
        i, j, k = i_idx[t], j_idx[t], k_idx[t]
        # Pj × Pk
        cross_jk = np.cross(Pj[t], Pk[t])
        grad_det[i] += cross_jk
        # Pk × Pi
        cross_ki = np.cross(Pk[t], Pi[t])
        grad_det[j] += cross_ki
        # Pi × Pj
        cross_ij = np.cross(Pi[t], Pj[t])
        grad_det[k] += cross_ij
    
    # Now: d(-soft_min)/d(bary_coord) = sum_t w_t * sign(det_t) * d(det_t)/d(bary_coord)
    signs = np.sign(dets)
    weighted_grad = weights * signs  # shape (N_TRIPLES,)
    
    # Accumulate gradient w.r.t. barycentric coords (u, v) for each point
    # bary = (u, v), and P = (u, v, 1-u-v)
    # dP/du = (1, 0, -1), dP/dv = (0, 1, -1)
    grad_uv = np.zeros((N, 2))
    for t in range(N_TRIPLES):
        i, j, k = i_idx[t], j_idx[t], k_idx[t]
        wt = weighted_grad[t]
        # For point i: d(det)/du_i = grad_det[i][0] - grad_det[i][2]
        #              d(det)/dv_i = grad_det[i][1] - grad_det[i][2]
        grad_uv[i, 0] += wt * (grad_det[i, 0] - grad_det[i, 2])
        grad_uv[i, 1] += wt * (grad_det[i, 1] - grad_det[i, 2])
        grad_uv[j, 0] += wt * (grad_det[j, 0] - grad_det[j, 2])
        grad_uv[j, 1] += wt * (grad_det[j, 1] - grad_det[j, 2])
        grad_uv[k, 0] += wt * (grad_det[k, 0] - grad_det[k, 2])
        grad_uv[k, 1] += wt * (grad_det[k, 1] - grad_det[k, 2])
    
    return grad_uv.flatten()

def _hard_objective(x):
    """Hard objective: negative min |det|."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    dets = _all_dets_from_bary(bary)
    return -np.min(np.abs(dets))

def _structured_init_bary(seed=42):
    """Generate structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    patterns = []
    
    # Pattern A: Symmetric with vertices, edge midpoints, centroid
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
    ])
    patterns.append(pA)
    
    # Pattern B: Hexagonal-ish
    pB = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.33, 0.33],
        [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
    ])
    patterns.append(pB)
    
    # Pattern C: More spread
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Concentric hexagons (from inspiration)
    pD = np.array([
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
    patterns.append(pD)
    
    # Pattern E: Magic init (from inspiration)
    pE = np.array([
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ])
    patterns.append(pE)
    
    # Pattern F: Hexagonal lattice (from inspiration)
    pF = np.array([
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
    patterns.append(pF)
    
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    jitter = rng.normal(0, 0.015, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    return bary

def _fps_init():
    """Farthest Point Sampling on a grid of barycentric coordinates."""
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
        dists = np.min(np.linalg.norm(candidates[:, None, :] - np.array(selected)[None, :, :], axis=2), axis=1)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _random_init_bary(seed):
    """Random initialization in barycentric coords."""
    rng = np.random.RandomState(seed)
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                bary[i] = [u, v]
                break
    return bary

def _l_bfgs_softmin(bary0, maxiter=400, beta=120.0):
    """L-BFGS-B optimization using soft-min surrogate with analytical gradient."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        def obj(x):
            return _softmin_objective(x, beta=beta)
        
        def grad(x):
            return _softmin_grad(x, beta=beta)
        
        res = minimize(
            obj, x0, method='L-BFGS-B',
            jac=grad,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _l_bfgs_hard(bary0, maxiter=200):
    """L-BFGS-B optimization using hard min."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _hard_objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _slsqp_refine(bary0, maxiter=200):
    """SLSQP refinement with explicit constraint u+v<=1."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        # Constraints: for each point, u_i + v_i <= 1
        cons = []
        for i in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1],
                'jac': lambda x, i=i: np.concatenate([np.zeros(2*i), [-1.0, -1.0], np.zeros(2*(N-i-1))])
            })
        
        res = minimize(
            _hard_objective, x0, method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'eps': 1e-8}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _active_set_refine(bary0, maxiter=150):
    """Refine by identifying active (binding) triples and equalizing them via penalty."""
    try:
        bary0 = _clamp_bary(bary0)
        dets = _all_dets_from_bary(bary0)
        areas = np.abs(dets)
        min_area = np.min(areas)
        
        tol = max(0.03 * min_area, 1e-8)
        active_mask = areas <= min_area + tol
        active_count = np.sum(active_mask)
        
        if active_count < 2:
            return bary0.copy()
        
        target = min_area
        
        def penalty_obj(x):
            bary = x.reshape(N, 2)
            bary = _clamp_bary(bary)
            d = _all_dets_from_bary(bary)
            a = np.abs(d)
            min_a = np.min(a)
            below_target = np.maximum(0, target - a)
            penalty_strong = 500.0 * np.sum(below_target ** 2)
            active_dev = a[active_mask] - target
            penalty_equalize = 50.0 * np.sum(active_dev ** 2)
            return -min_a + penalty_strong + penalty_equalize
        
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        res = minimize(
            penalty_obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        
        score_opt = _score_bary(bary_opt)
        score_orig = _score_bary(bary0)
        if score_opt > score_orig:
            return bary_opt
        return bary0.copy()
    except Exception:
        return bary0.copy()

def _differential_evolution_global(seed=42, maxiter=40, popsize=14):
    """Differential evolution for global search using soft-min objective."""
    try:
        bounds = [(0, 1)] * (2 * N)
        
        def de_obj(x):
            return _softmin_objective(x, beta=80.0)
        
        de_result = differential_evolution(
            de_obj,
            bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            updating='deferred'
        )
        
        bary_opt = _clamp_bary(de_result.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return None

def _nelder_mead_optimize(bary0, maxiter=200):
    """Nelder-Mead for local refinement."""
    try:
        x0 = bary0.flatten()
        res = minimize(
            _hard_objective, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _epigraph_refine(bary0, maxiter=300):
    """Epigraph formulation: maximize t subject to all |det| >= t, using smooth barrier."""
    try:
        bary0 = _clamp_bary(bary0)
        dets0 = _all_dets_from_bary(bary0)
        t0 = np.min(np.abs(dets0))
        
        def epigraph_obj(x):
            # x = [bary (2N), t]
            bary = x[:2*N].reshape(N, 2)
            bary = _clamp_bary(bary)
            t = x[2*N]
            dets = _all_dets_from_bary(bary)
            areas = np.abs(dets)
            
            # Soft barrier: penalize areas below t
            violations = np.maximum(0.0, t - areas)
            barrier = 100.0 * np.sum(violations ** 2)
            
            # Also penalize t being too high relative to the actual min
            actual_min = np.min(areas)
            t_penalty = 10.0 * max(0.0, t - actual_min) ** 2
            
            return -t + barrier + t_penalty
        
        x0 = np.concatenate([bary0.flatten(), [t0]])
        bounds = [(0, 1)] * (2 * N) + [(0, 1.0)]
        
        res = minimize(
            epigraph_obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        bary_opt = _clamp_bary(res.x[:2*N].reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _simulated_annealing(bary0, deadline, max_iter=5000, t_start=0.05, t_end=0.0005):
    """Simulated annealing with adaptive step size to escape local optima."""
    try:
        bary = bary0.copy()
        current_score = _score_bary(bary)
        best_bary = bary.copy()
        best_score = current_score
        
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        
        n_iter = max_iter
        for i in range(n_iter):
            if time.time() > deadline - 2.0:
                break
            
            # Exponential cooling schedule
            t = t_start * (t_end / t_start) ** (i / max(1, n_iter - 1))
            
            # Adaptive perturbation scale: larger when temperature is high
            base_scale = 0.015
            scale = base_scale * (0.5 + t / t_start)
            # Occasionally take a larger jump
            if rng.random() < 0.1:
                scale *= 3.0
            
            pert = rng.normal(0, scale, size=(N, 2))
            new_bary = _clamp_bary(bary + pert)
            new_score = _score_bary(new_bary)
            
            # Accept if better or with probability exp((new-old)/t)
            if new_score >= current_score:
                bary = new_bary
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_bary = new_bary.copy()
            else:
                delta = new_score - current_score
                # Avoid overflow in exp
                if delta / max(t, 1e-10) > -50:
                    if rng.random() < np.exp(delta / max(t, 1e-10)):
                        bary = new_bary
                        current_score = new_score
        
        return best_bary
    except Exception:
        return bary0.copy()

def _polish_high_precision(bary0, maxiter=300):
    """High-precision polish: multi-scale beta annealing (50→2000) + epigraph + SLSQP."""
    try:
        candidates = []
        bary_curr = bary0.copy()
        
        # Multi-scale beta annealing: progressively increase beta
        betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
        iters_per_stage = max(30, maxiter // len(betas))
        
        for beta in betas:
            bary_next = _l_bfgs_softmin(bary_curr, maxiter=iters_per_stage, beta=beta)
            score_next = _score_bary(bary_next)
            candidates.append((bary_next, score_next))
            if score_next > _score_bary(bary_curr):
                bary_curr = bary_next
        
        # Epigraph refinement
        bary_epi = _epigraph_refine(bary_curr, maxiter=maxiter // 3)
        score_epi = _score_bary(bary_epi)
        candidates.append((bary_epi, score_epi))
        if score_epi > _score_bary(bary_curr):
            bary_curr = bary_epi
        
        # Hard min polish
        bary_hard = _l_bfgs_hard(bary_curr, maxiter=maxiter // 4)
        score_hard = _score_bary(bary_hard)
        candidates.append((bary_hard, score_hard))
        if score_hard > _score_bary(bary_curr):
            bary_curr = bary_hard
        
        # SLSQP final
        bary_slsqp = _slsqp_refine(bary_curr, maxiter=maxiter // 4)
        score_slsqp = _score_bary(bary_slsqp)
        candidates.append((bary_slsqp, score_slsqp))
        
        # Nelder-Mead fine tune
        bary_nm = _nelder_mead_optimize(bary_curr, maxiter=100)
        score_nm = _score_bary(bary_nm)
        candidates.append((bary_nm, score_nm))
        
        best = max(candidates, key=lambda c: c[1])
        return best[0]
    except Exception:
        return bary0.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    # Phase 1: Multi-start with diverse structured inits
    initial_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in initial_seeds:
        if time_left() < 5.0:
            break
        
        bary_init = _structured_init_bary(seed=seed)
        score_init = _score_bary(bary_init)
        
        if time_left() > 4.0:
            bary_opt = _l_bfgs_softmin(bary_init, maxiter=350, beta=120.0)
            score_opt = _score_bary(bary_opt)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        if time_left() > 3.0:
            bary_polish = _l_bfgs_hard(bary_init, maxiter=200)
            score_polish = _score_bary(bary_polish)
            if score_polish > score_init:
                score_init = score_polish
                bary_init = bary_polish
        
        if score_init > best_score:
            best_score = score_init
            best_bary = bary_init.copy()
    
    # Phase 2: FPS init
    if time_left() > 8.0:
        bary_fps = _fps_init()
        score_fps = _score_bary(bary_fps)
        
        if time_left() > 5.0:
            bary_opt = _l_bfgs_softmin(bary_fps, maxiter=300, beta=120.0)
            score_opt = _score_bary(bary_opt)
            if score_opt > score_fps:
                score_fps = score_opt
                bary_fps = bary_opt
        
        if time_left() > 3.0:
            bary_polish = _l_bfgs_hard(bary_fps, maxiter=200)
            score_polish = _score_bary(bary_polish)
            if score_polish > score_fps:
                score_fps = score_polish
                bary_fps = bary_polish
        
        if score_fps > best_score:
            best_score = score_fps
            best_bary = bary_fps.copy()
    
    # Phase 3: Differential evolution for global search
    if time_left() > 15.0:
        de_bary = _differential_evolution_global(seed=42, maxiter=35, popsize=12)
        if de_bary is not None:
            score_de = _score_bary(de_bary)
            if score_de > best_score:
                best_score = score_de
                best_bary = de_bary.copy()
                
                if time_left() > 8.0:
                    bary_ref = _polish_high_precision(de_bary, maxiter=250)
                    score_ref = _score_bary(bary_ref)
                    if score_ref > best_score:
                        best_score = score_ref
                        best_bary = bary_ref.copy()
    
    # Phase 4: Multi-start with random initializations
    rng = np.random.RandomState(123)
    for rs in range(8):
        if time_left() < 5.0:
            break
        
        bary_rand = _random_init_bary(seed=1000 + rs)
        score_rand = _score_bary(bary_rand)
        
        if time_left() > 4.0:
            bary_opt = _l_bfgs_softmin(bary_rand, maxiter=300, beta=120.0)
            score_opt = _score_bary(bary_opt)
            if score_opt > score_rand:
                score_rand = score_opt
                bary_rand = bary_opt
        
        if time_left() > 3.0:
            bary_polish = _l_bfgs_hard(bary_rand, maxiter=200)
            score_polish = _score_bary(bary_polish)
            if score_polish > score_rand:
                score_rand = score_polish
                bary_rand = bary_polish
        
        if score_rand > best_score:
            best_score = score_rand
            best_bary = bary_rand.copy()
    
    # Phase 5: Simulated annealing to explore landscape (longer, more iterations)
    if time_left() > 12.0:
        sa_budget = min(5000, int((time_left() - 2.0) * 200))  # ~200 iters/sec budget
        bary_sa = _simulated_annealing(best_bary, deadline, max_iter=sa_budget, t_start=0.04, t_end=0.0003)
        score_sa = _score_bary(bary_sa)
        if score_sa > best_score:
            best_score = score_sa
            best_bary = bary_sa.copy()
        
        # After SA, do a quick polish on the SA result
        if score_sa > best_score and time_left() > 5.0:
            bary_sa_polish = _l_bfgs_softmin(best_bary, maxiter=150, beta=200.0)
            score_sa_polish = _score_bary(bary_sa_polish)
            if score_sa_polish > best_score:
                best_score = score_sa_polish
                best_bary = bary_sa_polish.copy()
    
    # Phase 6: Active-set refinement
    if time_left() > 5.0:
        bary_as = _active_set_refine(best_bary, maxiter=150)
        score_as = _score_bary(bary_as)
        if score_as > best_score:
            best_score = score_as
            best_bary = bary_as.copy()
    
    # Phase 7: Multi-start perturbations around best (with multi-beta polish for promising candidates)
    restart_count = 0
    max_restarts = 50
    while restart_count < max_restarts and time_left() > 4.0:
        restart_count += 1
        
        scale = 0.025 * (1.0 - 0.6 * restart_count / max_restarts) + 0.008
        pert = rng.normal(0, scale, size=(N, 2))
        bary_start = _clamp_bary(best_bary + pert)
        
        if time_left() > 3.5:
            # Use a moderate beta first
            bary_opt = _l_bfgs_softmin(bary_start, maxiter=200, beta=150.0)
            score_opt = _score_bary(bary_opt)
            
            if score_opt > best_score and time_left() > 2.5:
                # Higher beta refinement
                bary_opt2 = _l_bfgs_softmin(bary_opt, maxiter=150, beta=400.0)
                score_opt2 = _score_bary(bary_opt2)
                if score_opt2 > score_opt:
                    score_opt = score_opt2
                    bary_opt = bary_opt2
            
            if score_opt > best_score and time_left() > 2.0:
                bary_polish = _l_bfgs_hard(bary_opt, maxiter=120)
                score_polish = _score_bary(bary_polish)
                if score_polish > score_opt:
                    score_opt = score_polish
                    bary_opt = bary_polish
            
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
                
                # For significantly improved candidates, try epigraph refinement
                if time_left() > 3.0:
                    bary_epi = _epigraph_refine(best_bary, maxiter=100)
                    score_epi = _score_bary(bary_epi)
                    if score_epi > best_score:
                        best_score = score_epi
                        best_bary = bary_epi.copy()
    
    # Phase 8: Nelder-Mead fine-tuning
    if time_left() > 5.0:
        bary_nm = _nelder_mead_optimize(best_bary, maxiter=150)
        score_nm = _score_bary(bary_nm)
        if score_nm > best_score:
            best_score = score_nm
            best_bary = bary_nm.copy()
    
    # Phase 9: High-precision final polish (multi-scale beta + SLSQP + epigraph)
    if time_left() > 10.0:
        bary_hp = _polish_high_precision(best_bary, maxiter=500)
        score_hp = _score_bary(bary_hp)
        if score_hp > best_score:
            best_score = score_hp
            best_bary = bary_hp.copy()
        
        # Extra: one more round of active-set + SLSQP if time permits
        if time_left() > 5.0:
            bary_as2 = _active_set_refine(best_bary, maxiter=100)
            score_as2 = _score_bary(bary_as2)
            if score_as2 > best_score:
                best_score = score_as2
                best_bary = bary_as2.copy()
            
            if time_left() > 4.0:
                bary_slsqp2 = _slsqp_refine(best_bary, maxiter=150)
                score_slsqp2 = _score_bary(bary_slsqp2)
                if score_slsqp2 > best_score:
                    best_score = score_slsqp2
                    best_bary = bary_slsqp2.copy()
    
    # Phase 10: Final active-set + SLSQP + epigraph combo
    if time_left() > 5.0:
        bary_as = _active_set_refine(best_bary, maxiter=100)
        score_as = _score_bary(bary_as)
        if score_as > best_score:
            best_score = score_as
            best_bary = bary_as.copy()
        
        if time_left() > 4.0:
            bary_epi = _epigraph_refine(best_bary, maxiter=150)
            score_epi = _score_bary(bary_epi)
            if score_epi > best_score:
                best_score = score_epi
                best_bary = bary_epi.copy()
        
        if time_left() > 3.0:
            bary_slsqp = _slsqp_refine(best_bary, maxiter=100)
            score_slsqp = _score_bary(bary_slsqp)
            if score_slsqp > best_score:
                best_score = score_slsqp
                best_bary = bary_slsqp.copy()
    
    # Final fallback
    if best_bary is None:
        best_bary = _structured_init_bary(seed=42)
    
    return _bary_to_cart(best_bary)
