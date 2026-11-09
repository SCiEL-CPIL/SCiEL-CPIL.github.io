# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _clip_and_normalize(P):
    """Clip negative values to 0 and normalize to sum 1."""
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return np.ones((10, 10)) / 100.0
    return P / total


def _compute_ratio(P):
    """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y
    
    # D: H of x - y (slope -1)
    vals_d = X - Y  # values from -9 to 9
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (vals_d == v - 9)
        counts_d[v] = P[mask].sum()
    H_d = 0.0
    for c in counts_d:
        if c > 0:
            H_d -= c * np.log2(c)
    
    # M: max of H for slopes 0, 1, 2
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        min_val = 0
        max_val = (n - 1) + r * (n - 1)
        counts = np.zeros(max_val + 1)
        for v in range(max_val + 1):
            mask = (vals == v)
            counts[v] = P[mask].sum()
        h = 0.0
        for c in counts:
            if c > 0:
                h -= c * np.log2(c)
        max_h = max(max_h, h)
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h


def _exp_family_P(params):
    """
    Build P from 4-param exponential family: P(x,y) ∝ exp(αx + βy + γxy + δ).
    params = [α, β, γ, δ]
    """
    n = 10
    alpha, beta, gamma, delta = params
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    log_P = alpha * X + beta * Y + gamma * X * Y + delta
    # Shift to avoid underflow/overflow
    log_P -= log_P.max()
    P = np.exp(log_P)
    P = _clip_and_normalize(P)
    return P


def _neg_ratio_exp(params):
    """Negative ratio for minimization of exponential family params."""
    P = _exp_family_P(params)
    return -_compute_ratio(P)


def _optimize_exp_family(deadline):
    """
    Optimize the 4-parameter exponential family using differential evolution + L-BFGS-B.
    Returns best P and ratio found.
    """
    start_time = time.time()
    
    # Define bounds for differential evolution
    bounds = [(-3.0, 3.0), (-3.0, 3.0), (-2.0, 2.0), (-1.0, 1.0)]
    
    best_P = None
    best_ratio = 0.0
    
    # Phase A: Differential evolution (global search)
    remaining = deadline - 60 - (time.time() - start_time)
    if remaining > 30:
        try:
            de_result = differential_evolution(
                _neg_ratio_exp,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-10,
                popsize=15,
                polish=False,
                init='sobol'
            )
            P_de = _exp_family_P(de_result.x)
            ratio_de = _compute_ratio(P_de)
            if ratio_de > best_ratio:
                best_ratio = ratio_de
                best_P = P_de.copy()
        except Exception:
            pass
    
    # Phase B: L-BFGS-B refinement from multiple starting points
    inits = []
    if best_P is not None:
        # Reverse-engineer approximate params from best_P (use de_result if available)
        pass
    
    # Add diverse initializations
    inits.extend([
        [0.1, 0.1, 0.0, 0.0],
        [-0.1, -0.1, 0.0, 0.0],
        [0.2, -0.2, 0.0, 0.0],
        [-0.2, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.0, -0.3, 0.0],
        [0.1, 0.1, 0.2, 0.0],
        [0.1, 0.1, -0.2, 0.0],
        [-0.1, -0.1, 0.2, 0.0],
        [-0.1, -0.1, -0.2, 0.0],
        [0.3, 0.3, 0.1, 0.0],
        [-0.3, -0.3, 0.1, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [-0.5, -0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, -0.5, 0.0],
        [0.2, 0.1, 0.1, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [-0.2, -0.1, 0.1, 0.0],
        [-0.1, -0.2, 0.1, 0.0],
    ])
    
    for init in inits:
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            result = minimize(
                _neg_ratio_exp,
                np.array(init),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            P_opt = _exp_family_P(result.x)
            ratio_opt = _compute_ratio(P_opt)
            
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_P = P_opt.copy()
        except Exception:
            pass
    
    # Phase C: More DE runs with different seeds
    for seed in [1, 7, 13, 23]:
        if time.time() - start_time > deadline - 60:
            break
        try:
            de_result = differential_evolution(
                _neg_ratio_exp,
                bounds,
                seed=seed,
                maxiter=40,
                tol=1e-10,
                popsize=12,
                polish=False,
                init='latinhypercube'
            )
            P_de = _exp_family_P(de_result.x)
            ratio_de = _compute_ratio(P_de)
            if ratio_de > best_ratio:
                best_ratio = ratio_de
                best_P = P_de.copy()
        except Exception:
            pass
    
    if best_P is None:
        return np.ones((10, 10)) / 100.0, 0.0
    
    return best_P, best_ratio


def _compute_ratio_from_d(d):
    """Compute D/M ratio from anti-diagonal weights d."""
    n = 10
    P = np.zeros((n, n))
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    vals_d = X - Y
    
    diag_counts = np.zeros(19)
    for k in range(19):
        diag_counts[k] = np.sum(vals_d == k - 9)
    
    d_norm = d / d.sum()
    
    for k in range(19):
        if d_norm[k] > 0:
            mask = (vals_d == k - 9)
            P[mask] = d_norm[k] / diag_counts[k]
    
    total = P.sum()
    if total <= 1e-9:
        return 0.0
    P = P / total
    
    H_d = 0.0
    for k in range(19):
        if d_norm[k] > 0:
            H_d -= d_norm[k] * np.log2(d_norm[k])
    
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        min_val = 0
        max_val = (n - 1) + r * (n - 1)
        counts = np.zeros(max_val + 1)
        for v in range(max_val + 1):
            mask = (vals == v)
            counts[v] = P[mask].sum()
        h = 0.0
        for c in counts:
            if c > 0:
                h -= c * np.log2(c)
        max_h = max(max_h, h)
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h


def _d_to_P(d):
    """Convert anti-diagonal weights to full P matrix."""
    n = 10
    P = np.zeros((n, n))
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    vals_d = X - Y
    
    diag_counts = np.zeros(19)
    for k in range(19):
        diag_counts[k] = np.sum(vals_d == k - 9)
    
    d_norm = d / d.sum()
    
    for k in range(19):
        if d_norm[k] > 0 and diag_counts[k] > 0:
            mask = (vals_d == k - 9)
            P[mask] = d_norm[k] / diag_counts[k]
    
    total = P.sum()
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / 100.0
    
    return P


def _optimize_anti_diag(d0, deadline):
    """Optimize anti-diagonal weights using L-BFGS-B."""
    start_time = time.time()
    
    # Initial guess
    d = np.ones(19) / 19.0
    best_d = d.copy()
    best_ratio = _compute_ratio_from_d(d)
    
    # Try multiple initializations
    inits = [
        np.ones(19) / 19.0,
        np.array([1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1], dtype=float),
        np.array([1,1,1,1,1,10,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=float),
        np.array([10,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,10], dtype=float),
        np.array([1,0.1,0.1,0.1,0.1,5,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1], dtype=float),
    ]
    
    for init in inits:
        if time.time() - start_time > deadline - 60:
            break
        
        d_init = init / init.sum()
        
        try:
            result = minimize(
                lambda d: -_compute_ratio_from_d(d),
                d_init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            d_opt = result.x
            d_opt = np.clip(d_opt, 1e-10, None)
            d_opt = d_opt / d_opt.sum()
            
            ratio_opt = _compute_ratio_from_d(d_opt)
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_d = d_opt.copy()
        except Exception:
            pass
    
    return best_d, best_ratio


def _full_refine(P0, deadline):
    """Full 100-parameter refinement using L-BFGS-B on log-softmax parameterization."""
    n = 10
    start_time = time.time()
    
    # Use log-softmax parameterization: P = softmax(theta)
    P_init = P0.copy()
    P_init = np.clip(P_init, 1e-15, None)
    P_init = P_init / P_init.sum()
    theta = np.log(P_init) - np.log(P_init).sum()  # not quite right, use log directly
    
    # Actually, use a simpler approach: optimize in log-space
    # theta[i,j] -> P = exp(theta) / sum(exp(theta))
    theta = np.log(P_init + 1e-15)
    
    def _neg_ratio_theta(theta_flat):
        theta_mat = theta_flat.reshape(n, n)
        # Numerical stability
        theta_mat -= theta_mat.max()
        P = np.exp(theta_mat)
        P = P / P.sum()
        return -_compute_ratio(P)
    
    theta_flat = theta.flatten()
    
    try:
        result = minimize(
            _neg_ratio_theta,
            theta_flat,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        theta_opt = result.x.reshape(n, n)
        theta_opt -= theta_opt.max()
        P_opt = np.exp(theta_opt)
        P_opt = P_opt / P_opt.sum()
        ratio_opt = _compute_ratio(P_opt)
        
        return P_opt, ratio_opt
    except Exception:
        return P0, _compute_ratio(P0)


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses exponential family ansatz with differential evolution, anti-diagonal weights,
    and full log-softmax refinement.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = 580.0  # seconds, with safety margin
    
    best_P = np.ones((n, n)) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    # Phase 1: Exponential family with DE + L-BFGS-B (4 params)
    P_exp, ratio_exp = _optimize_exp_family(deadline)
    if ratio_exp > best_ratio:
        best_ratio = ratio_exp
        best_P = P_exp.copy()
    
    # Phase 2: Anti-diagonal weight optimization (19 params)
    d_best, ratio_diag = _optimize_anti_diag(np.ones(19) / 19.0, deadline)
    P_diag = _d_to_P(d_best)
    if ratio_diag > best_ratio:
        best_ratio = ratio_diag
        best_P = P_diag.copy()
    
    # Phase 3: Full log-softmax refinement from best structured solution
    P_refined, ratio_refined = _full_refine(best_P, deadline)
    if ratio_refined > best_ratio:
        best_P = P_refined
        best_ratio = ratio_refined
    
    # Phase 4: More anti-diagonal restarts
    np.random.seed(42)
    for restart in range(8):
        if time.time() - start_time > deadline - 60:
            break
        
        d_rand = np.random.dirichlet(np.ones(19) * 2)
        d_opt, ratio_opt = _optimize_anti_diag(d_rand, deadline)
        P_opt = _d_to_P(d_opt)
        
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt
        
        # Quick refinement
        P_ref, ratio_ref = _full_refine(best_P, deadline)
        if ratio_ref > best_ratio:
            best_ratio = ratio_ref
            best_P = P_ref
    
    # Phase 5: More exponential family restarts
    np.random.seed(123)
    for restart in range(10):
        if time.time() - start_time > deadline - 30:
            break
        
        init = np.random.uniform(-1.5, 1.5, size=4)
        
        try:
            result = minimize(
                _neg_ratio_exp,
                init,
                method='L-BFGS-B',
                bounds=[(-3.0, 3.0), (-3.0, 3.0), (-2.0, 2.0), (-1.0, 1.0)],
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            P_opt = _exp_family_P(result.x)
            ratio_opt = _compute_ratio(P_opt)
            
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_P = P_opt.copy()
        except Exception:
            pass
    
    # Final validation
    best_P = _clip_and_normalize(best_P)
    return best_P


# EVOLVE-BLOCK-END
