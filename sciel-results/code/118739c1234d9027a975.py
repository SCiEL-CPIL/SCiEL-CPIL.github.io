"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtri
from scipy.stats import norm

# Track start time for deadline guard
_t0 = None
_DL = None

def _elapsed():
    if _t0 is None:
        return 0.0
    return time.time() - _t0

def _norm_cdf_arr(z):
    # Use scipy for accuracy but vectorized
    return norm.cdf(z)

def _norm_pdf_arr(z):
    return norm.pdf(z)

def _ei_min_vec(mu, sigma, f_best):
    eps = 1e-12
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    cdf = _norm_cdf_arr(z)
    pdf = _norm_pdf_arr(z)
    ei = (f_best - mu) * cdf + s * pdf
    return np.maximum(ei, 0.0)

def _qoei_min(mu, sigma, f_best, q):
    eps = 1e-12
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    z_q = ndtri(q)
    
    # Standard EI
    cdf_z = _norm_cdf_arr(z)
    pdf_z = _norm_pdf_arr(z)
    ei_full = (f_best - mu) * cdf_z + s * pdf_z
    
    # QOEI calculation
    # E[(y - f_best) * 1{y < f_best - q*s}]
    # Let Y ~ N(mu, s^2). We want E[(Y - f_best) * 1{Y < f_best - q*s}]
    # This is the "tail" of the improvement beyond the quantile f_best - q*s.
    # Alternatively, QOEI = EI - E[(Y - f_best) * 1{f_best - q*s <= Y < f_best}]
    # The second term is the improvement in the "middle" region.
    
    # Let's compute E[(Y - f_best) * 1{Y < f_best - q*s}] directly?
    # No, QOEI is usually defined as E[max(Y - f_best, 0)] - E[max(Y - f_best, 0) * 1{Y > f_best - q*s}]?
    # Actually, QOEI(q) = E[(Y - f_best) * 1{Y < f_best - q*s}] is not quite right for minimization.
    # For minimization, improvement is f_best - Y.
    # EI = E[max(f_best - Y, 0)]
    # QOEI(q) = E[(f_best - Y) * 1{Y < f_best - q*s}]
    # This captures the expected improvement from the best q fraction of the distribution.
    
    # Y ~ N(mu, s^2).
    # Threshold T = f_best - q*s.
    # We want E[(f_best - Y) * 1{Y < T}]
    # = f_best * P(Y < T) - E[Y * 1{Y < T}]
    # P(Y < T) = Phi((T - mu) / s) = Phi((f_best - mu)/s - q) = Phi(z - q)
    # E[Y * 1{Y < T}] = mu * P(Y < T) + s * phi((T - mu)/s)
    #                   = mu * Phi(z - q) + s * phi(z - q)
    
    d = z - q
    cdf_d = _norm_cdf_arr(d)
    pdf_d = _norm_pdf_arr(d)
    
    term1 = f_best * cdf_d
    term2 = mu * cdf_d + s * pdf_d
    qoei = term1 - term2
    
    # Numerical safety
    return np.maximum(qoei, 0.0)

def _ucb_vec(mu, sigma, beta):
    return -mu + beta * np.maximum(sigma, 1e-12)

def _minmax(a):
    a = np.asarray(a, dtype=float)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    rng = hi - lo
    if rng < 1e-12:
        return np.ones_like(a) * 0.5
    out = (a - lo) / rng
    return np.nan_to_num(out, nan=0.5, posinf=0.5, neginf=0.5)

def _score_components(mu, sigma, f_best, cost, q, beta):
    """Compute all components. Returns dict of normalized scores."""
    n = len(mu)
    if n < 2:
        return None
    
    ei = _ei_min_vec(mu, sigma, f_best)
    qoei = _qoei_min(mu, sigma, f_best, q)
    ucb = _ucb_vec(mu, sigma, beta)
    
    # Cost-aware versions
    cost_safe = np.maximum(cost, 1e-12)
    ei_cost = ei / cost_safe
    qoei_cost = qoei / cost_safe
    ucb_cost = ucb / cost_safe
    
    return {
        'ei': _minmax(ei),
        'qoei': _minmax(qoei),
        'ucb': _minmax(ucb),
        'ei_cost': _minmax(ei_cost),
        'qoei_cost': _minmax(qoei_cost),
        'ucb_cost': _minmax(ucb_cost),
    }

def _fit_acquisition(mu, sigma, f_best, cost):
    """
    Fit acquisition parameters by simulating budgeted greedy search.
    """
    n = len(mu)
    if n < 2:
        return 0.5, 1.0, (0.2, 0.2, 0.2, 0.2, 0.1, 0.1)
    
    # Budget estimate
    budget = 20.0
    
    # Grid search
    q_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    beta_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Weight grid: w_ei, w_qoei, w_ucb, w_ei_cost, w_qoei_cost, w_ucb_cost
    w_grid = [
        (0.3, 0.3, 0.2, 0.1, 0.1, 0.0),
        (0.2, 0.3, 0.2, 0.1, 0.1, 0.1),
        (0.1, 0.3, 0.2, 0.2, 0.1, 0.1),
        (0.1, 0.2, 0.2, 0.2, 0.2, 0.1),
        (0.0, 0.3, 0.2, 0.2, 0.2, 0.1),
        (0.1, 0.2, 0.2, 0.1, 0.2, 0.2),
        (0.2, 0.2, 0.2, 0.1, 0.2, 0.1),
    ]
    
    best_regret = np.inf
    best_q = 0.5
    best_beta = 1.0
    best_w = (0.2, 0.2, 0.2, 0.2, 0.1, 0.1)
    
    # Precompute score components
    score_cache = {}
    for q in q_grid:
        for beta in beta_grid:
            key = (q, beta)
            comps = _score_components(mu, sigma, f_best, cost, q, beta)
            if comps is not None:
                score_cache[key] = comps
    
    for w in w_grid:
        for q in q_grid:
            for beta in beta_grid:
                key = (q, beta)
                if key not in score_cache:
                    continue
                comps = score_cache[key]
                
                score_vec = (
                    w[0] * comps['ei'] +
                    w[1] * comps['qoei'] +
                    w[2] * comps['ucb'] +
                    w[3] * comps['ei_cost'] +
                    w[4] * comps['qoei_cost'] +
                    w[5] * comps['ucb_cost']
                )
                score_vec = np.nan_to_num(score_vec, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Greedy budget simulation
                remaining = budget
                visited = np.zeros(n, dtype=bool)
                best_val = f_best
                
                for _ in range(min(15, n)):
                    feasible = (~visited) & (cost <= remaining + 1e-9)
                    if not np.any(feasible):
                        break
                    cand_score = np.where(feasible, score_vec, -np.inf)
                    pick = int(np.argmax(cand_score))
                    if not feasible[pick]:
                        break
                    visited[pick] = True
                    remaining -= cost[pick]
                    if mu[pick] < best_val:
                        best_val = mu[pick]
                
                regret = f_best - best_val
                if regret < best_regret:
                    best_regret = regret
                    best_q = q
                    best_beta = beta
                    best_w = w
    
    return best_q, best_beta, best_w

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 6d.
    Uses QOEI, EI, UCB, and cost-aware components, with parameters fitted
    via budgeted greedy simulation.
    """
    global _t0, _DL
    if _t0 is None:
        _t0 = time.time()
        _DL = _t0 + 320.0
    
    if time.time() > _DL:
        # Emergency fallback
        ei = _ei_min_vec(mu, sigma, f_best)
        ce = 1.0 / (cost + 1e-12)
        out = 0.8 * _minmax(ei) + 0.2 * _minmax(ce)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = len(mu)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    
    try:
        q, beta, w = _fit_acquisition(mu, sigma, f_best, cost)
    except Exception:
        q, beta, w = 0.5, 1.0, (0.2, 0.2, 0.2, 0.2, 0.1, 0.1)
    
    # Build final score
    comps = _score_components(mu, sigma, f_best, cost, q, beta)
    if comps is None:
        ei = _ei_min_vec(mu, sigma, f_best)
        return _minmax(ei)
    
    score = (
        w[0] * comps['ei'] +
        w[1] * comps['qoei'] +
        w[2] * comps['ucb'] +
        w[3] * comps['ei_cost'] +
        w[4] * comps['qoei_cost'] +
        w[5] * comps['ucb_cost']
    )
    
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
# EVOLVE-BLOCK-END
