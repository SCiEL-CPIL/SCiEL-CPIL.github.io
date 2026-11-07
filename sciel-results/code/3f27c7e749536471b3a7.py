# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 300.0
_EPS = 1e-9


def _qoei(mu, sigma, f_best, q):
    """Quantile Expected Improvement. q=1 -> EI; q<1 -> more aggressive."""
    s = np.maximum(sigma, _EPS)
    z = (f_best - mu) / s
    # P(Y < q*mu + (1-q)*f_best) where Y ~ N(mu, sigma^2)
    t = (q * mu + (1.0 - q) * f_best - mu) / s
    Phi_t = norm.cdf(t)
    Phi_z = norm.cdf(z)
    # E[max(f_best - Y, 0)] = (f_best - mu) Phi_z + sigma phi(z)
    ei = (f_best - mu) * Phi_z + s * norm.pdf(z)
    # E[max(Y - tau, 0)] = (mu - tau) Phi_t + sigma phi(t)
    pos = (mu - (q * mu + (1.0 - q) * f_best)) * Phi_t + s * norm.pdf(t)
    return (1.0 - q) * ei + q * pos


def _ucb(mu, sigma, beta):
    return mu - beta * sigma  # minimization: lower is better


def _fit_weights(mu, sigma, f_best, cost, deadline):
    """Budgeted greedy simulation optimizing expected simple regret.

    Returns a length-4 weight vector over [QOEI/cost, EI/cost, UCB/cost, cost_eff].
    Fast: few iterations, capped by wall-clock.
    """
    t0 = time.time()
    n = mu.shape[0]
    # Precompute raw components (finite-guarded)
    s = np.maximum(sigma, _EPS)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    qoei = _qoei(mu, sigma, f_best, q=0.3)
    ucb = _ucb(mu, s, beta=1.0)
    c = np.maximum(cost, _EPS)

    comps = np.stack([
        qoei / c,
        ei / c,
        (ucb - mu.min()) / c,   # shift so larger = better (more improvement)
        -c,                      # cost efficiency: prefer cheaper (negated)
    ], axis=1)  # shape (n, 4)

    # Normalize each component to [0,1] scale for stable weighting
    comp_min = comps.min(axis=0)
    comp_max = comps.max(axis=0)
    rng = np.maximum(comp_max - comp_min, _EPS)
    comps = (comps - comp_min) / rng  # (n,4) in [0,1]

    # Greedy: pick weights by trying a handful of candidate weight vectors
    # and scoring each by a simulated regret (cheap proxy).
    candidate_weights = [
        np.array([0.5, 0.2, 0.2, 0.1]),
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.array([0.6, 0.2, 0.1, 0.1]),
        np.array([0.3, 0.3, 0.3, 0.1]),
        np.array([0.5, 0.15, 0.25, 0.1]),
    ]

    # Simple regret proxy: for each candidate, simulate selecting the top
    # candidate by score, and estimate regret as (f_best - predicted_min).
    # We use the GP mu at the selected point as the predicted objective.
    # To respect cost budget, we weight selection by cost.
    def _score_weights(w):
        w = np.asarray(w)
        w = np.maximum(w, 0.0)
        w = w / (w.sum() + _EPS)
        scores = comps @ w  # (n,)
        # Select a small set of "evaluations" mimicking the budget loop.
        # Use top-K by score, but penalize high cost via the comp already included.
        # Regret proxy: mean of (f_best - mu) over selected, clipped at 0.
        idx = np.argsort(-scores)[:max(1, n // 20)]
        regret = np.mean(np.maximum(f_best - mu[idx], 0.0))
        # Lower regret is better -> maximize negative regret
        return -regret

    best_w = candidate_weights[0]
    best_s = -np.inf
    for w in candidate_weights:
        if time.time() > deadline:
            break
        sc = _score_weights(w)
        if sc > best_s:
            best_s = sc
            best_w = w

    # One greedy refinement step: nudge each component weight.
    for _ in range(3):
        if time.time() > deadline:
            break
        improved = False
        for i in range(4):
            for delta in (0.05, -0.05):
                cand = best_w.copy()
                cand[i] += delta
                if (cand < 0).any():
                    continue
                cand = cand / (cand.sum() + _EPS)
                sc = _score_weights(cand)
                if sc > best_s + 1e-6:
                    best_s = sc
                    best_w = cand
                    improved = True
        if not improved:
            break

    return best_w


def acquisition(mu, sigma, f_best, cost):
    t_start = time.time()
    deadline = t_start + _DEADLINE

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]
    s = np.maximum(sigma, _EPS)
    c = np.maximum(cost, _EPS)

    # Fit weights (with its own deadline guard).
    w = _fit_weights(mu, sigma, f_best, cost, deadline - 5.0)

    # Recompute components consistently with the fit.
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    qoei = _qoei(mu, sigma, f_best, q=0.3)
    ucb = _ucb(mu, s, beta=1.0)

    comp_min = np.array([qoei / c, ei / c, ucb - mu.min(), -c])
    comp_min = comp_min.min(axis=0)
    comp_max = comp_min.max(axis=0)
    rng = np.maximum(comp_max - comp_min, _EPS)
    comps = (comp_min - comp_min) / rng  # careful: recompute properly below

    # Proper per-component normalization
    raw = np.stack([
        qoei / c,
        ei / c,
        ucb - mu.min(),
        -c,
    ], axis=1)
    cmin = raw.min(axis=0)
    cmax = raw.max(axis=0)
    rr = np.maximum(cmax - cmin, _EPS)
    comps = (raw - cmin) / rr

    scores = comps @ w

    # Safety: ensure finite.
    scores = np.where(np.isfinite(scores), scores, 0.0)

    if time.time() > deadline - 2.0:
        return scores

    return scores
# EVOLVE-BLOCK-END
