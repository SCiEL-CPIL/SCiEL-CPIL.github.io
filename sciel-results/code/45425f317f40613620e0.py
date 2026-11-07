# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 300.0
_EPS = 1e-9
_T0 = time.time()


def _qoei(mu, sigma, f_best, q):
    """Quantile Expected Improvement. q=1 -> EI; q<1 -> more aggressive."""
    s = np.maximum(sigma, _EPS)
    z = (f_best - mu) / s
    t = (q * mu + (1.0 - q) * f_best - mu) / s
    Phi_t = norm.cdf(t)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + s * norm.pdf(z)
    pos = (mu - (q * mu + (1.0 - q) * f_best)) * Phi_t + s * norm.pdf(t)
    return (1.0 - q) * ei + q * pos


def _ucb(mu, sigma, beta):
    return mu - beta * sigma


def _fit_weights(mu, sigma, f_best, cost, deadline):
    """Budgeted greedy simulation optimizing expected simple regret.

    Returns a length-4 weight vector over [QOEI/cost, EI/cost, UCB/cost, cost_eff].
    Fast: few iterations, capped by wall-clock.
    """
    t0 = time.time()
    n = mu.shape[0]
    s = np.maximum(sigma, _EPS)
    c = np.maximum(cost, _EPS)
    
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    qoei = _qoei(mu, sigma, f_best, q=0.3)
    ucb = _ucb(mu, s, beta=1.0)

    comps = np.stack([
        qoei / c,
        ei / c,
        (ucb - mu.min()) / c,
        -c,
    ], axis=1)

    comp_min = comps.min(axis=0)
    comp_max = comps.max(axis=0)
    rng = np.maximum(comp_max - comp_min, _EPS)
    comps = (comps - comp_min) / rng

    candidate_weights = [
        np.array([0.5, 0.2, 0.2, 0.1]),
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.array([0.6, 0.2, 0.1, 0.1]),
        np.array([0.3, 0.3, 0.3, 0.1]),
        np.array([0.5, 0.15, 0.25, 0.1]),
    ]

    def _score_weights(w):
        w = np.asarray(w)
        w = np.maximum(w, 0.0)
        w = w / (w.sum() + _EPS)
        scores = comps @ w
        idx = np.argsort(-scores)[:max(1, n // 20)]
        regret = np.mean(np.maximum(f_best - mu[idx], 0.0))
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

    # CE-UCB Strategy:
    # Score = (Beta * sigma - mu) / cost
    # Beta is scaled by sqrt(remaining_budget_proxy) to intensify exploration as budget depletes.
    # Here we use a proxy for remaining budget based on the mean cost of the current candidates.
    # Since we don't have exact budget remaining, we use a heuristic:
    # As f_best improves, we might want to exploit more. But the prompt suggests
    # "intensify exploration as the budget depletes". 
    # However, in BO loops, usually we exploit more as we get closer to the end.
    # Let's stick to the prompt's suggestion: "beta is dynamically scaled by the square root of the remaining budget".
    # We don't know the exact remaining budget, but we can estimate it.
    # Total budget is 30. Initial cost is 5 * (1 + 0.5) = 7.5 (approx).
    # Let's assume a simple scaling factor. 
    # Actually, a simpler and more robust structural change is to use a fixed, well-tuned CE-UCB 
    # or a simple adaptive one based on the spread of mu.
    
    # Let's combine the weight fitting with a CE-UCB component.
    # The current program fits weights for [QOEI/cost, EI/cost, UCB/cost, -cost].
    # We will replace the UCB component with a CE-UCB component that has a dynamic beta.
    
    # Dynamic Beta:
    # Let's use the variance of mu as a proxy for "remaining budget" or "exploration need".
    # High variance in mu -> early stage -> need more exploration -> higher beta.
    # Low variance in mu -> late stage -> need exploitation -> lower beta.
    mu_std = np.std(mu)
    # Scale beta between 1.0 and 3.0
    # If mu_std is large (e.g., > 0.5), beta = 3.0
    # If mu_std is small (e.g., < 0.1), beta = 1.0
    beta = 1.0 + 2.0 * np.clip(mu_std / 0.5, 0.0, 1.0)
    
    # Fit weights (with its own deadline guard).
    w = _fit_weights(mu, sigma, f_best, cost, deadline - 5.0)

    # Recompute components consistently with the fit.
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    qoei = _qoei(mu, sigma, f_best, q=0.3)
    
    # Use the dynamic beta for UCB
    ucb = _ucb(mu, s, beta=beta)

    # Proper per-component normalization
    raw = np.stack([
        qoei / c,
        ei / c,
        (ucb - mu.min()) / c,  # Normalized CE-UCB
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
