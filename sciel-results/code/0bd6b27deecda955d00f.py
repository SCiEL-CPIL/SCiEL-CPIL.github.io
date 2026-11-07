# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None
_deadline = None

def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time

def _ei_min(mu, sigma, f_best):
    """Expected Improvement for minimization, clipped >= 0, vectorized."""
    eps = 1e-12
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.maximum(ei, 0.0), s

def _ucb_score(mu, sigma, beta):
    """LCB-based score (higher = more promising for minimization): -mu + beta*sigma."""
    return -mu + beta * np.maximum(sigma, 1e-12)

def _norm_arr(a):
    """Min-max normalization to [0, 1]."""
    lo, hi = np.min(a), np.max(a)
    rng = hi - lo
    if rng < 1e-12:
        return np.ones_like(a) * 0.5
    return (a - lo) / rng

def _ts_score(mu, sigma, cost, seed):
    """
    Thompson Sampling score: sample from posterior, then score by cost-adjusted value.
    Lower sampled value / cost is better, so we invert for 'higher is better'.
    """
    rng = np.random.default_rng(seed)
    # Draw one sample from the posterior for each candidate
    # y ~ N(mu, sigma^2)
    noise = rng.standard_normal(len(mu))
    y = mu + np.maximum(sigma, 1e-12) * noise
    
    # We want to minimize y / cost (or similar cost-aware metric)
    # Let's use y / cost as the metric to minimize.
    # Score = - (y / cost) or 1 / (y / cost) if positive?
    # Since y can be negative (hartmann), 1/(y/cost) is problematic.
    # Better: Score = -y / cost. This rewards low y and low cost.
    # Or Score = -y * (1/cost).
    
    # Let's try: Score = -y / cost
    # If y is very negative (good), score is positive and large.
    # If cost is small, score is larger.
    
    score = -y / np.maximum(cost, 1e-12)
    return score

def _fit_params_ts(mu, sigma, f_best, cost, budget):
    """
    Fit a mix of Thompson Sampling score and Deterministic EI/UCB.
    We search over:
    - w_ts: weight for TS score
    - w_ei: weight for EI
    - w_ucb: weight for UCB
    - beta: UCB exploration parameter
    - ts_seed: seed for TS noise (to simulate multiple TS samples)
    
    We evaluate a greedy budgeted sequence for each configuration.
    """
    n = len(mu)
    if n < 2:
        return 0.5, 0.3, 0.2, 1.0, 42

    if budget is None:
        budget = 20.0

    # Precompute EI and Cost Efficiency
    ei, s = _ei_min(mu, sigma, f_best)
    cost_eff = 1.0 / (cost + 1e-12)
    ei_n = _norm_arr(ei)
    ce_n = _norm_arr(cost_eff)

    # Grid for weights (w_ts, w_ei, w_ucb) summing to 1
    # And beta
    # And a few TS seeds to average out randomness or find a lucky one?
    # Averaging is more robust. Let's average over a few seeds for the score calculation in the loop?
    # No, TS is stochastic. We can just pick a fixed seed for the final call, but for fitting,
    # we should probably average the regret over a few TS samples to get a stable metric.
    
    w_grid = [
        (0.6, 0.2, 0.2),
        (0.5, 0.3, 0.2),
        (0.4, 0.4, 0.2),
        (0.3, 0.4, 0.3),
        (0.7, 0.1, 0.2),
        (0.5, 0.2, 0.3),
    ]
    beta_grid = [0.8, 1.2, 1.6, 2.0]
    ts_seeds = [1, 2, 3, 4, 5]  # Use 5 seeds to average TS performance

    best_obj = np.inf
    best_p = (0.5, 0.3, 0.2, 1.2, 1)

    for (wts, wei, wucb) in w_grid:
        for b in beta_grid:
            ucb = _ucb_score(mu, sigma, b)
            ucb_n = _norm_arr(ucb)
            
            # Average TS score over seeds for a stable fit
            ts_scores = []
            for seed in ts_seeds:
                ts_s = _ts_score(mu, sigma, cost, seed)
                ts_s = _norm_arr(ts_s)
                ts_scores.append(ts_s)
            ts_avg = np.mean(ts_scores, axis=0)
            
            score = wts * ts_avg + wei * ei_n + wucb * ucb_n
            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

            # Greedy selection
            remaining = budget
            visited = np.zeros(n, dtype=bool)
            best_val = f_best
            
            for _ in range(min(10, n)):
                feasible = (~visited) & (cost <= remaining)
                if not np.any(feasible):
                    break
                cand_score = np.where(feasible, score, -np.inf)
                pick = int(np.argmax(cand_score))
                if not feasible[pick]:
                    break
                visited[pick] = True
                remaining -= cost[pick]
                if mu[pick] < best_val:
                    best_val = mu[pick]
            
            obj = -(f_best - best_val)
            if obj < best_obj:
                best_obj = obj
                best_p = (wts, wei, wucb, b, 1) # Seed 1 is just a placeholder, we'll use a fixed seed or random?
                # Actually, for the final call, we should use a specific seed.
                # Using a fixed seed makes it deterministic.
                # Let's stick with seed 1 for consistency in the return.

    return best_p[0], best_p[1], best_p[2], best_p[3], 1

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Thompson Sampling and EI/UCB.
    """
    global _start_time, _deadline
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 322.0
    
    if time.time() > _deadline:
        # Emergency: Cheap EI
        ei, _ = _ei_min(mu, sigma, f_best)
        ce = 1.0 / (cost + 1e-12)
        out = 0.8 * ei + 0.2 * ce
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = len(mu)
    if n == 0:
        return np.array([])

    remaining_budget = 22.5 # Nominal

    try:
        w_ts, w_ei, w_ucb, beta, ts_seed = _fit_params_ts(mu, sigma, f_best, cost, remaining_budget)
    except Exception:
        w_ts, w_ei, w_ucb, beta, ts_seed = 0.5, 0.3, 0.2, 1.2, 1

    # Build final score
    ei, s = _ei_min(mu, sigma, f_best)
    ucb = _ucb_score(mu, sigma, beta)
    
    ei_n = _norm_arr(ei)
    ucb_n = _norm_arr(ucb)
    
    # Use the fitted seed for the final TS draw
    ts_score = _ts_score(mu, sigma, cost, ts_seed)
    ts_n = _norm_arr(ts_score)
    
    score = w_ts * ts_n + w_ei * ei_n + w_ucb * ucb_n
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale to [0, 1]
    mx = np.max(score)
    if mx > 1e-12:
        score = score / mx
    
    return score
# EVOLVE-BLOCK-END
