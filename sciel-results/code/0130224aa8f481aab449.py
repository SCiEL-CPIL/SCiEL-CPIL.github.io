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
    lo = np.min(a)
    hi = np.max(a)
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
    
    # We want to minimize y subject to cost. 
    # A common heuristic is to minimize y + lambda * cost, or y/cost.
    # Since cost varies, let's use a cost-aware regret surrogate.
    # Actually, for Thompson Sampling in BO, we typically just sample the function value
    # and pick the argmin. But we need a score for ALL candidates.
    # The score should reflect the probability of being the best, or expected improvement.
    # A simpler TS score: -y / cost. But this can be unstable.
    # Let's stick to EI for stability, but use TS to select the top candidate?
    # No, the task asks for 2000 scores.
    # Let's try a different structural change: 
    # Instead of fitting weights, use a direct cost-aware EI formulation.
    # EI_cost(x) = EI(x) / cost(x) ? Or EI(x) * (1/cost(x))^alpha?
    
    # For now, this function is a placeholder for a potential TS-based approach,
    # but the main acquisition will use a refined cost-aware EI.
    return -y / (cost + 1e-12)

def _fit_params_ts(mu, sigma, f_best, cost, budget):
    """
    Fit parameters for a cost-aware score using a greedy simulation.
    We try different values of alpha (cost exponent) and beta (UCB).
    """
    n = len(mu)
    if n < 2:
        return 1.0, 1.0

    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    betas = [0.5, 1.0, 1.5, 2.0]

    best_obj = np.inf
    best_alpha = 1.0
    best_beta = 1.0

    # Precompute EI
    ei, _ = _ei_min(mu, sigma, f_best)
    cost_eff = 1.0 / (cost + 1e-12)

    for alpha in alphas:
        for beta in betas:
            # Score = EI * cost_eff^alpha + UCB term?
            # Let's try: Score = EI * (cost_eff)^alpha
            # And add a small UCB component for exploration if needed?
            # The task says "cost-aware acquisition".
            # A common choice is EI(x) / cost(x).
            # Let's simulate greedy selection with this score.
            
            score = ei * (cost_eff ** alpha)
            # Add UCB? Maybe not, EI already has exploration.
            # But if sigma is high, EI is high.
            # Let's stick to pure cost-aware EI for the simulation.
            
            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            
            remaining = budget
            visited = np.zeros(n, dtype=bool)
            best_val = f_best
            total_cost = 0.0
            
            # Limit iterations for speed
            for _ in range(min(15, n)):
                feasible = (~visited) & (cost <= remaining)
                if not np.any(feasible):
                    break
                cand_score = np.where(feasible, score, -np.inf)
                pick = int(np.argmax(cand_score))
                if not feasible[pick]:
                    break
                visited[pick] = True
                total_cost += cost[pick]
                remaining -= cost[pick]
                if mu[pick] < best_val:
                    best_val = mu[pick]
            
            obj = -(f_best - best_val) # We want to minimize regret, so maximize improvement
            if obj < best_obj:
                best_obj = obj
                best_alpha = alpha
                best_beta = beta

    return best_alpha, best_beta

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses a fitted cost-exponent for EI.
    """
    global _start_time, _deadline
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 322.0  # 8s safety margin under 330s kill
    if time.time() > _deadline:
        # Emergency: cheap cost-aware EI
        ei, _ = _ei_min(mu, sigma, f_best)
        ce = 1.0 / (cost + 1e-12)
        out = ei * ce
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = len(mu)
    if n == 0:
        return np.array([])

    # Approximate remaining budget
    remaining_budget = 30.0 - 7.5  # nominal

    # Fit parameters
    try:
        alpha, beta = _fit_params_ts(mu, sigma, f_best, cost, remaining_budget)
    except Exception:
        alpha = 1.0
        beta = 1.0

    # Build final score
    ei, _ = _ei_min(mu, sigma, f_best)
    cost_eff = 1.0 / (cost + 1e-12)
    
    # Primary score: Cost-aware EI
    score = ei * (cost_eff ** alpha)
    
    # Add a small UCB component to encourage exploration in uncertain regions,
    # also cost-aware?
    ucb = _ucb_score(mu, sigma, beta)
    ucb_n = _norm_arr(ucb)
    ei_n = _norm_arr(ei)
    
    # Combine? The simulation only optimized the cost-aware EI part.
    # Let's stick to the simulated score to be consistent with the fit.
    # score = ei * (cost_eff ** alpha)
    
    # Ensure finite and positive (for higher is better)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale
    mx = np.max(score)
    if mx > 1e-12:
        score = score / mx
    
    return score

# EVOLVE-BLOCK-END
