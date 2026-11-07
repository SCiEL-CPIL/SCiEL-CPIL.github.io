"""Cost-aware acquisition for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_GLOBAL = {}


def _elapsed():
    return time.perf_counter() - _GLOBAL.get("t0", time.perf_counter())


def acquisition(mu, sigma, f_best, cost):
    if "t0" not in _GLOBAL:
        _GLOBAL["t0"] = time.perf_counter()
        _GLOBAL["global_best"] = None
    # 330 s wall-clock budget; keep a 5 s safety margin
    t_end = _GLOBAL["t0"] + 325.0

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = mu.shape[0]

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0.0), cost, 1.0)

    # Compute a robust, adaptive exploration coefficient beta.
    # We use the median ratio of sigma to |mu| (clipped) to get a scale-free measure.
    # This adapts to the GP's uncertainty relative to the function scale.
    valid = sigma > 1e-9
    if np.any(valid):
        ratio = sigma[valid] / np.maximum(np.abs(mu[valid]), 1e-9)
        ratio = np.clip(ratio, 0.0, 1e4)
        beta = float(np.median(ratio))
    else:
        beta = 1.0
    beta = float(np.clip(beta, 0.5, 3.0))

    # GP-UCB acquisition: optimistic mean + beta * std
    # We want to minimize f, so we maximize (mu + beta * sigma) / cost
    # Note: since we are minimizing, "optimistic" means high predicted value is bad,
    # but UCB for minimization typically uses (mu - beta*sigma) or (mu + beta*sigma) depending on sign convention.
    # Standard UCB for minimization: we want to find low values.
    # The GP predicts mu. Uncertainty is sigma.
    # A point is promising if it might be low.
    # Lower bound of the posterior: mu - beta * sigma.
    # However, standard "UCB" is often defined for maximization as mu + beta * sigma.
    # For minimization, we can use LCB (Lower Confidence Bound): mu - beta * sigma.
    # Or we can maximize -f, so we maximize -(mu - beta*sigma) = -mu + beta*sigma?
    # Let's stick to the standard form: we want to pick x that minimizes f.
    # The acquisition function a(x) should be high for good x.
    # If we use LCB: a(x) = mu(x) - beta * sigma(x). We maximize this.
    # Wait, if mu is the predicted mean, and we want to find the minimum,
    # we prefer points where the mean is low AND we have high uncertainty (which could hide a lower minimum).
    # So we prefer low mu and high sigma.
    # a(x) = mu(x) - beta * sigma(x) is the standard LCB.
    # Let's verify: if sigma is high, -beta*sigma is large negative, so a(x) is low. That's wrong.
    # We want to MAXIMIZE the acquisition.
    # If we use LCB for minimization: we are looking for the lower bound of the function.
    # The lower bound is mu - beta*sigma. We want to maximize this lower bound?
    # No, we want to find the global minimum.
    # Actually, for minimization, the "optimistic" value is the LOWER bound.
    # We want to select the point with the HIGHEST lower bound? No, that would be conservative.
    # We want to select the point that has the potential to be the lowest.
    # So we want to minimize the lower bound?
    # Let's think about Expected Improvement (EI).
    # EI for minimization: E[max(0, f_best - f(x))].
    # This is high when f(x) is likely to be much lower than f_best.
    # UCB/UCB-style:
    # Common practice: Use (mu - beta * sigma) and MAXIMIZE it?
    # If I have two points:
    # A: mu=0, sigma=0 -> bound=0
    # B: mu=10, sigma=10 -> bound=10 - 10 = 0 (if beta=1)
    # C: mu=-10, sigma=10 -> bound=-10 - 10 = -20
    # We want to pick C because it might be even lower than -10.
    # So we want to MINIMIZE (mu - beta * sigma).
    # Or MAXIMIZE (-mu + beta * sigma).
    # Let's define acquisition score s = -mu + beta * sigma.
    # Higher s is better.
    
    # Let's check the previous code:
    # raw = m + beta * s
    # scores = raw / (c + 1e-9)
    # It used m + beta * s. This is for MAXIMIZATION problems (high mu is good, high sigma is good).
    # But we are MINIMIZING Styblinski-Tang.
    # If the previous code scored 0.7, maybe the sign was wrong or the cost weighting was key.
    # Let's switch to the correct minimization form:
    # We want to find low f.
    # Optimistic prediction for minimum: mu - beta * sigma.
    # We want to pick the point with the LOWEST optimistic prediction.
    # So acquisition = -(mu - beta * sigma) = -mu + beta * sigma.
    # Or simply: acquisition = -mu + beta * sigma.
    
    # However, we must consider cost.
    # Cost = 1 + mean(x scaled to [0,1]).
    # The cost is passed as an array `cost`.
    # We want to balance improvement vs cost.
    # Score = ( -mu + beta * sigma ) / cost ?
    # Or Score = ( -mu + beta * sigma ) - lambda * cost ?
    # The previous code divided by cost. This penalizes high cost.
    
    # Let's use: Score = ( -mu + beta * sigma ) / (cost + eps)
    # This rewards high potential improvement (low mu, high sigma) and penalizes high cost.
    
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        # Minimization: low mu is good, high sigma is good (uncertainty).
        # Potential improvement: -mu + beta * sigma
        potential = -mu + beta * sigma
        scores = potential / (cost + 1e-9)
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    # Add a deterministic grid component to ensure coverage?
    # The previous code did this. It might help avoid local optima in the acquisition landscape.
    # But with 2000 random candidates, coverage is usually good.
    # Let's try a simpler approach first: just the GP-UCB/PI style.
    
    # If we want to be safer, we can mix with a random component or ensure some exploration.
    # But let's stick to the structural change: correct sign for minimization.
    
    # Deadline guard
    if _elapsed() > t_end:
        out = np.full(n, -1e9)
        # Pick the best one found so far or just return valid scores
        best_idx = np.argmax(scores)
        if np.isfinite(scores[best_idx]):
            out[best_idx] = scores[best_idx]
        return out

    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
