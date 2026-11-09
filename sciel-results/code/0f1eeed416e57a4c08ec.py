"""Cost-aware acquisition function with adaptive probability-of-improvement and cost correction."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def _compute_poi_adaptive(mu, sigma, f_best, delta):
    """
    Compute adaptive Probability of Improvement.
    
    POI = P(f(x) < f_best - delta) = Phi((f_best - delta - mu) / sigma)
    
    This is structurally different from EI because:
    1. It's a probability (bounded [0,1]) rather than an expected value
    2. The delta parameter controls the "ambition" of improvement
    3. It naturally handles the exploration-exploitation tradeoff through delta
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    # We want P(f(x) < f_best - delta)
    # = P((f(x) - mu) / sigma < (f_best - delta - mu) / sigma)
    # = Phi((f_best - delta - mu) / sigma)
    z = (f_best - delta - mu) / sigma_safe
    poi = norm.cdf(z)
    return np.where(np.isfinite(poi), poi, 0.0)


def _compute_expected_improvement_normalized(mu, sigma, f_best):
    """
    Compute normalized Expected Improvement for scale-invariant comparison.
    
    EI_norm = EI / max(sigma_mean, epsilon)
    This makes EI comparable across different uncertainty regimes.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    # Normalize by typical scale
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-9:
        ei_norm = ei / sigma_mean
    else:
        ei_norm = ei
    return np.where(np.isfinite(ei_norm), ei_norm, 0.0)


def _compute_cost_aware_ucb(mu, sigma, f_best, cost, kappa):
    """
    Compute cost-aware UCB with a different cost correction mechanism.
    
    Instead of dividing by cost, we subtract a cost penalty:
    score = (f_best - mu + kappa * sigma) - penalty * (cost - 1)
    
    This is structurally different because it's additive rather than multiplicative.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    ucb = f_best - mu + kappa * sigma_safe
    # Additive cost penalty
    penalty = 0.3  # Tune: how much to penalize higher cost
    score = ucb - penalty * (cost - 1.0)
    return np.where(np.isfinite(score), score, -1e10)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).

    Uses an adaptive blend of:
    1. Probability of Improvement (POI) with adaptive delta
    2. Normalized Expected Improvement (EI_norm)
    3. Cost-aware UCB with additive cost penalty

    This is a structural change from the previous EI/UCB blend:
    - POI provides a probabilistic measure of improvement likelihood
    - EI_norm is scale-invariant for better cross-regime comparison
    - Additive cost penalty is a different cost correction mechanism

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).

    Returns
    -------
    np.ndarray, shape (2000,)
        Finite scores; higher = prefer to evaluate next.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        # Return a safe, finite fallback using simple EI
        sigma_safe = np.maximum(sigma, 1e-9)
        improvement = f_best - mu
        z = improvement / sigma_safe
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei / (cost + 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_mean = np.mean(sigma_safe)

    # Adaptive delta for POI: larger delta when far from optimum, smaller when close
    # delta controls how "ambitious" we are: we want improvement by at least delta
    # When f_best is large (far from optimal), we can afford to be ambitious (large delta)
    # When f_best is small (close to optimal), we should be conservative (small delta)
    # delta = 0.1 + 0.4 * (1 - sigmoid(f_best, 0, 2))
    t_close = 1.0 / (1.0 + np.exp(-3.0 * (f_best - 1.0)))  # Sigmoid centered at f_best=1
    delta = 0.05 + 0.35 * (1.0 - t_close)
    delta = float(np.clip(delta, 0.01, 0.5))

    # Compute the three acquisition components
    poi = _compute_poi_adaptive(mu, sigma, f_best, delta)
    ei_norm = _compute_expected_improvement_normalized(mu, sigma, f_best)
    
    # Adaptive kappa for UCB component
    # When uncertainty is high, explore more; when low, exploit more
    kappa = 1.0 + 0.5 * (sigma_mean / max(sigma_mean, 0.1))
    kappa = float(np.clip(kappa, 0.5, 2.5))
    
    ucb_cost = _compute_cost_aware_ucb(mu, sigma, f_best, cost, kappa)

    # Adaptive blending weights based on optimization state
    # Early on (high f_best, high uncertainty): weight POI and UCB more
    # Late (low f_best, low uncertainty): weight EI_norm more (exploit)
    
    # t_exploit: 0 when f_best is large, 1 when f_best is small
    t_exploit = 1.0 / (1.0 + np.exp(-4.0 * (f_best - 0.5)))
    
    # Weights: sum to 1
    w_poi = 0.3 * (1.0 - t_exploit) + 0.1 * t_exploit  # 0.3 early, 0.1 late
    w_ei = 0.3 * (1.0 - t_exploit) + 0.6 * t_exploit   # 0.3 early, 0.6 late
    w_ucb = 1.0 - w_poi - w_ei                          # 0.4 early, 0.3 late
    
    # Ensure weights are positive and sum to 1
    w_poi = max(w_poi, 0.05)
    w_ei = max(w_ei, 0.1)
    w_ucb = max(w_ucb, 0.05)
    total_w = w_poi + w_ei + w_ucb
    w_poi /= total_w
    w_ei /= total_w
    w_ucb /= total_w

    # Combine: POI is in [0,1], EI_norm is roughly O(1), UCB_cost is in objective units
    # Need to scale them to comparable ranges
    # Scale POI: multiply by a factor to bring it to O(1) range
    poi_scaled = poi * 2.0  # Scale up since POI is in [0,1]
    
    # Scale EI_norm: it's already roughly O(1)
    ei_scaled = ei_norm
    
    # Scale UCB_cost: normalize by typical value scale
    if sigma_mean > 1e-9:
        ucb_scaled = ucb_cost / sigma_mean
    else:
        ucb_scaled = ucb_cost

    combined = w_poi * poi_scaled + w_ei * ei_scaled + w_ucb * ucb_scaled
    
    # Cost-aware scaling: use a softer cost correction
    # Instead of dividing by cost, use a power-law cost penalty
    cost_penalty = np.power(cost, 0.5)  # sqrt(cost) for softer penalty
    cost_penalty = np.maximum(cost_penalty, 1e-6)
    
    total_score = combined / cost_penalty
    
    # Ensure finiteness
    total_score = np.where(np.isfinite(total_score), total_score, -1e10)
    
    # Return raw scores
    return total_score
# EVOLVE-BLOCK-END
