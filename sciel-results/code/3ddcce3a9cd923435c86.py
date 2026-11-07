import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a Predictive Entropy Search (PES) variant that scores each candidate
    by the expected reduction in the global minimum's posterior entropy,
    approximated by the mutual information between the function value at the
    candidate and the location of the current best.
    
    This directly targets the multimodal Weierstrass landscape's global structure
    rather than local exploitation, which is crucial for 1D multimodal optimization.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.5 * sigma_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # === Predictive Entropy Search (PES) variant ===
    # PES scores each candidate by the expected reduction in the entropy
    # of the global minimum. We approximate this via mutual information
    # between f(x) and the location of the current best.
    
    # Key insight: A candidate is valuable if evaluating it is likely to
    # reveal a better global minimum. This happens when:
    # 1. mu(x) is low (promising region)
    # 2. sigma(x) is high (uncertainty - could be much lower)
    # 3. The probability that f(x) < f_best is significant
    
    # Compute z-score for each candidate
    z = (f_best - mu) / sigma_safe
    
    # Probability that f(x) < f_best
    prob_improve = norm.cdf(z)
    
    # Expected improvement (standard EI)
    ei = (f_best - mu) * prob_improve + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # PES-like score: We want to maximize the expected reduction in
    # the posterior entropy of the global minimum. A good proxy is:
    # - High probability of improvement (exploit promising regions)
    # - High uncertainty (explore uncertain regions)
    # - But penalize regions that are very likely to be bad (mu >> f_best)
    
    # The mutual information between f(x) and the global min location
    # is maximized when there's significant uncertainty about whether
    # x contains the global minimum. This is captured by the variance
    # of the indicator "f(x) < f_best" weighted by the improvement.
    
    # Variance of the improvement indicator
    var_indicator = prob_improve * (1 - prob_improve)
    
    # PES score combines:
    # 1. Expected improvement (exploitation)
    # 2. Variance of improvement indicator (uncertainty about global min)
    # 3. A bonus for high sigma in regions with moderate mu
    
    # Normalize EI to [0, 1]
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize variance indicator to [0, 1]
    vi_min, vi_max = np.min(var_indicator), np.max(var_indicator)
    vi_range = vi_max - vi_min
    if vi_range > 1e-12:
        vi_norm = (var_indicator - vi_min) / vi_range
    else:
        vi_norm = np.zeros_like(var_indicator)
    
    # Compute an exploration bonus: high sigma where mu is not too far above f_best
    # This helps escape local minima in the multimodal landscape
    # The bonus is highest when sigma is large and mu is within ~2*sigma of f_best
    exploration_bonus = sigma_safe * np.exp(-((mu - f_best) / (2 * sigma_safe))**2)
    eb_min, eb_max = np.min(exploration_bonus), np.max(exploration_bonus)
    eb_range = eb_max - eb_min
    if eb_range > 1e-12:
        eb_norm = (exploration_bonus - eb_min) / eb_range
    else:
        eb_norm = np.zeros_like(exploration_bonus)
    
    # Adaptive weights based on convergence
    # frac_better: fraction of points where mu < f_best
    # As this increases, we're finding better points, so exploit more
    frac_better = np.mean(mu < f_best)
    
    # Start with more exploration (PES-like), shift to exploitation as we converge
    weight_ei = 0.3 + 0.4 * frac_better
    weight_vi = 0.3 + 0.2 * (1 - frac_better)
    weight_eb = 0.2 + 0.2 * (1 - frac_better)
    
    # Normalize weights
    total_weight = weight_ei + weight_vi + weight_eb
    weight_ei /= total_weight
    weight_vi /= total_weight
    weight_eb /= total_weight
    
    # Combine scores
    score = weight_ei * ei_norm + weight_vi * vi_norm + weight_eb * eb_norm
    
    # Add a small diversity term based on sigma ranking to encourage
    # exploring distinct regions of the multimodal landscape
    sigma_rank = np.argsort(np.argsort(sigma_safe)) / n
    score += 0.1 * sigma_rank
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
