"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track progress across calls within a seed
_step_count = 0
_start_time = None


def _quantile_acquisition(mu, sigma, f_best):
    """Quantile Acquisition Function: P(X < f_best) = Phi((f_best - mu) / sigma)"""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    return norm.cdf(z)


def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, kappa):
    """Lower Confidence Bound for minimization (we maximize the negative)."""
    return -mu + kappa * sigma


def _entropy_proxy(mu, sigma):
    """
    Entropy maximization proxy: log(sigma^2).
    Directly targets regions of highest posterior uncertainty/information gain.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    return np.log(sigma_safe ** 2)


def acquisition(mu, sigma, f_best):
    global _step_count, _start_time
    
    # Initialize time tracking on first call
    if _start_time is None:
        _start_time = time.time()
        _step_count = 0
    
    elapsed = time.time() - _start_time
    _step_count += 1
    
    # Safety margin: return simple UCB if we're getting close to deadline
    if elapsed > 300:
        scores = -mu + 2.0 * sigma
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if not np.isfinite(f_best):
        f_best = 1e6
    
    # Compute component scores
    qaf_scores = _quantile_acquisition(mu, sigma, f_best)
    ei_scores = _expected_improvement(mu, sigma, f_best, xi=0.01)
    
    # Adaptive UCB: beta scales with distance from optimum
    # The inspiration shows high beta (5.0) is effective for escaping local minima
    # We use a base of 5.0 but dampen it slightly as we progress to allow convergence
    abs_f_best = abs(f_best)
    # Start with high exploration (beta=5.0)
    # As f_best gets closer to 0 (the minimum), we can afford to reduce exploration
    # However, Beale is tricky, so we keep it high for most of the run.
    # Let's stick to a fixed high beta as per the strongest inspiration, 
    # or a very slow decay.
    # The current program used 2.0-4.0. The best inspiration used 5.0.
    # Let's use 5.0 for the UCB component.
    beta = 5.0
    ucb_scores = _ucb(mu, sigma, beta)
    
    # Entropy proxy score
    entropy_scores = _entropy_proxy(mu, sigma)
    
    # Normalize each score to [0, 1] for combination
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    qaf_norm = normalize(qaf_scores)
    ei_norm = normalize(ei_scores)
    ucb_norm = normalize(ucb_scores)
    entropy_norm = normalize(entropy_scores)
    
    # Adaptive weighting based on progress
    # progress = min(1.0, _step_count / 25.0)  # 25 steps to fully exploit
    
    # Strategy:
    # Early: High UCB (5.0) and Entropy to map the landscape.
    # Late: Shift towards EI for refinement.
    
    # We have 25 steps total (30 evals - 5 initial).
    # Let's use a smooth transition.
    progress = min(1.0, _step_count / 20.0)
    
    # Weights:
    # UCB is the strongest performer in isolation (score 6.79).
    # Entropy is the new structural change to force exploration of unvisited regions.
    # EI is for exploitation.
    # QAF is a probability of improvement, similar to EI but different shape.
    
    # Let's prioritize UCB and Entropy early, and EI late.
    
    # Base weights:
    # w_ucb = 0.5 * (1 - 0.5 * progress)  # Starts at 0.5, ends at 0.25
    # w_entropy = 0.3 * (1 - 0.5 * progress) # Starts at 0.3, ends at 0.15
    # w_ei = 0.2 + 0.5 * progress # Starts at 0.2, ends at 0.7
    # w_qaf = 0.0 # Drop QAF to simplify? Or keep it small.
    
    # Let's try a mix that heavily favors the high-beta UCB but adds entropy
    # to break ties in high-uncertainty areas where mu might be misleading.
    
    # Since UCB with beta=5 was the single best performer, we should keep it dominant.
    # The entropy term helps when UCB is flat or when we want to explicitly target
    # "unknown" regions rather than just "potentially good" regions.
    
    # Let's assign:
    # Early (progress=0): 40% UCB, 30% Entropy, 20% EI, 10% QAF
    # Late (progress=1):  20% UCB, 10% Entropy, 60% EI, 10% QAF
    
    w_ucb = 0.4 * (1 - progress) + 0.2 * progress
    w_entropy = 0.3 * (1 - progress) + 0.1 * progress
    w_ei = 0.2 * (1 - progress) + 0.6 * progress
    w_qaf = 0.1 # Constant small weight
    
    # Combine
    scores = w_ucb * ucb_norm + w_entropy * entropy_norm + w_ei * ei_norm + w_qaf * qaf_norm
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores
# EVOLVE-BLOCK-END
