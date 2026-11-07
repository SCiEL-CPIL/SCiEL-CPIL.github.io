# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses a hybrid of Probability of Improvement (PI) and Expected Improvement (EI)
    to balance exploration and exploitation.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute z-score
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)

    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Knowledge Gradient (KG) approximation for minimization
    # KG approximates the expected reduction in the global minimum.
    # A common approximation is E[max(0, f_best - f(x))] which is EI, 
    # but KG can be viewed as prioritizing points that are likely to be the new global best.
    # A more direct KG-like score for minimization is the expected improvement 
    # weighted by the probability that the point is the new best.
    # Since we only have the current best, a simple KG proxy is EI itself, 
    # but we can enhance it by considering the "potential" of the point.
    
    # Let's implement a structural change: Adaptive UCB blended with EI.
    # The inspiration with score 9.7863 used UCB with adaptive beta.
    # The inspiration with score 13.3320 used PI scaled by f_best.
    
    # Let's combine:
    # 1. Compute adaptive beta for UCB based on f_best (early exploration).
    max_f = 25.0
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    beta = 0.5 + 1.0 * np.sqrt(ratio)
    
    # UCB score for minimization (higher is better): -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # 2. Compute PI scaled by f_best (from inspiration 2)
    scale = np.maximum(f_best, 1e-6)
    pi_scaled = pi * scale
    
    # 3. Combine EI, UCB, and PI.
    # EI is the strongest signal for local improvement.
    # UCB helps with global exploration.
    # PI helps with probabilistic improvement.
    
    # We want a score that is high for points that are likely to improve f_best.
    # Let's use a weighted sum.
    # Weights: EI is primary, UCB is secondary, PI is tertiary.
    # However, the scales are different. EI can be large, UCB is in [-25, 25] roughly, PI is in [0, 1].
    
    # Let's normalize UCB and PI to be comparable to EI? 
    # Or just use a simple blend.
    
    # Let's try: score = ei + 0.1 * ucb + 0.01 * pi_scaled
    # This keeps EI as the main driver but adds exploration from UCB and PI.
    
    score = ei + 0.1 * ucb + 0.01 * pi_scaled
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple exploitation score (negative mean)
        return -mu
    
    return score
# EVOLVE-BLOCK-END
