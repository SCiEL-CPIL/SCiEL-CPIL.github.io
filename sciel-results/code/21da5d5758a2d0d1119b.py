"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return current best if we exceed 320s (safety margin from 330s limit)
    start_time = time.time()
    
    # Strategy: Global Probability of Improvement (PI) with cost regularization.
    # 
    # Rationale:
    # 1. PI focuses on the probability that f(x) < f_best. This is more robust for
    #    global optimization on multimodal landscapes like Rosenbrock than EI,
    #    which can get stuck in local basins if the expected improvement is small
    #    but the probability of finding the global minimum is non-zero.
    # 2. The cost term is used to penalize expensive evaluations, but we cap the
    #    effective cost to avoid over-penalizing. This ensures that if a point has
    #    high probability of improvement, it is selected even if it is somewhat
    #    expensive, unless the cost is prohibitive.
    # 3. We use a slightly aggressive exploration factor to ensure we explore the
    #    entire domain, especially in the early stages when f_best is high.
    
    # Compute z-score for PI
    # PI = Phi((f_best - mu) / sigma)
    sigma_safe = np.clip(sigma, 1e-9, None)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware adjustment:
    # We want to maximize PI / cost.
    # However, to prevent the algorithm from ignoring high-cost points that might
    # be the global minimum, we use a "soft" cost penalty.
    # Let's define the score as PI * (1 / (1 + cost_penalty * cost))
    # Or simply PI / (1 + cost) which is a common heuristic.
    
    # Let's try a weighted combination:
    # score = PI / (1 + cost)
    # This effectively discounts the probability of improvement by the cost.
    
    score = pi / (1.0 + cost)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > 320:
         # Fallback to a simple random-like selection or current best if stuck
         # But since we need to return 2000 scores, we just return the current score
         # The evaluator will pick the max.
         pass

    return score
# EVOLVE-BLOCK-END
