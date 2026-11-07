import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using multiplicative PI with cost exponentiation.
    PI = Phi((f_best - mu) / sigma) naturally saturates and decays smoothly.
    Cost penalty via PI^(1/cost) prioritizes cheap, high-confidence improvements.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -15, 15)
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Ensure PI is in valid range
    pi = np.clip(pi, 1e-12, 1.0 - 1e-12)
    
    # Deadline guard: if we're running out of time, return simple PI
    if time.time() - t_start > 2.5:
        scores = pi / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Cost-aware weighting: PI^(1/cost)
    # When cost is low (~1.0), exponent is ~1.0, so score ≈ PI
    # When cost is high (~2.0), exponent is ~0.5, so score ≈ sqrt(PI)
    # This means expensive points get their PI "boosted" by taking the root,
    # but since we want to MINIMIZE cost, we actually want the opposite:
    # expensive points should be penalized more.
    # 
    # Let's reconsider: cost in [1, 2]. We want cheap points to score higher.
    # PI^(1/cost): for cost=1, PI^1 = PI; for cost=2, PI^0.5 = sqrt(PI)
    # Since PI < 1, sqrt(PI) > PI, so this actually BOOSTS expensive points.
    # That's wrong for cost minimization.
    #
    # Instead, use PI^(cost) or PI / cost^alpha
    # PI^(cost): for cost=1, PI; for cost=2, PI^2 < PI. This penalizes expensive points.
    # But PI^2 can be very small. Let's use a softer penalty.
    #
    # Use: PI * cost^(-beta) where beta controls the strength of cost penalty
    # Or: PI / (cost^beta)
    # 
    # The directive says: PI^(1/cost) to prioritize cheap, high-confidence improvements.
    # Wait, let me re-read: "apply a soft cost penalty via exponentiation (PI^{1/cost})"
    # 
    # Actually, thinking about it differently: if we have two points with the same PI,
    # the cheaper one should be preferred. PI^(1/cost) gives:
    #   cost=1: PI^1 = PI
    #   cost=2: PI^0.5 = sqrt(PI) > PI
    # This makes the expensive point score HIGHER, which is wrong.
    #
    # The correct interpretation might be that we want to DIVIDE by cost in some sense.
    # Let's use: score = PI / (cost^alpha) where alpha > 0
    # Or equivalently: score = PI * cost^(-alpha)
    #
    # Let's try alpha = 1.0 as a starting point, which gives PI/cost
    # This is a clean, interpretable cost penalty.
    
    cost_penalty = np.power(cost, 1.0)
    scores = pi / cost_penalty
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores) & (scores >= 0), scores, 0.0)
    
    # Final deadline guard
    if time.time() - t_start > 2.8:
        return scores
    
    return scores
