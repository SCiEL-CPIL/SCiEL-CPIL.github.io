"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for time management
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we are too close to the limit
    elapsed = time.time() - _start_time
    if elapsed > 295:
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is strictly positive
    cost = np.maximum(cost, 1e-9)
    
    # z-score for Expected Improvement / Probability of Improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # We want to maximize the acquisition score.
    # Cost-awareness: divide by cost.
    # 
    # Strategy: Blend PI and EI.
    # PI is bounded [0, 1], robust to scale.
    # EI is unbounded, more aggressive exploitation.
    #
    # Let's try a weighted sum: Score = (w * PI + (1-w) * EI) / cost
    #
    # What is a good w?
    # If we use pure EI / cost (w=0), it can be unstable if f_best is not well-estimated or if sigma is noisy.
    # If we use pure PI / cost (w=1), it is very conservative.
    #
    # The previous best score (2.0555) used a 50/50 blend of UCB and EI.
    # UCB = mu - beta * sigma (for minimization, we maximize mu - beta*sigma? No, we minimize f.
    # So UCB for minimization is mu - beta * sigma. We want to maximize this?
    # Wait. If we minimize f, and f ~ N(mu, sigma^2), the "upper" bound (worst case) is mu + beta*sigma.
    # The "lower" bound (best case) is mu - beta*sigma.
    # Standard UCB for maximization (where higher is better) is mu + beta*sigma.
    # For minimization, we want to find low values.
    # So we want to maximize the potential for low values.
    # A common heuristic is to maximize (f_best - mu) + beta * sigma? No.
    # Let's stick to the definition:
    # We want to select x that minimizes f(x).
    # EI is defined for minimization as E[max(f_best - f(x), 0)].
    # This is exactly what we calculated: ei = (f_best - mu) * Phi(z) + sigma * phi(z).
    # This value is >= 0. Higher is better.
    #
    # PI is P(f(x) < f_best). Higher is better.
    #
    # Let's try a blend.
    # The inspiration says "50/50 blend of UCB and EI" scored 2.0555.
    # What is UCB in this context?
    # Often UCB is defined as mu - beta * sigma for minimization?
    # If we define Score_UCB = (f_best - mu) + beta * sigma?
    # Let's look at the "UCB" formula commonly used in minimization:
    # We want to maximize the "optimistic" value.
    # Optimistic value = mu - beta * sigma.
    # But we want to compare to f_best?
    # Actually, EI is already an optimistic metric relative to f_best.
    #
    # Let's try the blend of EI and PI directly, as suggested by the directive:
    # "Use a cost-aware Probability of Improvement (PI) acquisition function...
    # Strategy to try in this change: Use a cost-aware Probability of Improvement (PI) acquisition function,
    # defined as Phi((f_best - mu) / sigma) / cost, which provides a bounded, sigmoidal exploration signal..."
    #
    # The directive specifically suggests PI / cost.
    # However, the inspiration says a blend of UCB and EI worked well.
    # Maybe the "UCB" in the inspiration was actually something else, or the blend was EI + beta*sigma?
    #
    # Let's follow the directive's structural change first: PI / cost.
    # But let's also consider that PI can be very flat in regions of high uncertainty.
    # EI captures the magnitude of improvement.
    #
    # Let's try a hybrid: Score = (PI + EI_normalized) / cost?
    # Or just PI / cost as per directive.
    #
    # Let's try PI / cost.
    scores = pi / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
