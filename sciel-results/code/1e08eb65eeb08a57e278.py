"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Dynamic blend of EI (exploitation) and UCB (exploration)
    # As evaluations progress (indicated by f_best improving or just generally),
    # we shift weight towards exploitation.
    
    # Estimate progress: 
    # f_best starts high (bad) and goes low (good).
    # We don't know the exact number of iterations, but we can infer it from the "tightness" of sigma or just use a fixed schedule.
    # However, a more robust structural change is to use a "Softmax" style aggregation or a dynamic beta.
    
    # Let's use a dynamic beta based on the relative uncertainty.
    # Early on, sigma is high relative to (f_best - mu).
    # Late on, sigma is low.
    
    # Actually, a simpler structural change: 
    # Pure EI is too greedy. Pure UCB is too random.
    # Let's use a "Confidence-based" switch.
    
    # Calculate EI
    z_ei = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB
    # Beta needs to be tuned. 
    # If we use a fixed beta, it might be suboptimal for different functions.
    # Let's try a beta that decays or is fixed but distinct from EI.
    # A common choice is beta=2.0 or beta=3.0.
    # But to be structurally different, let's use a "Thompson Sampling" like heuristic or a "Probability of Improvement" mix.
    
    # Alternative Structural Change: 
    # Instead of EI/cost, use "Expected Improvement" but WITHOUT the cost division, 
    # and add a small exploration bonus based on sigma.
    # The cost division in the seed program essentially says "prefer cheap points".
    # But the GP is trained on standardized coords. The cost is 1 + mean(x_scaled).
    # Points near the center (0.5, 0.5) are cheaper. Points near corners are expensive.
    # If the optimum is in the center, cost-normalization helps.
    # If the optimum is in the corner, cost-normalization hurts.
    # The task says minimizers are moved to uniform points in inner 80%.
    # So optimizers can be anywhere in [0.1, 0.9] roughly.
    
    # Let's try a hybrid that doesn't divide by cost, but uses a "confidence" term.
    # Score = EI + beta * sigma
    # This is UCB.
    
    # Let's try a "Weighted Sum" of normalized EI and UCB.
    # Or simply, let's abandon cost normalization entirely and use a tuned UCB.
    # Why? Because cost normalization creates a bias towards the center.
    # If the true minimum is at 0.8, cost normalization will penalize it.
    # Pure UCB explores based on uncertainty, not cost.
    
    # Let's use a UCB with a beta that adapts slightly or is just a strong exploration term.
    # Beta = 2.5 is a reasonable default.
    
    beta = 2.5
    ucb = mu + beta * sigma
    
    # To ensure we return finite scores and handle edge cases:
    # UCB can be negative or positive.
    # EI is always non-negative.
    # We want higher = better.
    
    # Let's try a "Soft" combination:
    # If we just return UCB, we might ignore f_best improvements if sigma is high elsewhere.
    # But UCB is a standard, robust BO acquisition function.
    
    # Let's try to blend EI and UCB based on the "progress" proxy.
    # Since we don't have the iteration count, let's look at the scale of sigma.
    # If sigma is very small everywhere, we should exploit.
    # If sigma is large, explore.
    
    # Actually, the simplest structural change that often beats EI/cost is just **UCB**.
    # EI/cost is a specific variant. UCB is a different family (confidence-bound based).
    
    # Let's implement UCB with a fixed beta.
    # To be safe, we can clip sigma to avoid huge values if GP is weird, but GP should be fine.
    
    scores = mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
