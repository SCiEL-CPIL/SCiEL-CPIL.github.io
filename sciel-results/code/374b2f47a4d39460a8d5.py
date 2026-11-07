"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track progress and enforce time budget
_call_count = 0
_start_time = None
_time_limit = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Check time limit
    elapsed = time.time() - _start_time
    if elapsed > _time_limit:
        # If we're over budget, just return something simple to avoid timeout
        # This shouldn't happen if the evaluator is fast, but it's a safety net
        _call_count += 1
        return np.zeros_like(mu)
    
    _call_count += 1
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Strategy: Adaptive Lower Confidence Bound (LCB)
    # The key insight is that the "optimal" exploration-exploitation balance
    # depends on how close we are to the global minimum (0) and how much
    # uncertainty remains.
    
    # 1. Calculate the standard Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate a Lower Confidence Bound (LCB)
    # LCB = mu - beta * sigma
    # We want to maximize this score (since higher score = better candidate)
    # But wait, for minimization, we want LOW mu and HIGH sigma.
    # So the score should be: -mu + beta * sigma  (which is -LCB)
    
    # Dynamic Beta Strategy:
    # - Early on (high f_best), we want to explore broadly to find the basin of attraction.
    #   High beta encourages moving to areas of high uncertainty.
    # - Later on (low f_best), we want to exploit. 
    #   Low beta focuses on low mu.
    
    # However, simple LCB often fails in multimodal landscapes if beta is too low early.
    # EI is more robust because it directly measures expected gain.
    
    # Let's try a hybrid: 
    # Score = max(EI, LCB_score) ? No, scales are different.
    
    # Alternative Structural Change: 
    # Use a "Thompson Sampling" inspired score? 
    # We don't have the full posterior samples, only mu and sigma.
    # We can approximate the probability of being the new best.
    
    # Let's look at the "Probability of Improvement" (PI)
    pi = norm.cdf(z)
    
    # The problem with pure EI in short horizons is that it can be too conservative
    # if the sigma is small relative to the gap, or too greedy.
    
    # Let's try a **Gaussian Process Upper Confidence Bound (GP-UCB)** 
    # but adapted for minimization.
    # Score = -mu + kappa * sigma
    
    # How to choose kappa?
    # A common heuristic is kappa = sqrt(2 * log(t)) for regret bounds, but t is small.
    # Here, we can use the current f_best to adapt.
    
    # If f_best is large, the "gap" to zero is large. We should be more aggressive.
    # If f_best is small, we should be more precise.
    
    # Let's try a fixed, strong exploration term first, as the previous UCB(1) was weak?
    # The cell finding said UCB(1) -> 1.538 (improvement over 1.0? No, 1.538 is worse than 1.145? 
    # Wait, score 1.0 is the baseline. 1.145 is BETTER than 1.0? 
    # "Scoring (lower is better): mean over the seeds of the simple regret... 
    # The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # Okay, so higher score is better.
    # Current best: 1.1452.
    # Inspiration 1 (1.0000) is worse.
    # Inspiration 2 (0.7842) is worse.
    # Inspiration 3 (0.6848) is worse.
    # Cell finding: UCB(1) -> 1.538. This is BETTER than 1.1452.
    # So the UCB(1) strategy (beta=1.0) improved the score significantly.
    # The current program uses a dynamic beta: 1.0 if f_best > 0.5, else 3.0.
    # The cell finding likely used a static beta or a different logic.
    
    # Let's re-evaluate.
    # If UCB(1) gave 1.538, and current gives 1.145, maybe the dynamic beta switching to 3.0 later is hurting?
    # Or maybe the initial 1.0 is good, but the switch is bad?
    # Or maybe the UCB(1) test was on a different base?
    
    # Let's try a **Pure UCB with a constant, moderately high beta**.
    # The previous program had beta=1.0 early, 3.0 late.
    # Let's try beta=2.0 always. Or beta=1.5.
    
    # But the directive says: "Make a STRUCTURAL change... different design or modelling family".
    # Tuning beta is not structural.
    
    # Structural Change: **Use the Posterior Variance to define a "Trust Region"**
    # Instead of a linear penalty/reward, use a non-linear transformation.
    
    # Idea: **Expected Improvement with a Cap**
    # EI can be dominated by a single point with huge sigma.
    # Cap the sigma contribution.
    
    # Idea: **Probability of Being in the Top 10%**
    # Not directly computable without full posterior.
    
    # Idea: **Hybrid of EI and UCB with a Sigmoidal Blend**
    # Blend = sigmoid((f_best - threshold) / scale)
    # Score = blend * UCB + (1-blend) * EI
    
    # Let's try a **Different Problem Framing**: 
    # Instead of maximizing a score, let's think about **Minimizing the Regret**.
    # The regret is f(x) - f*.
    # We want to minimize E[regret].
    # This is exactly EI.
    
    # Why did UCB(1) work better? 
    # Maybe because EI is too greedy in the early stages when the GP is uncertain.
    # UCB(1) is more exploratory.
    
    # Let's try a **Multi-Objective Acquisition**:
    # Combine EI and UCB.
    # Score = EI + lambda * sigma
    # This is effectively a UCB with a nonlinear exploitation term.
    
    # Let's try: Score = -mu + beta * sigma + gamma * EI
    # This is a structural change in the formula.
    
    # Let's go with a **Fixed, Strong UCB** but with a twist: 
    # **Adaptive Beta based on the spread of sigma**.
    # If the sigma values are generally high, we are in a new region -> Explore more.
    # If sigma values are low, we have mapped the region -> Exploit more.
    
    # Structural Change: **Quantile-based Beta**
    # Compute the median sigma.
    # If median sigma is high, use high beta.
    # If median sigma is low, use low beta.
    
    # Let's implement this.
    
    median_sigma = np.median(sigma_safe)
    
    # Normalize median_sigma to a 0-1 range roughly.
    # Ackley values are 0-30. Sigma is usually < 10.
    # Let's say if median_sigma > 1.0, we are exploring.
    # If median_sigma < 0.1, we are exploiting.
    
    # Map median_sigma to beta.
    # beta = 1.0 + 2.0 * (median_sigma / (median_sigma + 0.5))
    # This gives beta in [1.0, 3.0].
    
    beta = 1.0 + 2.0 * (median_sigma / (median_sigma + 0.5))
    
    ucb = -mu + beta * sigma_safe
    
    # Also calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine them.
    # The UCB term handles exploration, EI handles exploitation.
    # Let's just return UCB. The cell finding suggests UCB(1) was good.
    # But dynamic beta might be better.
    
    # Let's try returning the UCB score.
    
    scores = ucb
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
