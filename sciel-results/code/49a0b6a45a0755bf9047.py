"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300  # seconds, with 30s safety margin from 330s limit

# Global RNG for Thompson Sampling
_rng = np.random.default_rng(12345)

def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return uniform random scores to avoid timeout
        return np.random.uniform(size=len(mu))
    
    # Thompson Sampling: sample from N(mu, sigma)
    # We sample a single realization of the function at all candidate points
    noise = _rng.standard_normal(len(mu))
    sample = mu + sigma * noise
    
    # Cost-aware adjustment:
    # The cost of evaluating x is 1 + mean(x_scaled). 
    # We don't have x, but we have `cost` which is the cumulative total cost so far.
    # However, the prompt says "Evaluating x costs 1 + mean(x scaled to [0, 1])".
    # The `cost` parameter in the function signature is described as having shape (2000,).
    # Let's re-read carefully: "mu, sigma and cost have shape (2000,)."
    # This implies `cost` is the *individual* cost of each candidate x, not the cumulative total.
    # "Runs start with 5 uniform random points (cost counted). While total cost is below 30..."
    # So `cost[i]` is the cost of evaluating candidate i.
    
    # We want to minimize the expected regret, but with a cost penalty.
    # A simple heuristic: penalize high-cost candidates more heavily when the predicted value is close to f_best.
    # Or, simply subtract a fraction of the cost from the sample to prefer cheaper points if the quality is similar.
    
    # Let's use a "cost-adjusted" score.
    # We want to maximize the score.
    # Raw sample is a predicted value (lower is better).
    # So we want to minimize `sample + penalty`.
    # Or maximize `-sample - penalty`.
    
    # Let's try: score = -sample - beta * cost
    # Beta should be small enough that we don't ignore the function value, but large enough to break ties in favor of cheaper points.
    # The cost is in [1, 2]. The function values can be large (Rosenbrock).
    # If f_best is close to 0, the differences in `sample` might be small.
    
    # Let's normalize the sample relative to f_best?
    # No, just use the raw sample.
    
    # A robust approach:
    # score = -sample
    # If we want to incorporate cost, we can do:
    # score = -sample - 0.01 * cost
    # But 0.01 might be too small or too large depending on the scale of `sample`.
    
    # Let's look at the scale.
    # Rosenbrock min is 0. Values can be up to ~100+ in the domain [-2,2]^2.
    # Cost is [1, 2].
    # If sample differences are ~1, then 0.01 * cost difference (max 1) is 0.01, which is negligible.
    # If sample differences are ~0.1, then 0.01 is significant.
    
    # Let's try a dynamic penalty.
    # If we are close to convergence (f_best is small), we might want to be more cost-efficient.
    # But since we don't have x, we can't know if we are "close" in space.
    
    # Let's stick to a simple weighted sum.
    # Since the goal is to minimize the observed value, and cost is a constraint on the total budget,
    # minimizing the number of evaluations (by picking cheaper ones when uncertain) helps extend the search.
    
    # Let's try: score = -sample / (1 + 0.1 * cost)
    # This slightly dampens the score for higher cost points.
    
    # Alternatively, just use the sample and let the GP handle the exploration.
    # The GP's sigma already accounts for uncertainty.
    # If we just return -sample, we are doing pure exploitation of the sample.
    # This is standard Thompson Sampling.
    
    # Let's try adding a small exploration bonus based on sigma, which is already in the sample.
    # The sample *is* the exploration.
    
    # Let's try: score = -sample
    # And see if that works. If not, we can adjust.
    # But we need to be careful with the cost.
    # The prompt says "cost-aware acquisition".
    
    # Let's try: score = -sample - 0.05 * cost
    # This is a linear penalty.
    
    score = -sample - 0.05 * cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
