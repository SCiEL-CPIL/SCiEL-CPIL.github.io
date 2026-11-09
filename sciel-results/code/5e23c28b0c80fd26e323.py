"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300  # seconds, with 30s safety margin from 330s limit

# Pre-compute a fixed random seed for reproducibility of the sampling noise
# within the function call, but we want it to vary per call to allow exploration.
# However, since we don't have a stateful RNG inside the function easily without
# global state, we will generate noise based on the input shapes and a global
# counter or just use a fixed seed if we want deterministic behavior per call?
# Actually, Thompson Sampling requires sampling from N(mu, sigma).
# If we use the same random seed every time, it's deterministic.
# If we use a global RNG, it's stochastic.
# Let's use a global RNG to ensure proper exploration over the loop.
_rng = np.random.default_rng(42)

def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return uniform random scores to avoid timeout
        return np.ones(len(mu)) * 0.5
    
    # Thompson Sampling: Sample from the posterior predictive distribution
    # The score for each candidate is a sample from N(mu_i, sigma_i)
    # We want to MINIMIZE the function, so we want the LOWEST sampled value.
    # However, the task says "higher = evaluate next".
    # So we should return the negative of the sampled value, or a score that
    # is high when the sampled value is low.
    
    # Sample from N(mu, sigma)
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate samples
    # To make it efficient and avoid massive memory usage if 2000 is large,
    # we generate the noise once.
    noise = _rng.normal(0, 1, size=mu.shape)
    samples = mu + sigma_safe * noise
    
    # The acquisition function should return higher scores for points likely to be better (lower f).
    # So we return -samples.
    # But wait, cost is involved.
    # The prompt says: "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # And "cost" argument is the cumulative cost? Or the cost of the candidate?
    # "mu, sigma and cost have shape (2000,). Return 2000 finite scores"
    # The variable `cost` in the function signature likely represents the *predicted cost* of evaluating each candidate x_i.
    # Let's re-read: "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # The input `cost` is shape (2000,). This is almost certainly the cost of evaluating each of the 2000 candidates.
    # The previous program used `cost` as if it was cumulative or normalized.
    # If `cost` is the per-candidate cost, we should penalize high cost.
    # A common cost-aware acquisition is Cost-Effective Thompson Sampling or similar.
    # Simple approach: Score = -Sample / Cost? Or Score = -Sample + lambda * (1/Cost)?
    # Let's look at the previous program's usage:
    # `cost_norm = np.clip(cost / 30.0, 0.0, 1.0)`
    # This suggests `cost` might be the *cumulative* cost passed in? No, shape is (2000,).
    # If `cost` is shape (2000,), it must be the cost of each candidate.
    # The previous program did `ei / (cost + 1e-9)`. This implies dividing by the cost of the candidate.
    # So, high cost -> low score.
    
    # Let's use a cost-adjusted Thompson Sampling.
    # We want to minimize f(x) / cost(x) or similar?
    # Actually, the objective is to minimize f(x). The constraint is budget.
    # A standard cost-aware acquisition is often $E[I(x)] / c(x)$ or $P(I(x)) / c(x)$.
    # For Thompson Sampling, we sample $\tilde{f}(x) \sim p(f(x)|D)$.
    # We want to pick $x$ that minimizes $\tilde{f}(x)$ subject to cost.
    # A simple heuristic is to score $x$ by $-\tilde{f}(x) / c(x)$.
    # Or simply $-\tilde{f}(x) - \lambda c(x)$.
    # Given the previous program divided by cost, let's stick to a ratio-like penalty to be safe,
    # but Thompson Sampling's stochasticity is the key change.
    
    # Let's try: Score = -samples / (cost + 1e-9)
    # This heavily penalizes expensive points.
    
    # However, if `cost` is the *cumulative* cost so far (scalar broadcast to array?),
    # the shape (2000,) strongly implies per-candidate cost.
    # Let's assume `cost` is the per-candidate cost.
    
    # Normalize cost to avoid division by zero and to scale appropriately.
    # Costs are in range [1, 2] roughly (1 + mean(x in [0,1])).
    # So cost is between 1 and 2.
    
    # Let's refine the score.
    # If we just use -samples / cost, we are doing "Cost-Effective Thompson Sampling".
    
    # One issue: Thompson Sampling can be very greedy if the variance is low.
    # But it handles exploration naturally via the variance.
    
    # Let's add a small exploration bonus to ensure we don't get stuck?
    # No, TS is already stochastic.
    
    # Let's implement:
    # 1. Sample posterior: y ~ N(mu, sigma)
    # 2. Score = -y / cost
    
    # To ensure we don't pick points with extremely high cost if the sample is slightly better,
    # the division handles that.
    
    # Let's check the "higher = evaluate next" requirement.
    # Lower f is better.
    # Sample y is an estimate of f.
    # We want low y.
    # So -y is high for low y.
    # Dividing by cost: if cost is high, score is lower.
    # This seems correct.
    
    # Edge case: if cost is very small, score explodes.
    # Cost is >= 1. So it's safe.
    
    scores = -samples / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
