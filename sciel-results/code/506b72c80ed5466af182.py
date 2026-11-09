"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to maintain across calls within the same evaluation loop
_state = {
    'last_time': 0.0,
    'selected': []
}

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    # Check if this is a new evaluation loop (new problem instance)
    if start_time - _state['last_time'] > 1.0:
        _state['selected'] = []
    _state['last_time'] = start_time
    
    n_candidates = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Compute Expected Improvement (EI) without cost normalization first
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware adjustment: we prefer points with lower cost
    # But we want to balance exploration (high EI) with cost
    # Instead of dividing by cost (which can be unstable), we use a penalty
    # or we select based on a combined score.
    
    # Strategy: Batched greedy selection with diversity
    # We select 10 points per call.
    # Score for selection: EI * (1 / (1 + cost)) is one option, but let's stick to EI
    # and use diversity to avoid clustering.
    
    # To make it "cost-aware" in a stable way, we can scale EI by a factor 
    # that discounts high-cost points.
    # cost is in [1, 2] roughly.
    # Let's use: score = EI / (cost ** 0.5) or similar.
    # Or simply: The loop evaluates the BEST one. So we just need to rank them.
    # The "cost" is incurred regardless of which one is picked? No, "Evaluating x costs...".
    # The evaluator picks the best one from the 2000 based on our returned scores.
    # So we should return scores that reflect our preference.
    # If we return high scores for cheap, high-EI points, the evaluator picks those.
    
    # Let's try a simple cost-aware EI: EI / cost
    # But the current program does this and scores 1.0.
    # The problem might be that it's stateless.
    
    # Let's implement the batched selection.
    # We maintain a list of selected points (indices) in _state['selected'].
    # We want to select points that are far from already selected points in this batch.
    
    # Since we don't have coordinates, we cannot compute distance in x-space.
    # However, we can use the GP properties? No, mu/sigma are just values.
    # Wait, "No coordinates are passed."
    # This is a crucial constraint. We cannot compute distance between candidates.
    # Therefore, "diversity" in x-space is impossible to enforce directly without coordinates.
    
    # So, how to do batched selection without coordinates?
    # We can only use mu, sigma, and cost.
    # Maybe the "diversity" comes from the fact that the GP model changes?
    # But the GP model is NOT updated in the function signature.
    # The function is called with the SAME mu, sigma, f_best, cost for the 2000 candidates?
    # "the evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
    # This implies the function is called ONCE with arrays of length 2000.
    # The evaluator picks the index with the highest score.
    # So the "batched" aspect is internal to the evaluator's logic, not our function's ability to return multiple points.
    # We return 2000 scores. The evaluator picks the max.
    
    # So the "stateful" aspect can only be used to change behavior across calls.
    # But each call is independent in terms of input data (mu, sigma change as f_best updates? No, f_best is passed).
    # Wait, if f_best changes, mu/sigma might change? No, the GP is fit on observed points.
    # The problem statement says: "Each function runs on five seeded instances... While total cost is below 30... the evaluator scores 2000 new uniform candidates... and evaluates the best one."
    # This implies a loop:
    # 1. Fit GP on observed data.
    # 2. Generate 2000 candidates.
    # 3. Call acquisition(mu, sigma, f_best, cost) for these 2000.
    # 4. Pick best.
    # 5. Evaluate.
    # 6. Update GP.
    # 7. Repeat.
    
    # So the function is called multiple times, each time with NEW mu/sigma arrays (from the updated GP).
    # The "state" can track how many times we've been called, or what we've selected.
    # But since we don't see coordinates, we can't enforce diversity in space.
    
    # Alternative Strategy: Use the cost to bias exploration.
    # The current EI/cost might be too conservative or aggressive.
    # Let's try a different acquisition: Upper Confidence Bound (UCB) with cost penalty.
    # UCB = mu + kappa * sigma
    # Cost-aware UCB = (mu + kappa * sigma) / cost
    # Or: UCB - lambda * cost
    
    # Let's try a hybrid:
    # If we are early on (low cost spent), explore more (higher kappa).
    # If we are late (high cost spent), exploit more (lower kappa).
    # But we don't know "time" directly, only `cost` of current candidates and `f_best`.
    # We can track total cost in state?
    # The `cost` parameter is the cost of the 2000 candidates.
    # The total cost is not passed.
    
    # Let's look at the "inspirations".
    # Inspiration 2: "batched greedy selection with diversity enforcement".
    # But without coordinates, how?
    # Maybe the "diversity" is in the decision variable space, which we can't access.
    
    # Let's reconsider the "cost-aware" aspect.
    # Cost = 1 + mean(x_scaled).
    # x_scaled is in [0,1].
    # So cost is between 1 and 2.
    # Lower cost is better.
    
    # What if we use a simple penalty?
    # Score = EI - 0.5 * cost
    # Or Score = EI / (cost + 1)
    
    # Let's try a structural change: **Thompson Sampling**.
    # Instead of deterministic EI, we sample from the GP posterior.
    # But we only have mu and sigma. We can sample y ~ N(mu, sigma).
    # We want to minimize, so we want low y.
    # But we also want low cost.
    # Score = - (sampled_y + cost_penalty * cost)
    # This is stochastic.
    
    # Let's implement a cost-aware Thompson Sampling-like score.
    # We don't want to return random scores every time if the GP doesn't change?
    # But the GP DOES change every iteration.
    
    # Let's try:
    # 1. Compute EI.
    # 2. Apply a non-linear cost discount.
    # 3. Add a small random noise to encourage exploration? No, that might hurt.
    
    # Let's try **Probability of Improvement (PI)** with cost.
    # PI = Phi(z)
    # Cost-aware PI = PI / cost
    
    # Or **Knowledge Gradient**? Too complex.
    
    # Let's go back to **UCB**.
    # EI is prone to local optima. UCB is more exploratory.
    # Let's try UCB with a cost penalty.
    
    # Score = (mu + kappa * sigma) / cost
    # Choose kappa based on... what?
    # Maybe a fixed kappa = 2.0?
    
    # Let's try a mix: 0.5 * EI/cost + 0.5 * UCB/cost
    
    # Actually, let's look at the "stagnation" hint.
    # "Make a STRUCTURAL change: a different design or modelling family..."
    # Maybe the GP is not the issue, but the acquisition.
    
    # Let's try **Batched EI** conceptually.
    # Since we can only return one score per candidate, and the evaluator picks the max,
    # we are effectively doing single-point selection.
    
    # What if we modify the mu/sigma before computing EI?
    # No, that's cheating/invalid.
    
    # Let's try a **Cost-Adjusted UCB** with adaptive kappa.
    # We track the number of evaluations in state.
    # As evaluations increase, decrease kappa (shift from exploration to exploitation).
    
    _state.setdefault('eval_count', 0)
    _state['eval_count'] += 1
    
    # Adaptive kappa: start high, decay.
    # kappa = 2.0 / (1 + 0.1 * _state['eval_count'])
    
    # Let's use a simpler structural change:
    # **Min-Expected Improvement (MEI)**? No.
    
    # How about **Expected Regret**? Same as EI.
    
    # Let's try **Posterior Sampling** (Thompson Sampling).
    # We sample a function value from the posterior for each candidate.
    # y_i ~ N(mu_i, sigma_i)
    # We want to minimize y_i + lambda * cost_i
    # So we return scores = - (y_i + lambda * cost_i)
    # This is a valid acquisition function.
    
    # Let's implement this.
    # lambda = 1.0 (weight of cost)
    
    # To ensure stability, we can use a fixed random seed? No, that would make it deterministic.
    # Randomness is good for exploration.
    
    # One risk: Thompson Sampling can be unstable.
    # Let's combine it with EI.
    # Score = 0.5 * EI/cost - 0.5 * (sampled_y + cost)
    
    # Actually, let's just do pure cost-aware Thompson Sampling.
    # It's a different design family (stochastic vs deterministic).
    
    # Generate random samples
    # Note: We must ensure the output is finite.
    
    # Let's add a deadline guard.
    if time.time() > deadline:
        # Return a simple fallback
        return np.zeros(n_candidates)
    
    # Thompson Sampling
    # Sample from N(mu, sigma)
    # Use np.random.normal
    # To avoid issues with large sigma, we can clip?
    # No, just sample.
    
    # The "cost" is in [1, 2].
    # The objective value (Rosenbrock) can be large.
    # So cost is a small penalty.
    # We want to minimize f(x) + lambda * cost(x).
    # So we want to MAXIMIZE - (f(x) + lambda * cost(x)).
    
    # Let's try lambda = 0.5.
    
    # But wait, mu and sigma are in "objective units".
    # f_best is the lowest observed value.
    
    # Let's implement it.
    
    # To make it more robust, let's blend with EI.
    # If we are early, use more TS. If late, use more EI.
    
    # We don't know "early" or "late" perfectly, but we can use eval_count.
    
    # Let's try a fixed blend: 0.5 * EI/cost + 0.5 * TS_score
    
    # TS_score = - (sampled_y + 0.5 * cost)
    # EI_score = EI / cost
    
    # Let's normalize the scores to be comparable?
    # EI is positive. TS_score is negative (since we minimize).
    # We want higher scores to be selected.
    # So for TS, we should return - (sampled_y + penalty).
    # But sampled_y can be very negative (good). So -sampled_y is positive.
    
    # Let's refine:
    # 1. Compute EI.
    # 2. Sample y.
    # 3. Score_EI = EI / cost
    # 4. Score_TS = -(y + 0.5 * cost)  # We want to minimize y + cost, so maximize negative
    # 5. Combine: Score = 0.5 * Score_EI + 0.5 * Score_TS
    
    # However, the scales might be different.
    # EI is bounded by f_best - min(mu).
    # y is unbounded.
    
    # Let's stick to a simpler structural change: **UCB with Cost**.
    # UCB is deterministic and often outperforms EI in multimodal landscapes.
    # Rosenbrock is multimodal (in 2D, it's a valley).
    
    # Score = (mu + kappa * sigma) / cost
    # Let's try kappa = 2.5.
    
    # And add a small exploration term based on state?
    
    # Let's try this:
    # Score = (mu + 2.5 * sigma) / cost
    
    # This is a different design family (UCB vs EI).
    
    # Let's implement this.
    
    kappa = 2.5
    ucb = mu + kappa * sigma
    score = ucb / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If time is running out, return quickly
    if time.time() > deadline:
        return score
    
    return score

# EVOLVE-BLOCK-END
