"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-awareness:
    # The evaluator minimizes simple regret.
    # We have a total cost budget of 30.
    # Initial cost is roughly 5 * (1 + 0.5) = 7.5 (assuming mean x is 0.5).
    # So we have ~22.5 cost units left for ~15-20 evaluations.
    # We want to explore early and exploit late.
    
    # Adaptive Beta for GP-UCB
    # Beta(t) = sqrt(log(t + 1)) is standard, but here we have a finite budget.
    # Let's define progress p = (current_cost - min_cost) / (max_cost - min_cost)
    # However, we don't know max_cost exactly, but we know the limit is 30.
    # Let's assume the cost passed is the cumulative cost of the *next* evaluation?
    # The contract says: "cost has shape (2000,)". "Evaluating x costs 1 + mean(x...)"
    # "Runs start with 5 uniform random points (cost counted)."
    # "While total cost is below 30... evaluates the best one."
    # The `cost` array likely contains the *incremental* cost of each candidate x.
    # Why? Because x is a point in [0,1]^3. cost(x) = 1 + mean(x).
    # If `cost` was cumulative, it would be a scalar or the same for all candidates in a batch step?
    # No, the loop evaluates ONE point per iteration.
    # So `cost` array contains the cost of evaluating each of the 2000 candidates.
    
    # Let's verify the interpretation of `cost`.
    # If `cost` is the incremental cost of evaluating x_k, then:
    # score_k = Acquisition(x_k) / cost_k is a common cost-aware metric.
    
    # Adaptive Exploration:
    # We need to know how much budget is left.
    # The function signature doesn't provide the *current* total cost, only the candidate costs.
    # Wait, `f_best` is the lowest observed value.
    # Can we infer the number of evaluations? No.
    # Can we infer the current total cost? No.
    
    # However, we can look at the distribution of `cost`.
    # If the `cost` array is the incremental cost, it's determined solely by x.
    # We cannot directly know the global budget state from the inputs provided (mu, sigma, f_best, cost).
    
    # BUT, we can use a heuristic based on the *potential* remaining budget.
    # Or, we can stick to a static but well-tuned UCB, or use a heuristic based on the 
    # "spread" of mu/sigma which might indicate convergence?
    # Actually, a simpler structural change is to use the **Cost-Aware UCB** with a 
    # fixed but optimized beta, or a beta that scales with the *variance* of the GP.
    
    # Let's try a different structural angle: 
    # The prompt suggests: "adaptive beta parameter that decreases as the total cost budget is consumed".
    # Since we don't have the total cost, we can't do this directly.
    # UNLESS `cost` in the signature is the *cumulative* cost up to that point? 
    # "cost has shape (2000,)". This implies it varies per candidate. So it must be incremental cost.
    
    # Alternative Strategy: 
    # Use **Probability of Improvement (PI)** combined with Cost? 
    # Or **Thompson Sampling**? 
    # Thompson Sampling is often better for finite budgets as it naturally balances exploration/exploitation 
    # without needing a manually tuned beta.
    
    # Let's implement Thompson Sampling with Cost-Awareness.
    # 1. Sample a function value from the posterior N(mu, sigma^2) for each candidate.
    # 2. Select the candidate with the lowest sampled value.
    # 3. To make it cost-aware, we can penalize the sampled value by the cost, 
    #    or select based on Sampled_Value / Cost?
    #    Dividing random samples by cost is statistically tricky.
    #    Better: Sample f(x). Then minimize f(x) + lambda * cost(x).
    #    Since cost(x) is deterministic, this is equivalent to minimizing f(x) - lambda*cost(x) 
    #    if we were maximizing, but we are minimizing.
    #    So we want to minimize f(x). The cost is a constraint/penalty.
    #    If we simply pick argmin(sampled_f), we ignore cost.
    #    If we pick argmin(sampled_f / cost), we favor cheap points with low values.
    
    # Let's try: Score = Sampled_Value / Cost.
    # Higher score is better? 
    # If we want to MINIMIZE f, we want LOW f.
    # The contract says: "Return 2000 finite scores (higher = evaluate next)".
    # So we need to convert "minimize f" to "maximize score".
    # Score = - Sampled_Value / Cost? 
    # If f is negative (Hartmann 3d min is -3.86), -f is positive.
    # If cost is ~1.5, -f/cost is large for good (low f) and cheap points.
    # This seems like a very strong cost-aware exploitation strategy.
    
    # However, Thompson Sampling requires a random seed or state to be consistent?
    # No, it just draws samples.
    # The issue: Is it better to explore? 
    # TS explores by sampling high sigma regions. If sigma is high, the sample can be very low (good) 
    # or very high (bad). The expectation of the score is not simply EI.
    # But TS is known to perform well in finite budgets.
    
    # Let's refine the Score for TS:
    # We want to maximize the probability of being the best, adjusted for cost.
    # A simpler cost-aware TS variant:
    # sample_f = np.random.normal(mu, sigma_safe)
    # score = -sample_f / cost
    # This heavily penalizes expensive points.
    
    # Let's check the deadline.
    # Generating 2000 normal samples is fast.
    
    # One concern: TS can be noisy. 
    # Let's try a **Cost-Aware UCB** first, as it is more deterministic and stable, 
    # but use a **Beta** that is scaled by the **Standard Deviation of the GP** to be adaptive to the model's confidence?
    # No, the prompt specifically asked for "adaptive beta ... as total cost budget is consumed".
    # Since I can't access total cost, I will implement **Thompson Sampling** as the structural change, 
    # which is a different optimization method (Bayesian optimization via sampling vs. greedy acquisition).
    
    # Implementation of Cost-Aware Thompson Sampling:
    # 1. Draw samples from the posterior.
    # 2. Calculate a score that favors low function values and low costs.
    #    Score = -sampled_value / cost
    
    # To ensure we don't get stuck in a local minimum due to bad luck in sampling, 
    # we can blend with a small amount of EI? 
    # No, keep it pure for the structural change.
    
    # Wait, if I use random sampling, the result is non-deterministic. 
    # The evaluator runs 5 instances. This is fine.
    
    # Let's ensure the scores are finite.
    
    # One detail: "minimization".
    # Hartmann 3d values are negative.
    # f_best is negative.
    # mu is negative.
    # sigma is positive.
    # sample = mu + sigma * z.
    # If we minimize f, we want small f.
    # Score = -f / cost.
    # If f = -3.8, cost = 1.5, Score = 2.53.
    # If f = -1.0, cost = 1.0, Score = 1.0.
    # We want to maximize Score. So we pick the candidate with the highest -f/cost.
    # This effectively minimizes f/cost.
    
    # Is minimizing f/cost the right objective?
    # It prioritizes cheap solutions. 
    # If a cheap solution has f=0 and an expensive solution has f=-3.8, 
    # cheap: 0/1.5 = 0. Expensive: -3.8/2.0 = -1.9. 
    # -f: 0 vs 1.9. We pick expensive. Correct.
    # If cheap f=-1.0, expensive f=-1.2.
    # Cheap: 1.0/1.0 = 1.0. Expensive: 1.2/2.0 = 0.6.
    # We pick cheap. This is cost-aware.
    
    # Let's implement this.
    
    # To avoid division by zero in cost (cost is >= 1.0, so safe), but let's be safe.
    cost_safe = np.maximum(cost, 1e-9)
    
    # Draw samples
    # Use a fixed seed? No, standard TS uses random.
    # But for reproducibility in debugging, maybe? No, the task allows randomness.
    
    # However, pure TS can be very aggressive or very conservative.
    # Let's add a small exploitation bias by blending the sample with the mean?
    # No, pure TS is the structural change.
    
    # Let's try to make it slightly more robust by using the **Expected** score of the sample?
    # E[-f/cost] is not -E[f]/cost.
    # Let's stick to the sample.
    
    # One potential issue: If sigma is very large, the sample can be extremely high (bad), 
    # leading to a very negative score. This is correct, we avoid it.
    # If sigma is large, the sample can be extremely low (good), leading to a very high score. 
    # This encourages exploration.
    
    # Let's refine the score to be:
    # score = -sampled_value / cost_safe
    
    # Check for finite values
    sampled_values = mu + sigma_safe * np.random.randn(2000)
    scores = -sampled_values / cost_safe
    
    # Handle non-finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Fallback to simple EI if taking too long (unlikely for 2000 points)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei / cost_safe), ei / cost_safe, -np.inf)
        
    return scores

# EVOLVE-BLOCK-END
