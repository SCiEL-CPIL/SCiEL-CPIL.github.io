"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Global state to track iteration count for dynamic exploration
_call_count = 0
_MAX_COST = 30.0
_INITIAL_COST = 5.0 # 5 points * approx 1.5 cost? Actually cost is 1+mean(x). 
# The prompt says "Runs start with 5 uniform random points (cost counted)."
# We don't know the exact initial cost, but we can estimate or just use total cost passed in.
# However, 'cost' is an array of costs for the 2000 candidates, not the total spent.
# The prompt says: "While total cost is below 30... f_best is the lowest observed value."
# We need to estimate how much budget is left.
# We can track the number of calls. Each call adds roughly 1.0 to 2.0 cost.
# Let's assume average cost per eval is ~1.5.
# Total budget 30. Initial 5 points.
# If we have made N evaluations, remaining budget is approx 30 - N*1.5.

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    
    # Estimate remaining budget fraction
    # We don't know exact past costs, but we can use the number of calls as a proxy.
    # Each acquisition call corresponds to one evaluation in the loop.
    # The loop runs while total cost < 30.
    # Let's assume a linear decrease in "exploration weight" as we approach the limit.
    # A simple heuristic: if we have done ~15-20 evaluations, we should be more exploitative.
    
    _call_count += 1
    
    # Normalize sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Standardize the improvement potential
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Base Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-awareness:
    # We want to prefer points with lower cost, but not exclusively.
    # The 'cost' input is the cost of evaluating each of the 2000 candidates.
    # cost = 1 + mean(x_scaled). Range [1, 2].
    # We can normalize cost to [0, 1] where 0 is cheapest (cost=1) and 1 is most expensive (cost=2).
    min_cost = np.min(cost)
    max_cost = np.max(cost)
    if max_cost > min_cost:
        cost_norm = (cost - min_cost) / (max_cost - min_cost)
    else:
        cost_norm = np.zeros_like(cost)
        
    # Dynamic exploration parameter
    # Early on: high exploration (Thompson Sampling like)
    # Late on: high exploitation (EI like)
    
    # Estimate progress. 
    # If _call_count is small, we are early.
    # Max calls approx (30 - initial_cost) / avg_cost. 
    # Let's assume we have roughly 15-20 calls total.
    # progress = _call_count / 20.0
    # exploration_weight = 1.0 - 0.5 * progress  # Keep at least 0.5 weight on EI?
    
    # Alternative Structural Change:
    # Instead of just EI/Cost, use a "Probability of Improvement" (PI) blended with Uncertainty.
    # Or, better: Sample from the posterior for a subset of top EI candidates? 
    # No, we must return scores for all 2000.
    
    # Let's try a "Cost-Adjusted UCB" (Upper Confidence Bound for minimization, so Lower Confidence Bound)
    # LCB = mu - beta * sigma
    # But we need to handle cost.
    
    # The prompt asks for a DIFFERENT strategy.
    # Current: EI / Cost.
    # Inspiration 1 (1.32): Used numpy/scipy/time.
    # Inspiration 3 (0.98): Thompson Sampling (median of posterior).
    
    # Let's implement a deterministic approximation of Thompson Sampling combined with Cost.
    # Score = mu - k * sigma / (cost + epsilon)
    # This is essentially UCB per unit cost.
    # To make it structurally different from EI/Cost:
    # EI is a nonlinear function of z. UCB is linear in mu and sigma.
    # UCB tends to explore more broadly in early stages.
    
    # Let's use a dynamic beta that starts high and decays.
    # However, we don't know the exact number of iterations remaining.
    # We can use the magnitude of sigma as a proxy for uncertainty.
    
    # Strategy: "Risk-Adjusted Cost-Effective Search"
    # Score = (f_best - mu) / (sigma + cost_penalty)
    # This is similar to EI but linear.
    
    # Let's try: Score = mu - beta * sigma * (1 / cost)
    # This favors low mu (good value) and high sigma (uncertainty) and low cost.
    
    # To make it more robust, let's use a "Hybrid" that switches behavior.
    # But the directive says "Do not micro-tune constants".
    
    # Let's look at the "Stronger cost-aware normalization" finding.
    # "offset 0.6 vs 0.5 increases relative weight of low-cost candidates".
    
    # Let's try a different family: **Knowledge Gradient (KG)** approximation.
    # KG is hard to compute exactly for 2000 points without heavy integration.
    # A common approximation is EI.
    
    # How about **Entropy-based acquisition**?
    # Reduce uncertainty in the region of the minimum.
    # Score = sigma * (f_best - mu) / cost ?
    
    # Let's go with **UCB (Lower Confidence Bound) with Cost Normalization**.
    # This is structurally different from EI because it does not integrate the distribution tail; it uses a linear combination of mean and std.
    # It is often more robust in early stages when the GP is poorly fit.
    
    # Dynamic Beta:
    # Start with beta=1.0 (high exploration) and decay to 0.5 (exploitation).
    # We can estimate the decay based on _call_count.
    # Assuming ~20 total calls:
    progress = min(_call_count / 20.0, 1.0)
    beta = 1.0 - 0.5 * progress
    
    # Calculate LCB (Lower Confidence Bound for minimization)
    # We want to minimize f(x). So we look for low mu and high sigma.
    # lcb = mu - beta * sigma
    lcb = mu - beta * sigma_safe
    
    # Cost adjustment:
    # We want to penalize high cost.
    # score = lcb / (cost ** gamma)
    # Or score = lcb - penalty * cost
    # Division is more sensitive to cost differences.
    
    # Let's use: score = (mu - beta * sigma) / (cost + 0.1)
    # Note: mu can be negative (rastrigin min is 0, but values can be higher).
    # Wait, rastrigin min is 0. Values are >= 0.
    # So mu is likely positive.
    # We want to MINIMIZE the score? No, "Return 2000 finite scores (higher = evaluate next)".
    # So we want HIGHER score for BETTER candidates (lower function value).
    
    # If we use LCB = mu - beta*sigma, a candidate with low mu and high sigma has LOW lcb.
    # We want HIGH score.
    # So we should invert it or use -LCB.
    # score = -lcb / cost = (beta*sigma - mu) / cost
    
    # Let's refine:
    # score = (beta * sigma_safe - mu) / (cost + 1e-6)
    
    # This favors:
    # 1. High sigma (uncertainty)
    # 2. Low mu (current best estimate)
    # 3. Low cost
    
    # This is a structurally different "Optimistic" strategy compared to the "Expected Improvement" which is "Conservative" in a probabilistic sense (it integrates the probability of improvement).
    
    # One issue: If mu is very high (bad), beta*sigma might not overcome it.
    # If mu is very low (good), score becomes negative?
    # If mu < beta*sigma, score is positive.
    # If mu > beta*sigma, score is negative.
    # We should ensure scores are comparable.
    # Adding a constant offset?
    # Or just using the raw value. The evaluator picks the max.
    
    # Let's add a small exploration noise term to break ties and encourage diversity?
    # No, keep it deterministic for reproducibility in the loop if possible, or just rely on the GP variance.
    
    # Let's try a **Weighted Sum of LCB and Cost-Adjusted EI**?
    # No, stick to one clear family.
    
    # Let's go with **Cost-Effective UCB**.
    
    scores = (beta * sigma_safe - mu) / (cost + 1e-6)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
