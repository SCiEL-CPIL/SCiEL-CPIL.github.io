"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition using a proxy for Expected Hypervolume Improvement (EHVI)
    # or a robust Expected Improvement variant that handles the cost structure better.
    
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # However, simple EI can be too aggressive or too passive depending on the scale.
    # The cost here is 1 + mean(x_scaled). Since we minimize, and cost is higher 
    # for points further from the origin (in scaled space), we want to be careful.
    
    # Let's try a variant that penalizes high cost more strongly if the improvement is marginal.
    # Or, use the "Knowledge Gradient" approximation? No, too complex.
    
    # Let's stick to EI but with a specific tuning for the cost-awareness.
    # The current program divides EI by cost.
    # EI is in objective units. Cost is dimensionless (1 to 2).
    
    # A potential issue with EI/cost: if f_best is very good, EI is small.
    # If sigma is large, EI is large.
    
    # Let's try "Probability of Improvement" (PI) weighted by potential gain?
    # PI = Phi(z)
    # Score = PI * (f_best - mu) / cost? No, that's just EI without the sigma term.
    
    # Let's try a "Cost-Adjusted LCB" (Lower Confidence Bound).
    # LCB = mu - beta * sigma
    # We want to minimize LCB, so we maximize -LCB.
    # Score = -(mu - beta * sigma) / cost
    # Choosing beta is critical. For Ackley, which has a global minimum at 0, 
    # and local minima, we need enough exploration.
    
    # The prompt mentions "Thompson Sampling" as a strategy to try in the change directive.
    # Let's implement a deterministic surrogate for Thompson Sampling or a hybrid.
    # Actually, the directive says: "Implement a 'Thompson Sampling' variant by sampling a single scalar offset..."
    # But wait, the function is called multiple times in the loop? 
    # "Each function runs on five seeded instances... While total cost is below 30, the evaluator scores 2000 new uniform candidates..."
    # The acquisition function is called once per iteration of the inner loop (scoring 2000 points).
    # If we sample a random offset inside `acquisition`, it will be different for each of the 2000 points?
    # No, `acquisition` takes arrays. If I sample a scalar offset, it should be consistent for the batch?
    # The signature is `acquisition(mu, sigma, f_best, cost)`. 
    # If I use `np.random`, it might be non-deterministic across calls unless seeded.
    # The task says "five seeded instances". It doesn't explicitly say the acquisition function must be deterministic, 
    # but usually, acquisition functions are deterministic given the GP state. 
    # However, Thompson Sampling is inherently stochastic.
    
    # Let's look at the "Inspiration 1" which scored 3.45. It used LCB with cost-awareness.
    # Let's try a different structural change: **Expected Improvement with a dynamic beta for exploration.**
    # Or better, let's follow the specific directive in the prompt: 
    # "Strategy to try in this change: Implement a 'Thompson Sampling' variant by sampling a single scalar offset from a Gaussian distribution with mean f_best and a standard deviation that scales with the maximum uncertainty (max(sigma)) in the candidate set, then scoring each candidate as -(mu - offset) / cost."
    
    # This specific strategy is suggested. Let's implement it carefully.
    # We need to be careful about randomness. If the evaluator calls this function multiple times 
    # (once per iteration of the BO loop), the random offset will change each time, which is fine for TS.
    # But within one call, we score 2000 points. The offset should be fixed for that call.
    
    # However, there is a risk: if the random offset is bad, we might explore poorly.
    # A deterministic equivalent is often safer for scoring consistency unless the evaluator supports stochastic acquisition.
    # The prompt says "Return 2000 finite scores". It doesn't forbid randomness.
    
    # Let's implement the suggested TS variant.
    
    # 1. Calculate max uncertainty in the current candidate set
    max_sigma = np.max(sigma)
    
    # 2. Sample offset
    # Standard deviation of the offset should scale with max_sigma.
    # Let's use a factor, e.g., 1.0 * max_sigma.
    # The offset represents a "virtual best" or "target" we are trying to beat?
    # The formula given: score = -(mu - offset) / cost
    # This is equivalent to maximizing (offset - mu) / cost.
    # If offset > f_best, we are prioritizing points with low mu (exploitation towards the offset).
    # If offset < f_best, we are prioritizing points with high mu? No, that would be bad.
    # Wait, if offset is sampled from N(f_best, max_sigma), it can be larger or smaller than f_best.
    # If offset > f_best, we look for mu < offset.
    # If offset < f_best, we look for mu < offset (which is harder, requiring better points).
    # This effectively randomizes the "greediness" of the search.
    
    # Let's refine the std. If max_sigma is 0, we should not sample.
    if max_sigma < 1e-8:
        offset = f_best
    else:
        # Sample from N(f_best, max_sigma)
        # Using np.random is allowed? Standard library, NumPy, SciPy. Yes.
        # But is it safe? If the same function is called for different instances, 
        # the random state might affect reproducibility if not handled.
        # The problem says "five seeded instances". It implies the whole run is seeded.
        # Let's use a random offset.
        offset = np.random.normal(f_best, max_sigma)
        
    # 3. Calculate score
    # Score = (offset - mu) / cost
    # We want higher score = evaluate next.
    # If offset is large, we prefer low mu.
    # If cost is high, we penalize.
    
    # To avoid division by zero or very small costs (cost >= 1), this is safe.
    scores = (offset - mu) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
