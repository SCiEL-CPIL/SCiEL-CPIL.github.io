"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Strategy: Cost-Aware Expected Improvement (CAEI)
    # Standard EI is (f_best - mu) * Phi(z) + sigma * phi(z)
    # Cost-aware EI subtracts a penalty proportional to the cost.
    # To be robust, we normalize the cost and EI to have similar scales.
    
    # 1. Compute z-scores
    sigma_safe = np.clip(sigma, 1e-9, None)
    z = (f_best - mu) / sigma_safe
    
    # 2. Compute standard EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) = z * sigma
    # EI = z * sigma * Phi(z) + sigma * phi(z) = sigma * (z * Phi(z) + phi(z))
    
    # Vectorized computation of EI
    # phi(z) = norm.pdf(z)
    # Phi(z) = norm.cdf(z)
    
    # To avoid overflow/underflow issues with extreme z, we can use stable formulas,
    # but for typical BO scenarios, standard scipy is fine.
    
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    ei = sigma * (z * cdf + pdf)
    
    # 3. Cost normalization and penalty
    # We want to penalize high cost.
    # Let's normalize cost to [0, 1] range relative to the current candidate set.
    # Or use a fixed scale. The cost is 1 + mean(x_scaled). So cost is in [1, 2].
    # This is a very narrow range.
    # A linear penalty: score = EI - lambda * cost
    # What is a good lambda?
    # If EI is in objective units, and cost is dimensionless (or time units),
    # we need to scale them.
    
    # Let's look at the scale of EI.
    # f_best is the best observed value.
    # sigma is the standard deviation.
    # In early stages, f_best might be large (e.g., 100), sigma might be large.
    # In late stages, f_best is close to 0, sigma is small.
    
    # A robust way is to normalize EI by the current scale of the problem.
    # However, we don't have the full history.
    # We can use f_best as a proxy for scale, but if f_best is 0, we have issues.
    # Let's use a combination.
    
    # Alternative: Use the "Probability of Improvement" (PI) which is scale-free (0 to 1).
    # PI = cdf
    # Cost is in [1, 2].
    # If we use score = PI - lambda * cost, we need lambda to be small enough.
    # If lambda = 0.1, cost penalty is 0.1 to 0.2. PI is 0 to 1.
    # This seems reasonable.
    
    # Let's try a hybrid:
    # If f_best is large, EI is large, so we can afford a larger lambda.
    # If f_best is small, EI is small, so we need a smaller lambda.
    
    # Let's estimate the "value" of one unit of cost.
    # Suppose we want to improve by 10% of f_best per unit of cost?
    # This is getting heuristic.
    
    # Let's stick to the PI strategy which performed well (2.43) but refine the cost handling.
    # The previous program used PI / (1 + cost).
    # Since cost is in [1, 2], 1+cost is in [2, 3].
    # So it's basically PI / 2.5 roughly.
    # This effectively scales PI down.
    
    # Let's try: score = PI * (1 / cost)
    # Or score = PI - 0.1 * cost
    
    # Let's look at the inspiration: "PI/cost" improved to 2.743.
    # The current program IS PI/cost (score = pi / (1.0 + cost)).
    # Wait, the prompt says "CURRENT PROGRAM (score 2.4373)" uses `pi / (1.0 + cost)`.
    # And the knowledge says "Replaces EI with PI/cost... Measured: 1 -> 2.743".
    # This suggests that PI/cost is a good direction.
    
    # How to improve further?
    # 1. Maybe the cost penalty is too strong or too weak?
    # 2. Maybe PI is not the best metric?
    # 3. Maybe we should combine PI and EI?
    
    # Let's try a weighted sum of PI and normalized EI.
    # But normalization is tricky.
    
    # Let's try adjusting the cost sensitivity.
    # If cost is [1, 2], maybe we should penalize it less?
    # score = pi / (0.5 + cost) ? No, that increases the score.
    # score = pi * (2 - cost) ?
    # If cost=1, factor=1. If cost=2, factor=0.
    # This aggressively avoids expensive points.
    
    # Let's try: score = pi * (1 / (1 + 0.1 * (cost - 1)))
    # This makes the penalty softer.
    
    # Actually, let's look at the "GPOC" idea more seriously.
    # The optimal acquisition function for cost-aware BO is often derived as:
    # max_x [ E[improvement] - lambda * cost(x) ]
    # where lambda is chosen to balance exploration and cost.
    
    # Let's compute EI and subtract a scaled cost.
    # We need to scale EI to be comparable to cost.
    # Let's use the mean of EI as a scale factor?
    # Or just use a fixed lambda that works for the Rosenbrock scale.
    
    # Rosenbrock values:
    # Min 0.
    # Initial points might be 10-100.
    # So EI can be 10-100.
    # Cost is 1-2.
    # So lambda should be around 0.1 - 1.0.
    
    # Let's try lambda = 0.5.
    # score = EI - 0.5 * cost
    
    # But EI can be negative if mu > f_best (though z would be negative, EI formula handles it?
    # If z < 0, cdf < 0.5, pdf > 0.
    # EI = sigma * (z*cdf + pdf).
    # If z is very negative, z*cdf is large negative, pdf is small positive.
    # EI can be negative.
    # We should clip EI to 0?
    # Standard EI is max(0, ...).
    
    ei_clipped = np.clip(ei, 0, None)
    
    # Let's try a mix of PI and EI.
    # PI is robust to scale. EI is sensitive to scale.
    
    # Let's try: score = 0.5 * pi + 0.5 * (ei_clipped / (f_best + 1e-6))
    # This normalizes EI by the current best value.
    # If f_best is large, EI is normalized to ~1.
    # If f_best is small, EI is normalized to ~1.
    
    # And then subtract cost?
    # score = 0.5 * pi + 0.5 * (ei_clipped / (f_best + 1e-6)) - 0.1 * cost
    
    # This seems reasonable.
    
    # Let's implement this.
    
    # Normalized EI
    norm_ei = ei_clipped / (f_best + 1e-6)
    
    # Combine PI and Norm EI
    # Weights: 0.5 each?
    # Maybe weight PI more in early stages?
    # We don't know the stage.
    # But f_best gives a hint.
    # If f_best > 10, we are early.
    # If f_best < 1, we are late.
    
    # Let's use a dynamic weight.
    # weight_pi = 1 / (1 + f_best) ?
    # If f_best=10, w_pi=0.09, w_ei=0.91.
    # If f_best=0.1, w_pi=0.91, w_ei=0.09.
    
    # This might be good. Early on, EI is noisy and scale-dependent, PI is robust.
    # Wait, if f_best is large, EI is large too.
    # Maybe we want to exploit more when we have a good f_best?
    
    # Let's stick to a simple weighted sum.
    # score = pi + norm_ei - 0.1 * cost
    
    # Let's try this.
    
    score = pi + norm_ei - 0.1 * cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - start_time > 320:
        pass
        
    return score

# EVOLVE-BLOCK-END
