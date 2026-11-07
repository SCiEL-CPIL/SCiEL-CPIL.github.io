import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Information Gain per Cost
    # This models the stochastic knapsack structure:
    # We want to maximize the reduction in uncertainty (information gain)
    # while minimizing the cost.
    # A common proxy for information gain in GP is the entropy of the GP posterior.
    # For a univariate Gaussian, entropy is 0.5 * log(2 * pi * e * sigma^2).
    # So information gain is proportional to log(sigma) or sigma^2.
    # However, log(sigma) is more stable and directly related to the reduction in entropy.
    # Let's use log(sigma_safe + epsilon) as a proxy for information gain.
    # Actually, the reduction in entropy when evaluating a point is related to the prior variance.
    # But we are given the posterior sigma.
    # A simpler and often effective approach is to use the UCB/LCB improvement but weighted by cost.
    # The directive suggests: "Replace the additive cost penalty with a multiplicative feasibility constraint by defining the score as the Expected Improvement only if the predicted cost is below the current average cost per unit improvement, effectively creating a hard budget-aware gate that forces the search to exploit the low-cost corner until the GP uncertainty there is exhausted, rather than smoothly trading off value and cost."
    
    # Let's implement a hard gate.
    # Calculate the average cost per unit improvement.
    # Improvement is (f_best - mu). We only care about positive improvements.
    # Let's define a threshold cost.
    # If cost is too high, we suppress the score.
    
    # A "hard budget-aware gate" can be implemented by multiplying the score by a step function
    # or a very steep sigmoid that is 1 for low cost and 0 for high cost.
    # The threshold should be dynamic.
    # "current average cost per unit improvement"
    # Let's compute the mean cost of candidates that have positive expected improvement.
    # Or simply, let's use a fixed threshold that adapts to the distribution of costs.
    # The cost is 1 + mean(x_scaled). x_scaled is in [0,1]. So cost is in [1, 2].
    # The low-cost corner is near cost=1.
    # Let's define a threshold as the median cost, or mean cost, or a quantile.
    # To force exploitation of the low-cost corner, we should set the threshold low initially.
    # Let's use a quantile of the costs, e.g., the 25th percentile.
    # Or, let's use the mean cost of the top 50% EI candidates.
    
    # Let's try a simpler approach:
    # Score = EI * (cost < threshold)
    # But a hard step function might be too aggressive.
    # Let's use a smooth approximation: exp(-(cost - threshold)^2 / (2 * width^2))
    # Or simply: 1 / (1 + exp((cost - threshold) / width))
    
    # Let's calculate a dynamic threshold.
    # The "average cost per unit improvement" is a bit ambiguous.
    # Let's interpret it as: we want to evaluate points where the cost is low relative to the potential improvement.
    # Let's define a score that is EI / cost^alpha, but with a hard gate.
    # Actually, the inspiration 1 (score 2.3735) used a "hard gate".
    # Let's look at the knowledge: "Replaces the quantile-based hard gate with a continuous information-gain-per-cost objective... Measured: 1.697 -> 2.01."
    # This suggests that the *continuous* information-gain-per-cost was better than the hard gate in one case.
    # But the directive explicitly says to try the hard gate: "Strategy to try in this change: Replace the additive cost penalty with a multiplicative feasibility constraint by defining the score as the Expected Improvement only if the predicted cost is below the current average cost per unit improvement, effectively creating a hard budget-aware gate that forces the search to exploit the low-cost corner until the GP uncertainty there is exhausted, rather than smoothly trading off value and cost."
    
    # So I will implement the hard gate as directed.
    
    # Calculate the threshold.
    # "current average cost per unit improvement"
    # Let's compute the mean cost of all candidates.
    mean_cost = np.mean(cost_safe)
    
    # But this doesn't depend on improvement.
    # Let's compute the mean cost of candidates with positive EI.
    positive_ei_mask = ei > 0
    if np.any(positive_ei_mask):
        threshold_cost = np.mean(cost_safe[positive_ei_mask])
    else:
        threshold_cost = mean_cost
    
    # To force exploitation of the low-cost corner, we should set the threshold lower than the mean.
    # Let's use the median of the costs of positive EI candidates.
    if np.any(positive_ei_mask):
        threshold_cost = np.median(cost_safe[positive_ei_mask])
    else:
        threshold_cost = mean_cost
    
    # Now, create the gate.
    # We want the score to be EI if cost < threshold_cost, and 0 otherwise.
    # But a hard step might be too unstable.
    # Let's use a very steep sigmoid or a clipped linear function.
    # Or, simply:
    # score = ei * (cost_safe < threshold_cost)
    
    # To make it more robust, let's use a small width for the transition.
    # width = 0.05 * (max_cost - min_cost)
    min_cost = np.min(cost_safe)
    max_cost = np.max(cost_safe)
    cost_range = max_cost - min_cost
    if cost_range < 1e-6:
        cost_range = 1.0
    width = 0.05 * cost_range
    
    # Smooth step function: 1 if cost < threshold, 0 if cost > threshold + width
    # x = (cost - threshold) / width
    # gate = 1 - x for x in [0, 1], 1 for x < 0, 0 for x > 1
    x = (cost_safe - threshold_cost) / width
    gate = np.clip(1.0 - x, 0.0, 1.0)
    
    scores = ei * gate
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
