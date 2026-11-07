# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using
    multi-fidelity surrogate-guided ranking with Bayesian Regret Minimization.
    
    The key idea: instead of simple pointwise EI, we approximate the
    probability that each candidate is the global minimum within the
    remaining cost budget. This is done by:
    1. Computing a cost-adjusted "quality" score for each candidate
    2. Estimating the probability of being the argmin under the GP posterior
       conditioned on the cost constraint
    3. Weighting by the expected reduction in Bayesian regret
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x) scaled to [0,1]).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Step 1: Compute Expected Improvement (baseline signal)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Step 2: Estimate the probability that candidate i is the global minimum
    # within the remaining budget. We use a two-part approximation:
    #
    # (a) "Optimistic value" score: how good is this candidate likely to be?
    #     Use a cost-aware lower confidence bound (LCB):
    #     lcb_i = mu_i - kappa * sigma_i, where kappa balances exploration
    #
    # (b) "Cost efficiency" weighting: cheaper candidates get higher weight
    #     because we can evaluate more of them within the budget.
    #
    # The BRM objective approximates:
    #   P(i is argmin | budget) ∝ exp(-cost_i / B) * exp(-(mu_i - kappa*sigma_i)^2 / (2*sigma_i^2))
    # where B is the remaining budget. Since we don't have B directly, we use
    # the average cost as a proxy for the budget scale.
    
    # Compute a cost-aware LCB with adaptive kappa
    # kappa should be larger when we have less budget (more exploration needed)
    # Use the mean cost as a proxy: lower mean cost means more evaluations possible
    mean_cost = np.mean(cost)
    # Normalize kappa: when mean_cost is low (cheap evaluations), explore more
    kappa = 1.0 + 0.5 / (mean_cost + 1e-6)
    
    lcb = mu - kappa * sigma
    
    # Step 3: Compute the probability of being the global minimum
    # Use a softmax over the negative LCB values, scaled by cost efficiency
    # The idea: candidates with lower LCB (more promising) and lower cost
    # have higher probability of being the best within budget
    
    # Cost efficiency: 1/cost normalized so that mean efficiency is 1
    cost_efficiency = 1.0 / cost
    cost_efficiency = cost_efficiency / (np.mean(cost_efficiency) + 1e-12)
    
    # Combine LCB-based quality with cost efficiency
    # Use a temperature-scaled softmax for stability
    # Temperature based on the spread of LCB values
    lcb_range = lcb.max() - lcb.min()
    if lcb_range < 1e-6:
        temp = 1.0
    else:
        temp = lcb_range / 4.0
    
    # Quality score: lower LCB is better, so negate
    quality = -lcb / temp
    
    # Apply cost efficiency as a multiplicative weight in log-space
    # log_prob_i ∝ quality_i + log(cost_efficiency_i)
    log_weights = quality + np.log(cost_efficiency + 1e-12)
    
    # Numerical stability: subtract max
    log_weights_shifted = log_weights - log_weights.max()
    weights = np.exp(log_weights_shifted)
    weights = weights / (weights.sum() + 1e-12)
    
    # Step 4: Compute the expected reduction in Bayesian regret
    # The BRM score for candidate i is:
    #   BRM_i = P(i is argmin) * E[improvement_i | i is argmin]
    #
    # Approximate E[improvement_i | i is argmin] using a modified EI that
    # accounts for the fact that we're only evaluating if this is likely the best.
    # Use a "conditional EI" that is higher when the candidate is more likely
    # to be the global minimum.
    
    # Conditional EI: weight EI by the probability of being the best
    # and add a bonus for high-uncertainty candidates that are cheap (exploration value)
    exploration_bonus = sigma * (1.0 / cost)
    
    # Combine: BRM score = weights * (ei + exploration_bonus)
    scores = weights * (ei + 0.3 * exploration_bonus)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
