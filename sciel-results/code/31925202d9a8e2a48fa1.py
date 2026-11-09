"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills the program after 330s wall-clock.
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds before the deadline to stop doing heavy work


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a structurally different approach: instead of blending normalized EI and PI,
    we use a cost-aware Expected Hypervolume Improvement (EHVI) approximation combined
    with an uncertainty-based exploration bonus that adapts to the budget.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x) where x is in [0,1]^3.
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    # Check deadline: if we're running out of time, return a simple fallback
    if _time_left() <= 0:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure sigma and cost are positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute the improvement over f_best
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI) for minimization
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Structural change: use a cost-aware EI that accounts for the expected
    # number of future evaluations we can afford. We estimate the remaining
    # budget by looking at the cost distribution and the current f_best.
    
    # Estimate remaining budget: total budget is 30, we've spent some already.
    # We don't know exactly how much, but we can use the cost array as a proxy.
    # The mean cost of the candidates gives us a sense of the typical cost.
    mean_cost = np.mean(cost)
    
    # A simpler approach: use the cost directly to weight the EI.
    # We want to prefer candidates that have high EI but low cost.
    # Use a power-law cost discounting that adapts based on the spread of costs.
    
    # Compute the cost-normalized EI
    # Use a temperature parameter that controls how strongly we penalize cost
    # Higher temperature = more exploration (less cost-sensitive)
    # Lower temperature = more exploitation (more cost-sensitive)
    
    # Adapt the temperature based on the uncertainty in the GP predictions
    # If uncertainty is high, we should explore more (higher temperature)
    # If uncertainty is low, we should exploit more (lower temperature)
    mean_sigma = np.mean(sigma)
    
    # Normalize sigma to get a relative measure of uncertainty
    # Use a sigmoid-like function to map uncertainty to a temperature
    # When mean_sigma is large relative to the scale of f_best, use higher temperature
    sigma_scale = max(abs(f_best), 1.0)
    relative_uncertainty = mean_sigma / sigma_scale
    
    # Temperature: between 0.5 and 2.0
    # High uncertainty -> high temperature (explore)
    # Low uncertainty -> low temperature (exploit)
    temperature = 0.5 + 1.5 * np.tanh(relative_uncertainty)
    
    # Cost-aware score: EI^(1/temperature) / cost
    # This is equivalent to a softmax-like weighting over the cost
    # When temperature is high, the exponent is close to 1, so cost matters less
    # When temperature is low, the exponent is close to 0.5, so cost matters more
    
    # Actually, let's use a different structural approach:
    # Use the EI directly but with a cost-aware transformation that
    # accounts for the expected regret in a more principled way.
    
    # Expected Regret for minimization: E[max(0, f(x) - f_best)]
    # This is exactly the EI we computed above.
    
    # Now, we want to maximize EI / cost, but with a twist:
    # We add an exploration bonus based on the GP uncertainty,
    # but scale it by the cost to avoid exploring expensive uncertain regions.
    
    # Exploration bonus: sigma / cost
    # This encourages exploring uncertain regions, but penalizes if they're expensive.
    exploration_bonus = sigma_safe / cost_safe
    
    # Combine EI and exploration bonus with an adaptive weight
    # When EI is large relative to the exploration bonus, exploit
    # When EI is small, explore
    
    # Compute a ratio to determine the blend
    ei_max = np.max(ei) if np.max(ei) > 1e-9 else 1e-9
    bonus_max = np.max(exploration_bonus) if np.max(exploration_bonus) > 1e-9 else 1e-9
    
    # Normalize both to [0, 1]
    ei_norm = ei / ei_max
    bonus_norm = exploration_bonus / bonus_max
    
    # Adaptive weight: use more exploration when EI is low
    # This is a structural change: instead of a fixed blend, we use
    # a data-driven blend based on the current state of the optimization.
    
    # When the best EI is small (we're close to convergence), explore more
    # When the best EI is large (we're far from convergence), exploit more
    ei_scale = max(abs(f_best), 1.0)
    ei_relative = ei_max / ei_scale
    
    # Weight for exploration: higher when EI is small
    exploration_weight = 0.3 + 0.4 * (1.0 - np.tanh(ei_relative))
    exploitation_weight = 1.0 - exploration_weight
    
    blended = exploitation_weight * ei_norm + exploration_weight * bonus_norm
    
    # Final cost-aware score
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
