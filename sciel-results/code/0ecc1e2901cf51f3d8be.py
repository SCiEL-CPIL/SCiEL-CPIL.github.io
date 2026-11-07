"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    with an adaptive exploration parameter that shifts from exploration to exploitation
    as the budget is consumed.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate UCB
    # Adaptive beta: start with more exploration, reduce as budget is consumed
    # cost is the cost of each candidate, but we need a measure of budget consumption
    # Since we don't have total_cost directly, we use the mean cost as a proxy
    # for how "expensive" the current batch is, and scale beta accordingly
    
    # A simpler approach: use a fixed beta that balances exploration/exploitation
    # For minimization, UCB = mu - beta * sigma (we want to minimize)
    # But we want higher scores to be better, so we negate or adjust
    
    # Actually, let's think about this differently.
    # The standard approach for cost-aware BO is to maximize EI / cost.
    # But we can also blend with UCB.
    
    # Let's use a weighted combination:
    # score = w * (EI / cost) + (1-w) * (UCB_normalized)
    
    # For UCB in minimization: we want to find low mu with high sigma
    # UCB_min = mu - beta * sigma (lower is better for minimization)
    # To convert to "higher is better" score: score_ucb = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Adaptive beta based on the fraction of budget used.
    # Since we don't know total_cost, we can estimate it from the cost array.
    # The cost array represents the cost of each of the 2000 candidates.
    # A reasonable proxy for budget consumption is the mean cost relative to the max possible.
    
    # Let's use a simpler adaptive scheme:
    # beta = 2.0 / (1 + mean_cost)
    # This gives beta ~ 2 when mean_cost is small (early), and beta ~ 0 when mean_cost is large.
    
    mean_cost = np.mean(cost)
    beta = 2.0 / (1.0 + mean_cost)
    
    # UCB score (higher is better for our purposes)
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is already in objective units, UCB is also in objective units
    
    # Weight between EI and UCB: start with more EI, shift to UCB? Or vice versa?
    # Actually, EI already incorporates some exploration through sigma.
    # Let's just use EI/cost with a small UCB component for robustness.
    
    # Cost-aware EI
    ei_cost = ei / (cost + 1e-9)
    
    # Add a small UCB component to encourage exploration in uncertain regions
    # Scale UCB to be comparable to EI/cost
    # The scale of EI/cost depends on the problem, so let's use a fixed small weight
    
    # Actually, let's keep it simple and just use EI/cost, which is the standard approach.
    # But add a small exploration bonus based on sigma/cost to break ties and encourage
    # exploring uncertain regions when EI is similar.
    
    exploration_bonus = 0.1 * sigma_safe / (cost + 1e-9)
    
    scores = ei_cost + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
