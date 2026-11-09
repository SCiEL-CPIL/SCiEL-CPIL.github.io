import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) acquisition function.
    
    Strategy:
    Instead of PI (which is 0 when mu > f_best and ignores sigma in the ranking 
    if mu is fixed), we use LCB: mu - kappa * sigma.
    This encourages exploration of high-uncertainty regions (high sigma) which 
    is vital for finding the shifted minimizer in 8D where the GP might be 
    uncertain in large portions of the space.
    
    We subtract a cost penalty to ensure we prefer cheaper evaluations when the 
    expected improvement/exploration value is similar.
    
    Parameters:
    mu: array of shape (2000,) - GP mean
    sigma: array of shape (2000,) - GP standard deviation
    f_best: float - best observed value
    cost: array of shape (2000,) - cost of evaluation
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle potential numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # LCB: We want to minimize mu - kappa * sigma.
    # So we maximize -(mu - kappa * sigma) = -mu + kappa * sigma.
    # However, the cost-aware part suggests we should penalize high cost.
    # A common cost-aware LCB is: (mu - kappa * sigma) / cost or mu - kappa * sigma - lambda * cost.
    # Given the previous success of dividing by cost (which acts as a strong penalty for expensive points),
    # and the finding that "exponential cost gating" or similar helped, let's stick to a multiplicative 
    # or divisional structure but applied to the LCB term.
    
    # Let's try: score = (kappa * sigma - mu) / cost
    # This is equivalent to minimizing (mu - kappa * sigma) * cost.
    # Wait, if cost is in [1, 2], dividing by cost penalizes expensive points.
    # If we use (kappa * sigma - mu) / cost, we are maximizing the "optimistic" value per unit cost.
    
    # Let's tune kappa. In high dimensions, we need significant exploration.
    # A standard choice for LCB is around 1.0-2.0.
    # Since the function is cosine8, the landscape is complex.
    # Let's use a kappa that balances exploration and exploitation.
    kappa = 1.5
    
    # Calculate the LCB score
    # We want to maximize this value.
    # lcb_value = mu - kappa * sigma (this is the lower bound, we want to minimize it)
    # So acquisition score = -lcb_value = -mu + kappa * sigma
    
    # Apply cost awareness.
    # Dividing by cost is a strong penalty. 
    # score = (kappa * sigma - mu) / cost
    
    # To avoid division by zero or very small costs (min cost is 1), this is safe.
    scores = (kappa * sigma - mu) / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
