import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    mu (np.ndarray): Mean predictions from the GP, shape (2000,).
    sigma (np.ndarray): Standard deviation predictions from the GP, shape (2000,).
    f_best (float): The lowest observed function value.
    cost (np.ndarray): Cost of evaluating each candidate, shape (2000,).
    
    Returns:
    np.ndarray: Scores for the candidates, shape (2000,). Higher is better.
    """
    # Calculate the standard deviation of the predictions to adapt the cost penalty
    # If the predictions are spread out, we can tolerate a higher cost penalty
    # If they are tight, we need to be very sensitive to cost
    mu_std = np.std(mu)
    
    # Avoid division by zero or extremely small values
    if mu_std < 1e-9:
        mu_std = 1.0
        
    # The cost term should be small relative to the function values
    # We use a fraction of the standard deviation as the scale
    # 0.05 is a heuristic: 5% of the prediction uncertainty
    lambda_cost = 0.05 * mu_std
    
    # Sample from the posterior distribution N(mu, sigma^2)
    # Note: sigma is the standard deviation, so we pass it directly
    f_samples = np.random.normal(mu, sigma)
    
    # We want to minimize f(x) + lambda * cost(x)
    # So we compute the negative of this for the acquisition score
    # (since the evaluator picks the argmax)
    adjusted_scores = -(f_samples + lambda_cost * cost)
    
    # Ensure all scores are finite
    # If any samples are inf or nan, replace them with -inf
    adjusted_scores = np.where(np.isfinite(adjusted_scores), adjusted_scores, -np.inf)
    
    return adjusted_scores
