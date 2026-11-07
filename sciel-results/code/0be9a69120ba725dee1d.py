import numpy as np

def acquisition(mu, sigma, f_best):
    # Estimate the current level of uncertainty to determine exploration rate.
    # Since targets are standardized, the prior standard deviation is approx 1.0.
    # As points are evaluated, the posterior standard deviation (sigma) decreases.
    med_sigma = np.median(sigma)
    
    # Clip med_sigma to [0, 1] to handle any numerical anomalies or prior variations
    med_sigma = np.clip(med_sigma, 0.0, 1.0)
    
    # Map the uncertainty to a beta coefficient for the Lower Confidence Bound (LCB).
    # High uncertainty (med_sigma ~ 1.0) -> High beta (5.0) -> More Exploration
    # Low uncertainty (med_sigma ~ 0.0) -> Low beta (0.5) -> More Exploitation
    beta = 0.5 + 4.5 * med_sigma
    
    # LCB for minimization:
    # We want to minimize the true value f(x).
    # The lower confidence bound is mu(x) - beta * sigma(x).
    # We select the point with the lowest lower bound.
    # The acquisition function must return scores where HIGHER is better.
    # Therefore, we return the negative of the lower bound:
    # score = -(mu - beta * sigma) = beta * sigma - mu
    scores = beta * sigma - mu
    
    # Ensure all scores are finite to avoid invalid results
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
