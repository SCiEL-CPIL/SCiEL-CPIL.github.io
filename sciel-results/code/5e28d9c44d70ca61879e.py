import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB (Upper Confidence Bound)
    # Distinct from EI: Uses a confidence bound for exploration/exploitation trade-off.
    
    # Exploration parameter
    beta = 10.0
    
    # Safety clamps
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # For minimization, we want to maximize the lower bound of the predictive distribution?
    # No, UCB for minimization maximizes the "potential" for being low.
    # The interval is [mu - beta*sigma, mu + beta*sigma].
    # We hope the true value is near the lower end?
    # Actually, standard UCB for minimization minimizes: mu(x) - beta * sigma(x).
    # Therefore, the score to maximize is: -(mu(x) - beta * sigma(x)) = beta * sigma(x) - mu(x).
    
    score = (beta * sigma - mu) / cost
    
    return score
