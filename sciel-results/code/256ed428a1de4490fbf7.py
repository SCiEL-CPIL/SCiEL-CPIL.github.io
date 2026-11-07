import numpy as np

def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-9)
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    from scipy.stats import norm
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return scores
