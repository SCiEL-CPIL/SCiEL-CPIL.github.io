import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB for minimization
    # beta scales with the mean uncertainty to balance exploration/exploitation
    beta = np.mean(sigma)
    return -mu + beta * sigma
