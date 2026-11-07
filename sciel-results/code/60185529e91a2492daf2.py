import numpy as np

def acquisition(mu, sigma, f_best):
    beta = 2.0
    scores = -mu + beta * sigma
    return scores
