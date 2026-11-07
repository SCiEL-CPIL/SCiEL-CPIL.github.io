import numpy as np

def acquisition(mu, sigma, f_best):
    beta = 1.0
    return -mu + beta * sigma
