import numpy as np

def acquisition(mu, sigma, f_best):
        epsilon = 1e-9
        scores = (f_best - mu) / (sigma + epsilon)
        return scores
