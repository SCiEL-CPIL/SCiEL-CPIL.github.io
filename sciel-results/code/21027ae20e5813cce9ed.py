import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
       # Pure Expected Improvement
       # z = (f_best - mu) / sigma
       z = (f_best - mu) / (sigma + 1e-9)
       # Avoid overflow in cdf/pdf for large z
       # norm.cdf and norm.pdf are stable in scipy
       ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
       
       # Handle NaN/Inf
       ei = np.where(np.isfinite(ei), ei, 0.0)
       
       return ei
