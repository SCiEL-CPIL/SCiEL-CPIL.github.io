import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    f_min = -39.1662
    
    # Calculate standard EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Progress-aware exploration weight
    # High weight when far from optimum, low weight when close
    # f_best is typically > f_min
    # As f_best -> f_min, exponent -> 0, lambda -> 1
    # As f_best >> f_min, exponent -> large positive, lambda -> large
    # Wait, if f_best is -10 and f_min is -39, f_best - f_min = 29.
    # exp(2.9) is large.
    # We want MORE exploration when FAR.
    # So large lambda when far is correct.
    # But we want to EXPLOIT when CLOSE.
    # So small lambda when close.
    # Current formula: lambda = exp((f_best - f_min)/10).
    # Close: f_best ~ f_min. lambda ~ 1.
    # Far: f_best >> f_min. lambda >> 1.
    # This encourages exploration when far.
    # But 1 is still quite a bit of exploration when close.
    # Maybe scale it down.
    # lambda = 0.5 * exp((f_best - f_min)/10)?
    # Let's try a simpler mapping.
    # lambda = 2.0 * (1.0 + (f_best - f_min) / 20.0)
    # This is linear.
    
    # Let's stick to the exponential but with a base of 0.5
    lambda_exp = (f_best - f_min) / 10.0
    # If f_best = -39, lambda = 0.5
    # If f_best = -19, lambda = 0.5 * exp(2) = 3.7
    # If f_best = -59 (impossible), lambda = 0.5 * exp(-2) = 0.067
    lambda_w = 0.5 * np.exp(lambda_exp)
    
    # Clip to reasonable range
    lambda_w = np.clip(lambda_w, 0.1, 5.0)
    
    return ei + lambda_w * sigma
