import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Thompson Sampling: draws a single sample from the posterior 
    distribution at the candidate points and uses that sample as the score.
    
    For minimization, we want to find low values. 
    The posterior is N(mu, sigma^2).
    We draw y ~ N(mu, sigma).
    We return y as the score? 
    Wait, the evaluator picks the candidate with the HIGHEST score.
    If we want to minimize f(x), and we believe f(x) is likely low, 
    we want the score to be high.
    
    If we draw y ~ N(mu, sigma), a low y indicates a likely low function value.
    But we need to maximize the score.
    So we should return -y?
    Or simply return -mu + z*sigma?
    
    Actually, standard Thompson Sampling for minimization:
    1. Draw f_tilde ~ GP posterior.
    2. Select x_next = argmin_x f_tilde(x).
    
    Since the evaluator selects argmax_score, we need:
    score(x) = - f_tilde(x)
    f_tilde(x) = mu(x) + z * sigma(x) where z ~ N(0, 1) is a single scalar?
    
    Wait, in Thompson Sampling for a GP, the function values at different points are correlated.
    However, we are only given pointwise mu and sigma. We do not have the covariance matrix 
    between the 2000 candidate points. 
    Therefore, we cannot draw a *jointly* consistent function sample.
    We can only draw independent samples for each point:
    y_i = mu_i + z_i * sigma_i, with z_i ~ N(0, 1) independent for each i.
    
    Is this valid? 
    Yes, this is often called "independent Thompson Sampling" or "local Thompson Sampling".
    It loses the correlation structure but is a valid acquisition function that balances 
    exploration (high sigma increases variance of y_i) and exploitation (low mu decreases mean of y_i).
    
    For minimization:
    We want to select the point with the lowest likely function value.
    If we return score = -y_i, then the highest score corresponds to the lowest y_i.
    
    Let's verify:
    If mu is low and sigma is low, y is low, -y is high. Good.
    If mu is high but sigma is very high, y could be low (lucky draw) or high. 
    This introduces stochastic exploration.
    
    This is a fundamentally different mechanism from LCB (which is deterministic given the model) 
    and EI (which is deterministic given the model).
    """
    start_time = time.time()
    deadline = start_time + 290.0

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw a single standard normal random number z
    # Note: Using a single z for all points would imply perfect correlation.
    # Using independent z_i for each point is the standard approximation when 
    # only pointwise posterior parameters are available.
    # However, a single global z is sometimes used to preserve the "global" nature 
    # of the function draw better in simple heuristics, but independent is more 
    # standard for "pointwise" TS. Let's use independent samples.
    
    # Generate random samples
    # np.random.randn(N)
    z = np.random.randn(len(mu))
    
    # Compute the sampled function values
    # y = mu + z * sigma
    y = mu + z * sigma_safe
    
    # We want to MINIMIZE y.
    # The evaluator MAXIMIZES the returned score.
    # So we return -y.
    scores = -y
    
    # Safety check for time
    if time.time() > deadline:
        # This should be extremely fast, but just in case
        pass

    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
