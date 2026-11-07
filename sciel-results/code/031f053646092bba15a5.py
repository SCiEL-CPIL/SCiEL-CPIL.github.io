import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling-based cost-aware acquisition function.
    
    For each candidate, sample a latent function value from the GP predictive
    distribution (mu + sigma * z) and score as the ratio of expected improvement
    (relative to f_best) to cost. This stochastically balances exploration and
    exploitation.
    
    Args:
        mu: array of shape (2000,), mean predictions.
        sigma: array of shape (2000,), standard deviation predictions.
        f_best: float, best observed value.
        cost: array of shape (2000,), cost of evaluating each point.
        
    Returns:
        array of shape (2000,), scores (higher is better).
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: draw a single sample from the predictive distribution
    # For minimization, we want to find points where the sampled value is low
    # relative to f_best. We sample y ~ N(mu, sigma^2).
    # The "improvement" for a sampled value y is max(f_best - y, 0).
    # But since we only get one sample, we use the sampled value itself as a proxy.
    # A better approach: score based on how much the sampled value beats f_best.
    # If sampled_value < f_best, it's a good point.
    # We can use: score = max(f_best - sampled_value, 0) / cost
    # But this is 0 for most points where sampled_value > f_best, which is too conservative.
    
    # Alternative: Use the sampled value directly as a "regret" proxy.
    # Lower sampled value = better. So score = -sampled_value / cost? No, that penalizes high cost.
    # Actually, we want to maximize the chance of finding a better point.
    # Let's use: score = (f_best - sampled_value) / cost, but clip to be non-negative?
    # Or just use the negative of the sampled value normalized by cost?
    
    # Let's think about this differently. In Thompson Sampling for minimization:
    # We sample f(x) ~ N(mu(x), sigma(x)^2)
    # We pick x that minimizes the sampled f(x).
    # But we have a cost constraint. So we want to pick x that minimizes f(x)/cost(x)?
    # Or minimizes f(x) subject to cost budget?
    
    # The task says: "selecting the candidate that maximizes the ratio of this sampled value 
    # to its evaluation cost" - wait, that would maximize sampled_value/cost, which is wrong 
    # for minimization. We want to MINIMIZE the sampled value, or MAXIMIZE the improvement.
    
    # Let's re-read the directive: "maximizes the ratio of this sampled value to its evaluation cost"
    # This seems backwards for minimization. Maybe it means we should minimize sampled_value * cost?
    # Or perhaps the intent is to use a cost-aware version where we prefer points with low sampled 
    # value AND low cost.
    
    # A common approach: score = -sampled_value / cost (higher is better, so lower sampled value 
    # and lower cost give higher score). But this can be problematic if sampled_value is very negative.
    
    # Better: Use Expected Improvement from the sample. Since we have one sample y ~ N(mu, sigma^2),
    # the improvement is max(f_best - y, 0). But this is 0 with high probability if sigma is small 
    # and mu is close to f_best.
    
    # Let's use a hybrid: score = (f_best - mu) / cost + exploration_bonus
    # Where exploration_bonus encourages points with high sigma.
    
    # Actually, let's stick with Thompson Sampling properly:
    # Sample y = mu + sigma * z, where z ~ N(0, 1)
    # Then score = max(f_best - y, 0) / cost
    # This gives 0 for most points, which might be too conservative.
    
    # Alternative: score = (f_best - y) / cost, allowing negative values.
    # Points with y > f_best get negative scores, points with y < f_best get positive scores.
    # This naturally balances exploitation (low y) and exploration (high sigma leads to more variance in y).
    
    # Let's try: score = (f_best - (mu + sigma * z)) / cost
    # where z is a standard normal random variable.
    
    # Draw a single standard normal random variable for all candidates
    z = np.random.standard_normal(1)
    
    # Sampled values
    sampled_values = mu + sigma_safe * z
    
    # Score: higher is better. We want low sampled_values and low cost.
    # score = (f_best - sampled_values) / cost
    # This can be negative, which is fine for ranking.
    
    score = (f_best - sampled_values) / cost_safe
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    return score
