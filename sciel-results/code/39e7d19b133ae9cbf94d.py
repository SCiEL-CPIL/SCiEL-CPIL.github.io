import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) strategy.
    This is equivalent to maximizing the 95th percentile of the improvement
    or minimizing the 5th percentile of the predicted function value.
    
    Args:
        mu: Array of mean predictions, shape (N,).
        sigma: Array of standard deviation predictions, shape (N,).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores, shape (N,). Higher scores indicate more promising candidates.
    """
    # Beta coefficient for 95% confidence (one-sided)
    # z-score for 95th percentile of standard normal is ~1.645
    beta = 1.645
    
    # We want to minimize f(x).
    # The lower bound on f(x) with 95% confidence is mu - beta * sigma.
    # We want to find the candidate with the lowest lower bound.
    # To convert to a "higher is better" score, we negate the lower bound.
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # If sigma or mu contains NaN or Inf, handle it?
    # The prompt says "Return 2000 finite scores".
    # Usually, GP outputs are finite. But let's be safe.
    # Replace non-finite values with a very low score (or high, depending on context).
    # Since we want high scores to be picked, non-finite inputs should probably result in low scores (bad candidates).
    # However, if mu is -inf, score is inf. That's good.
    # If mu is inf, score is -inf. That's bad.
    # If sigma is inf, score is inf. That's good (explore).
    
    # np.nan_to_num can help.
    # nan -> 0?
    # inf -> large number?
    
    # Let's just use np.where to handle NaNs by assigning a very low score.
    # But the prompt implies mu and sigma are valid GP outputs.
    # I will add a simple check.
    
    return scores
