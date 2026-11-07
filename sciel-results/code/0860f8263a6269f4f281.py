import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # We want to maximize the potential for finding a low value.
    # The lower bound of the prediction is mu - beta * sigma.
    # We want the lowest lower bound?
    # No, in BO, we maximize an acquisition function.
    # If we define acquisition as the negative of the lower bound:
    # A(x) = -(mu(x) - beta * sigma(x)) = -mu(x) + beta * sigma(x)
    # Maximizing A(x) is equivalent to minimizing the lower bound.
    # This is a common strategy.
    
    # Alternatively, some define UCB as mu + beta * sigma for maximization problems.
    # For minimization, it is mu - beta * sigma (minimize this).
    # Since the interface requires "higher score = evaluate next", we return the negative of the quantity we want to minimize.
    # Quantity to minimize: mu(x) - beta * sigma(x) ?
    # Wait. If I minimize mu - beta*sigma, I am looking for points with low mean and high variance.
    # Yes.
    # So Score = -(mu - beta * sigma) = -mu + beta * sigma.
    
    beta = 2.0
    return -mu + beta * sigma
