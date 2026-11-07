import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # cost is an array of shape (2000,)
    # mu, sigma are arrays of shape (2000,)
    
    # Calculate total cost consumed so far? 
    # The prompt says "total cost is below 30". 
    # The 'cost' parameter in the function signature is the cost of the *candidates* being scored, 
    # not the cumulative cost. 
    # However, the directive says "beta is inversely proportional to the current budget consumption".
    # How do we know the current budget consumption inside the function?
    # The function signature is acquisition(mu, sigma, f_best, cost).
    # It does NOT pass the cumulative cost.
    # This is a problem for the directive's specific formula beta = 1 / (1 + total_cost/30).
    
    # We must infer the progress or use a fixed strategy that adapts implicitly.
    # Or, perhaps the 'cost' array itself can give a hint? No, it's just candidate costs.
    # f_best is the lowest observed value. 
    # As the optimization proceeds, f_best decreases (improves).
    # We can use f_best as a proxy for progress? 
    # Initial f_best is high (bad). Final f_best is low (good).
    # Hartmann 3D min is ~ -3.86. Initial random points might be ~ -1 to -3.
    # We can map f_best to a "progress" variable.
    
    # However, the directive explicitly mentions "current budget consumption".
    # If we don't have it, we can't use the exact formula.
    # BUT, looking at the seed program, it doesn't use cumulative cost either.
    # Maybe we can estimate the number of iterations? No, stateless function.
    
    # Let's re-read carefully: "Write acquisition(mu, sigma, f_best, cost) ... While total cost is below 30...".
    # The function is called in a loop. It doesn't receive iteration count or total cost.
    # Therefore, we cannot directly implement `beta = 1 / (1 + total_cost/30)` unless we assume something.
    # 
    # Alternative interpretation: Maybe the "cost" parameter IS the cumulative cost? 
    # "mu, sigma and cost have shape (2000,)". No, cost is per-candidate.
    #
    # So we must design an adaptive strategy that doesn't require explicit total cost, 
    # or we assume the "f_best" improvement correlates with budget usage.
    # 
    # Let's use a fixed, well-tuned beta, or a beta that depends on the spread of sigma?
    # 
    # Actually, a common trick in these stateless BO tasks is to use a fixed UCB beta that works well on average, 
    # or to use the relative improvement of f_best.
    # 
    # Let's try a fixed beta UCB with cost normalization.
    # Score = (-mu + beta * sigma) / cost
    # 
    # What is a good beta?
    # If beta is too high, we explore too much. If too low, we exploit too much.
    # For Hartmann 3D, which is multi-modal, we need enough exploration.
    # Let's try beta = 2.0.
    # 
    # To make it "adaptive" without total cost, we can look at the current best.
    # If f_best is very bad (close to 0 or positive), we are early.
    # If f_best is close to -3.86, we are late.
    # 
    # Let's define a progress metric based on f_best.
    # min_hartmann = -3.86278
    # max_hartmann ~ 0 (or higher for random points)
    # progress = 1 - (f_best - min_hartmann) / (max_hartmann - min_hartmann)
    # This is fragile if f_best isn't in range.
    #
    # Let's stick to a robust fixed CA-UCB first.
    # Score = (-mu + beta * sigma) / (1 + mean_x)
    #
    # Let's try beta = 1.5.
    
    # Calculate cost for each candidate
    # The 'cost' input is already the cost: 1 + mean(x)
    # We just use it directly.
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost-aware UCB
    # We want to maximize this score.
    # -mu favors low mu (good).
    # beta * sigma favors high uncertainty (exploration).
    # Dividing by cost favors cheap points.
    
    beta = 1.5  # Fixed exploration parameter
    
    # Calculate the UCB term
    ucb_term = -mu + beta * sigma
    
    # Calculate the cost-aware score
    # Add a small epsilon to cost to prevent division by zero, though min cost is 1.
    scores = ucb_term / (cost + 1e-9)
    
    return scores
