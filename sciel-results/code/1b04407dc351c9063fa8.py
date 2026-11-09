import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Check time limit
    # The evaluator kills at 330s.
    # We are just returning an array, so it's fast.
    
    # Thompson Sampling Strategy
    # 1. Sample a function value from the posterior distribution N(mu, sigma^2)
    # 2. Select the candidate with the minimum sampled value
    # 3. To make it stochastic and robust, we return scores that reflect this preference
    #    but add a small amount of noise to prevent cycling or getting stuck in local optima
    #    if the evaluator uses a "max score" selection.
    
    # Since we need to return 2000 scores where higher is better,
    # and we want to pick the minimum of the sampled function:
    
    # Generate a random sample from the posterior for each candidate
    # To ensure reproducibility within the same call if needed, we use a fixed seed 
    # based on the current time or a global counter, but here we just use standard randn.
    # Actually, to make it a proper acquisition function that returns a *score* for 
    # each candidate, we can interpret the score as the negative of the sampled value.
    # However, standard TS usually just picks argmin(sampled). 
    # If the evaluator picks argmax(scores), we should return -sampled_values.
    
    # But wait, if we return -sampled_values, the "score" changes every time the function 
    # is called for the same mu/sigma (if it were called twice, which it isn't in the loop 
    # per candidate, but the loop calls acquisition once per step with 2000 candidates).
    # The evaluator draws 2000 candidates, calls acquisition once, gets 2000 scores, 
    # and picks the max score.
    # So, if we return -sampled_values, the candidate with the lowest sampled value 
    # (most likely to be the minimum) will have the highest score (least negative).
    
    # Let's add a small exploration noise to the scores to avoid deterministic behavior 
    # if there are ties or if the GP is overconfident.
    
    # Sample from N(mu, sigma)
    # sigma is the standard deviation
    sampled_values = mu + sigma * np.random.randn(len(mu))
    
    # We want to minimize the function, so we prefer lower sampled_values.
    # The evaluator maximizes the score.
    # So score = -sampled_values
    scores = -sampleed_values
    
    # To ensure we don't get stuck in a local optimum due to the randomness of the sample
    # being too "greedy" or too "noisy", we can blend with a small LCB component 
    # or just rely on the TS property.
    # Pure TS is often superior to LCB in practice for multi-modal functions like Branin.
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
