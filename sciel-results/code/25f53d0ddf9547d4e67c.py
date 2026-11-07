"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling score: sample from the posterior predictive distribution.
    # The evaluator will select the candidate with the highest score.
    # Since we are minimizing, a lower sampled value is better.
    # However, the task states "higher meaning evaluate next".
    # So we must return scores where higher is better.
    # If we sample from N(mu, sigma), a low sample is a good candidate.
    # To make "higher score" correspond to "better candidate (lower f)", 
    # we can return negative of the sample, or simply the sample if the evaluator 
    # interprets "highest score" as "best candidate". 
    # Let's re-read: "calls acquisition and evaluates the candidate with the highest score".
    # "Return 2000 finite scores, higher meaning evaluate next".
    # This implies the evaluator picks argmax(acquisition_scores).
    # If we want to pick the candidate with the lowest predicted f, we should 
    # return scores that are high for low f.
    # A common trick in TS for minimization when the selection is argmax is to 
    # return -sampled_value.
    # Let's draw a sample from N(mu, sigma).
    # sample = mu + sigma * noise
    # score = -sample
    # This way, the candidate with the lowest sampled value gets the highest score.
    
    # We need to be careful about the randomness. 
    # The evaluator calls acquisition once per step. 
    # If we use random numbers inside, the score is stochastic.
    # This is the essence of Thompson Sampling.
    
    # Use a fixed seed? No, TS requires fresh samples each step to explore.
    # However, if the evaluator calls acquisition multiple times or expects determinism, 
    # this might be an issue. But the task says "calls acquisition... evaluates the candidate with the highest score".
    # It's a single call per step.
    
    # To ensure stability and avoid issues with random state persistence if the function is called in a weird way,
    # we can just generate random numbers directly.
    
    # One potential issue: if sigma is very small, the sample is close to mu.
    # If sigma is large, the sample is more exploratory.
    
    # Let's implement:
    # 1. Generate standard normal noise of shape (2000,)
    # 2. Compute sample = mu + sigma * noise
    # 3. Return -sample (so lower f -> higher score)
    
    # However, there is a subtle point. Standard TS draws a sample for each candidate? 
    # No, typically you draw one sample from the posterior for each candidate point to estimate f(x).
    # Yes, that's what `mu + sigma * noise` does for each of the 2000 candidates.
    
    # Is there a risk of non-finite values?
    # mu and sigma are from a GP, so they should be finite.
    # noise is finite.
    # So scores should be finite.
    
    # Let's add a small safeguard for sigma being negative or zero, though GP sigma should be positive.
    # sigma = np.maximum(sigma, 1e-9)
    
    # The prompt says "sigma (a standard deviation) are in objective units".
    
    # Let's use a simple random draw.
    noise = np.random.randn(len(mu))
    samples = mu + sigma * noise
    
    # We want to maximize the score to pick the best candidate.
    # Best candidate has the lowest f.
    # So score = -samples.
    
    return -samples
# EVOLVE-BLOCK-END
