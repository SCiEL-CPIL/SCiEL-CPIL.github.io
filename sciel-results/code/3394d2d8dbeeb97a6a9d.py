import numpy as np
import time

# Global state to track time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are getting close to the 330s limit, 
    # return a safe, deterministic result to avoid timeout penalties.
    # The task says "Each evaluation must finish within 300s" and 
    # "evaluator KILLS the program after 330s". 
    # Since this function is called inside a loop, if the total time 
    # is running out, we should just pick the point with lowest mu 
    # to be safe and fast.
    if time.time() - _start_time > 325.0:
        # Deterministic fallback: pick point with lowest mean, adjusted by cost
        # Just return -mu/cost to prioritize high value (low mu) and low cost
        return -mu / np.maximum(cost, 1e-9)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling for minimization:
    # We sample from the posterior p(f(x) | D).
    # Since the GP posterior is Gaussian with mean mu and std sigma,
    # we can draw a sample y ~ N(mu, sigma^2).
    # To minimize f, we want to maximize the probability that the sample is low.
    # Equivalently, we can just sample the objective values and pick the one
    # that yields the lowest sampled value, adjusted for cost.
    
    # However, a direct sample y = mu + sigma * z (z ~ N(0,1)) is a valid 
    # instance from the posterior. 
    # The "acquisition" score should reflect the desirability of evaluating x.
    # In TS, we usually just pick the argmin of the samples. 
    # But here we return scores for 2000 candidates, and the caller picks the max score.
    # So, Score(x) should be high if the sampled value y(x) is low (good for minimization).
    # Let's define Score(x) = -y(x) / cost(x).
    # This favors points where the posterior sample predicts a low function value, 
    # penalized by the cost of evaluation.
    
    # To ensure stability and avoid extreme outliers from the Gaussian tail 
    # dominating the selection in a way that ignores the mean structure too much,
    # some implementations use a Laplace perturbation or a specific scaling.
    # Standard TS uses Gaussian samples. Let's stick to Gaussian.
    
    # Generate random noise
    # Using a single random seed per call is not possible/needed, 
    # but we must ensure the random numbers are different each call to explore.
    # np.random is the standard way.
    
    z = np.random.randn(len(mu))
    samples = mu + sigma_safe * z
    
    # The score for minimization: we want high scores for low samples.
    # Score = -sample / cost
    # This is "Cost-Aware Thompson Sampling".
    
    scores = -samples / np.maximum(cost, 1e-9)
    
    # Handle any potential numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
