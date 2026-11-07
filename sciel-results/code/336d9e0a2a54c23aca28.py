import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive exploration
_state = {
    'call_count': 0,
    'start_time': None,
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and Thompson Sampling (TS).
    
    This hybrid approach uses EI for a stable exploitation baseline and adds a stochastic
    exploration term that adapts over time. As the optimization progresses (more calls),
    the exploration weight decreases, shifting focus from exploration to exploitation.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
        _state['call_count'] = 0
    
    # Deadline guard: if we are close to the 300s limit, return a safe, deterministic result
    # to ensure we don't timeout. The task says each evaluation must finish within 300s.
    # The evaluator kills after 330s. We have 25 evaluations (30 total - 5 initial).
    # If we are near the end, just return EI to be safe and fast.
    elapsed = time.time() - _state['start_time']
    # If elapsed time is high, we might be in danger. 
    # However, the acquisition function is called 25 times total.
    # The heavy lifting is in the GP posterior computation which happens *before* this call.
    # This function is very fast. So deadline guard is less critical here unless we add heavy logic.
    # We will keep it simple and fast.
    
    _state['call_count'] += 1
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI) component
    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Thompson Sampling (TS) component
    # Sample from the posterior
    # We scale the sample to be comparable to EI? Or just add it?
    # EI is in objective units. The sample is in objective units.
    # Adding them directly is valid if we want a mix.
    # However, EI is non-negative. The sample can be anything.
    # A common hybrid is: Score = EI + beta * (f_best - sample)
    # Or simply: Score = EI + beta * sample_noise
    # Let's try: Score = EI + beta * (f_best - sample)
    # If sample < f_best, this term is positive, boosting exploration.
    # If sample > f_best, this term is negative, penalizing it.
    
    sample = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Adaptive beta: Start with high exploration, decay to low exploration
    # Total evaluations ~ 30. We are at call_count (1-based for acquisition calls, so 1..25)
    # Let's decay beta from 1.0 to 0.1
    # beta = 1.0 / (1.0 + 0.05 * _state['call_count'])
    # Or simpler linear decay:
    # progress = _state['call_count'] / 25.0
    # beta = 1.0 - 0.8 * progress
    
    # Let's use an exponential decay for smoother transition
    # beta = 0.5 * np.exp(-0.1 * _state['call_count'])
    
    # Based on the insight "Adaptive UCB shifts from exploration to exploitation",
    # let's implement a similar adaptive weight for the exploration term.
    # The EI term already has some exploration (via sigma), but it's bounded.
    # The TS term is unbounded.
    
    # Let's try a weighted sum:
    # score = w_ei * EI + w_ts * (f_best - sample)
    # As we progress, w_ts should decrease.
    
    # Let's define a progress factor
    # 5 initial points + 25 acquisition calls = 30 points.
    # We don't know exactly how many calls we've had, but _state['call_count'] tracks it.
    
    # Let's assume total acquisition calls is 25.
    total_calls = 25
    progress = min(_state['call_count'] / total_calls, 1.0)
    
    # Exploration weight: starts at 1.0, ends at 0.1
    # Using a smooth curve
    w_ts = 0.1 + 0.9 * (1.0 - progress)
    
    # Exploitation weight: constant 1.0 for EI
    w_ei = 1.0
    
    # The term (f_best - sample) has the same scale as EI roughly.
    # If sample is very low, f_best - sample is large positive.
    # If sample is very high, f_best - sample is large negative.
    
    ts_term = f_best - sample
    
    # Combine
    scores = w_ei * ei + w_ts * ts_term
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
