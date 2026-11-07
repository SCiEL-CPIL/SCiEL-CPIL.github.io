import numpy as np
import time

# Global state
_acq_call_count = 0

def acquisition(mu, sigma, f_best):
    global _acq_call_count
    _acq_call_count += 1
    
    # t is the number of evaluations so far (including initial 5)
    # The call count is the number of acquisition calls.
    # Total evaluations = 30.
    # Initial 5 are uniform.
    # So the first acquisition call corresponds to the 6th evaluation.
    t = _acq_call_count + 5
    
    # Adaptive beta for LCB
    # We want high exploration early, low later.
    # beta = c / sqrt(t) or similar.
    # Let's try beta = 2.0 / (1 + 0.1 * t)
    # t=6 -> 2/1.6 = 1.25
    # t=30 -> 2/4 = 0.5
    
    beta = 2.0 / (1.0 + 0.1 * t)
    
    # Score = beta * sigma - mu
    # Higher score means more likely to be a good minimum (low mu, high sigma)
    scores = beta * sigma - mu
    
    return scores
