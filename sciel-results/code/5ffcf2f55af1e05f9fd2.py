"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    # The evaluator kills after 330s. We assume this function is called many times.
    # However, the prompt says "Each evaluation must finish within 300s".
    # This likely refers to the total time of the search loop or the specific call?
    # "The evaluator measures... The search score is...".
    # "Each evaluation must finish within 300 s." usually means the whole run.
    # But we only control this function.
    # If we assume this function is called ~20 times, it should be fast.
    # We will add a simple check just in case.
    
    # If we are close to a global timeout (which we don't know exactly, but can estimate),
    # we should return a quick result.
    # Since we don't have a global timer, we assume the caller manages the loop.
    # But to be safe, we can check if the current time is too close to a known limit?
    # No, we don't know the start time of the whole process.
    # The directive says: "Keep an EXPLICIT deadline guard: track elapsed time..."
    # This implies we should track time inside the function or have a global state.
    # Since we can't modify the caller, we can use a module-level variable to track start time.
    
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
        
    elapsed = time.time() - acquisition._start_time
    # Assume the total budget is 300s.
    # If we are past 290s, we are in danger.
    if elapsed > 290:
        # Return a very cheap, fast computation
        # Just 1/cost
        return 1.0 / cost

    # Cost-Aware UCB
    # Score = (Exploitation + Exploration) / Cost
    # Exploitation: f_best - mu (higher is better, i.e., lower mu)
    # Exploration: beta * sigma (higher is better, i.e., more uncertain)
    
    beta = 2.0
    
    # Vectorized calculation
    exploitation = f_best - mu
    exploration = beta * sigma
    
    scores = (exploitation + exploration) / cost
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
