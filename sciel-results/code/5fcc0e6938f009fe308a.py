"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


class BOState:
    def __init__(self):
        self.step = 0
        self.f_best_history = []
        self.start_time = time.time()
        self.deadline = 300.0  # seconds

    def is_expired(self):
        return (time.time() - self.start_time) > (self.deadline - 5.0)


state = BOState()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid approach:
    - Early steps: aggressive exploration (high beta UCB)
    - Later steps: balanced exploration/exploitation
    - Near deadline: conservative exploitation
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    state.step += 1
    
    # Track f_best history for adaptive behavior
    if len(state.f_best_history) == 0:
        state.f_best_history.append(f_best)
    else:
        state.f_best_history.append(f_best)
    
    # Keep history manageable
    if len(state.f_best_history) > 35:
        state.f_best_history = state.f_best_history[-35:]
    
    # Calculate recent improvement rate
    if len(state.f_best_history) >= 2:
        recent_improvement = state.f_best_history[-2] - state.f_best_history[-1]
    else:
        recent_improvement = 0.0
    
    # Determine phase based on step count
    total_steps_expected = 30  # 5 initial + 25 more evaluations
    
    if state.step <= 8:
        # Early phase: strong exploration
        beta = 3.0
        use_ucb = True
    elif state.step <= 15:
        # Mid phase: balanced
        beta = 1.5
        use_ucb = True
    elif state.step <= 22:
        # Late phase: more exploitation
        beta = 0.8
        use_ucb = True
    else:
        # Final phase: exploitation-focused with slight exploration
        beta = 0.3
        use_ucb = True
    
    # If no recent improvement, increase exploration
    if recent_improvement < 1e-4 and state.step > 5:
        beta = max(beta, 1.0)
    
    # Compute scores
    if use_ucb:
        # UCB for minimization: score = -mu + beta * sigma
        # Higher score = more likely to find lower value
        scores = -mu + beta * sigma
    else:
        # Fallback to EI
        z = (f_best - mu) / (sigma + 1e-9)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
