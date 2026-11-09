"""Cost-aware acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We keep a safety margin so we return before the hard limit.
_DEADLINE = 320.0
_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).

    mu, sigma: shape (2000,), GP mean and std in objective units.
    f_best: scalar, lowest observed value.
    cost: shape (2000,), cost of evaluating each candidate.
    
    This implementation uses a Cost-Aware Lower Confidence Bound (LCB) 
    for minimization, which is structurally different from Expected Improvement.
    Score = (-mu + beta * sigma) / cost
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return a safe fallback: uniform random scores
        return np.random.uniform(0, 1, size=len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma = np.maximum(sigma, 1e-9)
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)

    # Structural Change: Use LCB (Lower Confidence Bound) for minimization
    # Instead of EI which focuses on improvement over f_best, LCB focuses on
    # the potential low value of the function itself, weighted by uncertainty.
    # score = (-mu + beta * sigma) / cost
    # High score means: low predicted value (good for minimization) AND high uncertainty (exploration)
    # Divided by cost to prioritize cheap evaluations.
    
    beta = 1.5
    
    # Calculate the acquisition score
    # -mu: We want low mu, so we negate it to make high scores correspond to low values.
    # beta * sigma: We want high uncertainty to encourage exploration.
    raw_score = -mu + beta * sigma
    
    # Normalize by cost
    acq = raw_score / cost

    # Ensure finite values
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    # Shift scores to be positive if necessary, though only relative order matters.
    # Adding a constant doesn't change the argmax.
    # However, if all scores are negative, it's fine.
    
    return acq
# EVOLVE-BLOCK-END
