"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a pure exploitation strategy based on the posterior mean.
    Given the dense set of 2000 candidates and the low noise of the GP,
    minimizing the predicted mean is the most direct path to finding the
    global minimum. This avoids the exploration overhead of EI/UCB which
    can be suboptimal when the candidate pool is already large and diverse.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    
    # Pure exploitation: higher score means lower predicted mean
    # We negate the mean so that the argmax corresponds to the argmin of mu
    scores = -mu
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    # If there are any infinities, replace them with a large finite value
    # to be safe against any downstream issues, though -inf for bad points is fine
    # Actually, the contract says "Return 2000 finite scores".
    # So we must map -inf to a very large negative number and +inf to a very large positive number.
    # But since we are minimizing, -inf (bad) should have low score, +inf (good) should have high score.
    # Wait, mu is the mean. If mu is -inf (impossible for GP), score is +inf.
    # If mu is +inf (impossible for GP), score is -inf.
    # Let's just use a safe large number.
    max_float = 1e30
    scores = np.clip(scores, -max_float, max_float)
    
    return scores
# EVOLVE-BLOCK-END
