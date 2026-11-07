"""Rewritten program for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with a small exploration bonus.
    We use a standard EI formulation but add a tiny noise to avoid division by zero
    and clip sigma to a minimum to ensure numerical stability.
    
    The key structural change here is not just the formula, but ensuring robustness.
    However, to make a stronger algorithmic choice within the constraints, 
    we can consider that pure EI can be too exploitative early on. 
    But since we are given mu and sigma from a GP, and the task is to return scores,
    let's stick to a robust EI. 
    
    Actually, a common improvement is to use a temperature or a different acquisition 
    like Probability of Improvement (PI) or a mix. But EI is generally strong.
    
    Let's look at the "STAGNATION" directive: "Make a STRUCTURAL change".
    The current program is basic EI. 
    Perhaps we can use a different acquisition function that balances exploration better, 
    or optimize the EI calculation for speed/robustness.
    
    One structural change: Use a "Safe" EI or add a small exploration term based on sigma.
    Or, use the Upper Confidence Bound (UCB) acquisition function.
    UCB = -mu + kappa * sigma (since we are minimizing, we want low mu and high sigma).
    Higher score means evaluate next. So we want high UCB score.
    UCB score = -mu + kappa * sigma.
    
    Let's try UCB. It is structurally different from EI (which is non-linear and depends on f_best).
    UCB depends on a parameter kappa. A common choice is 1.0 or 2.0.
    
    Let's try a hybrid or just UCB.
    The task says "higher meaning evaluate next".
    For minimization:
    EI: (f_best - mu) * Phi(z) + sigma * phi(z). This is positive when mu < f_best.
    UCB: -mu + kappa * sigma.
    
    Let's try UCB with kappa = 1.0.
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB acquisition function for minimization
    # We want to maximize this score
    kappa = 1.0
    score = -mu + kappa * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
