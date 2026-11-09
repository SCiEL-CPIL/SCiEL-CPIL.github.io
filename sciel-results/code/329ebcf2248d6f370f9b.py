"""Adaptive Hybrid Acquisition for Bayesian Optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# State tracking for adaptive behavior
_state = {
    "start_time": None,
    "f_best_history": [],
    "call_count": 0
}

def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining UCB and EI.
    
    Early in the optimization process (when f_best is high or few calls made),
    it favors exploration (UCB) to map the multi-modal landscape.
    Later, it shifts to exploitation (EI) to refine the best solution.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Initialize state
    if _state["start_time"] is None:
        _state["start_time"] = time.time()
    
    # Safety deadline guard: if we are dangerously close to the 330s limit,
    # return a simple, fast EI score to avoid timeout.
    elapsed = time.time() - _state["start_time"]
    if elapsed > 325.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        improvement = f_best - mu
        z = improvement / sigma_safe
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    _state["call_count"] += 1
    _state["f_best_history"].append(f_best)
    
    # Normalize sigma for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate EI (Expected Improvement)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate UCB (Upper Confidence Bound)
    # We use a dynamic kappa. 
    # In the beginning, we want to explore. As f_best decreases (improves), 
    # we shift to exploitation.
    # Goldstein-Price min is 3. Initial uniform random points often yield high values (e.g., 10-20).
    # If f_best is still high relative to the minimum (3.0), we are likely in exploration.
    
    # Heuristic for progress:
    # Let's define a "confidence" in the current best.
    # If f_best is close to 3.0, we are exploiting.
    # If f_best is far from 3.0, we are exploring.
    
    # A simple mapping: 
    # progress = 1.0 - (f_best - 3.0) / (initial_f_best_estimate - 3.0)
    # But we don't know initial_f_best_estimate exactly.
    # Let's use the history.
    
    hist = _state["f_best_history"]
    if len(hist) > 1:
        initial_best = hist[0]
        # If initial_best is very close to current, maybe we are stuck? 
        # Let's use a fixed scale for Goldstein-Price.
        # Typical range of GP function is [3, ~18].
        # Let's say if f_best < 5.0, we are in exploitation phase.
        # If f_best > 8.0, we are in exploration phase.
        
        # Transition weight w: 1.0 means full UCB, 0.0 means full EI.
        # We want w to decrease as f_best improves.
        
        # Use a sigmoid-like transition or linear interpolation.
        # Let's map f_best from [10, 3] to w from [1.0, 0.2].
        # If f_best < 3, w=0.2 (still some exploration to avoid local minima).
        # If f_best > 10, w=1.0 (heavy exploration).
        
        w_min = 0.2
        w_max = 1.0
        f_high = 10.0
        f_low = 3.0
        
        if f_best >= f_high:
            w = w_max
        elif f_best <= f_low:
            w = w_min
        else:
            # Linear interpolation
            w = w_min + (w_max - w_min) * (f_best - f_low) / (f_high - f_low)
    else:
        # First call, assume heavy exploration
        w = w_max

    # Scale EI to be comparable to UCB?
    # UCB = mu + kappa * sigma.
    # EI is roughly improvement * prob + sigma * pdf.
    # To combine them meaningfully, we can normalize both.
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei)
        
    # Normalize UCB to [0, 1]
    # We need to pick kappa. A standard kappa for early exploration is around 1.0-2.0.
    # Let's use kappa = 1.5 * w + 0.5 * (1-w) ? No, let's just use a fixed kappa for UCB term
    # and blend the normalized scores.
    
    # Actually, blending normalized scores is safer.
    # UCB score: mu + kappa * sigma
    # Let's use kappa = 1.0 for the UCB component.
    kappa = 1.0
    ucb_raw = mu + kappa * sigma_safe
    
    ucb_min = np.min(ucb_raw)
    ucb_max = np.max(ucb_raw)
    if ucb_max > ucb_min:
        ucb_norm = (ucb_raw - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones_like(ucb_raw)
        
    # Hybrid Score
    scores = w * ucb_norm + (1 - w) * ei_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
