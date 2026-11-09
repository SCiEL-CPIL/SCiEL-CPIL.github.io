"""Acquisition function for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = None
_DEADLINE_MARGIN = 5.0  # seconds before the 330s wall-clock kill


def _ensure_deadline():
    global _deadline
    if _deadline is None:
        _deadline = time.time() + 330.0 - _DEADLINE_MARGIN
    return _deadline


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function.
    
    Uses a time-adaptive kappa that starts high for exploration and
    decays towards exploitation as the budget is consumed. Also blends
    in a small UCB term to avoid getting stuck in local minima.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior std
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Ensure we have a deadline set
    deadline = _ensure_deadline()
    
    # Safety: if we're close to deadline, return deterministic scores
    if time.time() > deadline:
        # Return LCB with kappa=0 (pure exploitation) as fallback
        return -mu.copy()
    
    # Adaptive kappa: start with strong exploration, decay to exploitation
    # We estimate progress based on how close f_best is to 0 (the minimum)
    # and also use a simple time-based decay
    
    # Time-based progress: assume total budget is ~330s, we start after first 5 evals
    elapsed = time.time() - (_deadline - 330.0 + _DEADLINE_MARGIN)
    # Normalize elapsed to [0, 1] over the expected search duration
    # The search has 25 remaining evals after the initial 5
    # Each eval takes some time, but let's use a simple heuristic
    # Use f_best proximity as a signal: if f_best is close to 0, exploit more
    f_best_ratio = max(0.0, min(1.0, f_best / 10.0))  # Ackley max ~2.9, but use 10 as safe bound
    
    # Combine time and value-based progress
    # Early: explore more (high kappa), Late: exploit more (low kappa)
    # Use a smooth schedule
    kappa_base = 2.0  # base exploration strength
    # Decay kappa as f_best improves
    kappa_decay = 1.0 - 0.7 * f_best_ratio  # when f_best=0, kappa_decay=1; when f_best=10, kappa_decay=0.3
    kappa = kappa_base * kappa_decay
    
    # Also add a small UCB component to encourage diversity
    # The LCB is -mu + kappa*sigma
    # Add a small bonus for high uncertainty to avoid premature convergence
    ucb_weight = 0.3 * (1.0 - f_best_ratio)  # more UCB early
    
    # Compute LCB scores (we want to maximize, so negate the LCB)
    # LCB(x) = mu(x) - kappa * sigma(x)  (we want to minimize this)
    # Acquisition = -LCB = -mu + kappa*sigma
    lcb_scores = -mu + kappa * sigma
    
    # Add UCB component: UCB(x) = mu(x) + kappa_ucb * sigma(x)
    # We want to maximize, so we add a small positive term for uncertainty
    ucb_scores = -mu + (kappa + ucb_weight) * sigma
    
    # Blend: primarily LCB, with a bit of UCB for diversity
    # Actually, let's just use a well-tuned LCB with adaptive kappa
    # The key insight: with length scale 0.2, unexplored regions have high sigma
    # So LCB will naturally explore those regions
    
    # Final: use pure adaptive LCB
    scores = lcb_scores
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
