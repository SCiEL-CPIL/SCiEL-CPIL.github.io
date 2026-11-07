"""Cost-aware acquisition function for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time for the first call in a session
_acq_start_time = None
_ACQ_DEADLINE_S = 320.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Knowledge Gradient (KG) with
    adaptive exploration.
    
    KG directly estimates the expected reduction in the best observed value
    after evaluating a candidate. We combine this with an adaptive LCB-like
    bonus that scales with the uncertainty, and use a softened cost penalty
    to balance exploration and exploitation while respecting the budget.
    
    The acquisition score is:
      score = (KG + exploration_bonus) / cost^0.8
    
    where exploration_bonus = kappa * sigma, and kappa adapts based on f_best.
    """
    global _acq_start_time
    
    # Initialize deadline tracking
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Check deadline
    elapsed = time.time() - _acq_start_time
    if elapsed > _ACQ_DEADLINE_S:
        # Fallback to a simple cost-aware LCB if we're running out of time
        s = np.maximum(sigma, 1e-9)
        lcb = mu - 1.5 * s
        improvement = f_best - lcb
        c = np.maximum(cost, 1e-9)
        scores = improvement / c
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores = np.maximum(scores, 0.0)
        return scores
    
    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-9)
    
    # Compute Knowledge Gradient: expected reduction in best value
    # mu: GP mean at candidate (shape: 2000,)
    # s: GP std at candidate (shape: 2000,)
    # f_best: scalar, current best observed value
    
    d = (f_best - mu) / s
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    kg = (f_best - mu) * Phi_d + s * phi_d
    
    # Ensure non-negative
    kg = np.maximum(kg, 0.0)
    
    # Adaptive exploration bonus
    # If f_best is high (early in optimization), explore more.
    # If f_best is low (close to optimum), exploit more.
    # f_best for Ackley 2D is typically between 0 and ~20.
    # Map f_best to a progress indicator: low f_best -> high progress.
    alpha = 0.3
    progress = 1.0 - np.exp(-alpha * f_best)
    progress = np.clip(progress, 0.0, 1.0)
    
    # kappa: exploration parameter
    # When progress is low (early), kappa is high (more exploration).
    # When progress is high (late), kappa is low (more exploitation).
    kappa_min = 0.5
    kappa_max = 2.0
    kappa = kappa_min + (kappa_max - kappa_min) * (1.0 - progress)
    
    # Exploration bonus: proportional to uncertainty
    exploration_bonus = kappa * s
    
    # Combine KG and exploration bonus
    # The KG already captures some exploration (via the sigma * phi(d) term),
    # but adding an explicit LCB-like bonus can help escape local optima.
    # We use a weighted combination to avoid double-counting.
    combined = kg + 0.5 * exploration_bonus
    
    # Cost awareness with softened penalty
    c = np.maximum(cost, 1e-9)
    cost_softened = np.power(c, 0.8)
    
    scores = combined / cost_softened
    
    # Safety: Ensure finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
