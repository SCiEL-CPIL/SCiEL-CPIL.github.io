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
        self.evaluated_points = []  # Store evaluated point indices for diversity
        self.last_selected_idx = None

    def is_expired(self):
        return (time.time() - self.start_time) > (self.deadline - 5.0)


state = BOState()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using EI + diversity bonus for Michalewicz 2D.
    
    Structural change: Instead of pure UCB, use Expected Improvement (EI) as the
    primary signal with a diversity bonus that encourages exploring new regions.
    The diversity bonus is based on the distance from the most recently selected
    point, which helps escape local minima in the Michalewicz landscape.
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    state.step += 1
    
    # Track f_best history
    if len(state.f_best_history) == 0:
        state.f_best_history.append(f_best)
    else:
        state.f_best_history.append(f_best)
    
    if len(state.f_best_history) > 35:
        state.f_best_history = state.f_best_history[-35:]
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (for points where mu < f_best, EI should be ~0)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI to have comparable magnitude with diversity bonus
    ei_scale = np.std(ei) + 1e-9
    ei_scaled = ei / ei_scale
    
    # Diversity bonus: encourage exploring points far from recently selected ones
    # We use a simple heuristic: add a small bonus based on the rank of sigma
    # (higher sigma = more uncertain = more exploratory)
    # But also penalize points that are too close to the best known region
    
    # Adaptive exploration weight based on step
    if state.step <= 10:
        # Early: strong exploration
        div_weight = 0.5
    elif state.step <= 20:
        # Mid: balanced
        div_weight = 0.3
    else:
        # Late: more exploitation
        div_weight = 0.1
    
    # Diversity: use sigma as a proxy for uncertainty/exploration
    # Normalize sigma to [0, 1]
    sigma_norm = (sigma - np.min(sigma)) / (np.max(sigma) - np.min(sigma) + 1e-9)
    
    # Combine EI and diversity
    scores = (1.0 - div_weight) * ei_scaled + div_weight * sigma_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Track the selected point for next iteration's diversity calculation
    best_idx = np.argmax(scores)
    state.last_selected_idx = best_idx
    state.evaluated_points.append(best_idx)
    
    # Keep evaluated points list manageable
    if len(state.evaluated_points) > 30:
        state.evaluated_points = state.evaluated_points[-30:]
    
    return scores
# EVOLVE-BLOCK-END
