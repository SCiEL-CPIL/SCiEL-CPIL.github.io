import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return a simple greedy score
    elapsed = time.time() - _START_TIME
    if elapsed > 290:
        return -mu

    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Compute total cost to determine budget progress
    total_cost = np.sum(cost)

    # Budget is 30, so progress goes from 0 to 1 as cost approaches 30
    budget_progress = np.clip(total_cost / 30.0, 0.0, 1.0)

    # Exploration coefficient for the floor term
    # Higher early to encourage exploration, lower late to exploit
    exploration_weight = 0.5 * (1.0 - budget_progress)

    # z = (f_best - mu) / sigma
    z = (f_best - mu) / s

    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)

    ei = (f_best - mu) * Phi_z + s * phi_z

    # Ensure EI is non-negative (numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)

    # Add exploration floor: exploration_weight * sigma / cost
    # This ensures that even when EI is 0 (high confidence that f(x) > f_best),
    # uncertain regions still have some score if they are cheap.
    exploration_floor = exploration_weight * s / c

    # Score = EI / cost + exploration_floor
    score = ei / c + exploration_floor

    # Clean up
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
