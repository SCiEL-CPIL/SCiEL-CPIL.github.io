"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _non_dominated_front_3d(f1, f2, f3):
    """
    Compute non-dominated front for 3 objectives.
    f1: maximize (EI)
    f2: maximize (Uncertainty)
    f3: minimize (Cost)
    
    Returns:
        levels: array of non-dominated sort levels (0 is best front)
        crowding: array of crowding distances within each front
    """
    n = len(f1)
    if n == 0:
        return np.array([]), np.array([])
    
    # Convert to minimization for consistency
    # f1_min = -f1
    # f2_min = -f2
    # f3_min = f3
    f_min = np.column_stack([-f1, -f2, f3])
    
    levels = np.full(n, -1)
    crowding = np.zeros(n)
    dominated = np.zeros((n, n), dtype=bool)
    domination_count = np.zeros(n, dtype=int)
    dominated_sets = [set() for _ in range(n)]
    
    # Compute domination relationships
    for i in range(n):
        for j in range(i + 1, n):
            if np.all(f_min[i] <= f_min[j]) and np.any(f_min[i] < f_min[j]):
                dominated[i, j] = True
                domination_count[j] += 1
                dominated_sets[i].add(j)
            elif np.all(f_min[j] <= f_min[i]) and np.any(f_min[j] < f_min[i]):
                dominated[j, i] = True
                domination_count[i] += 1
                dominated_sets[j].add(i)
    
    # Non-dominated sort
    front = []
    for i in range(n):
        if domination_count[i] == 0:
            front.append(i)
            levels[i] = 0
    
    current_front = 0
    while front:
        next_front = []
        for i in front:
            for j in list(dominated_sets[i]):
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    levels[j] = current_front + 1
                    next_front.append(j)
        front = next_front
        current_front += 1
    
    # Compute crowding distance for each front
    max_level = levels.max() if n > 0 else 0
    for level in range(max_level + 1):
        front_indices = np.where(levels == level)[0]
        if len(front_indices) <= 2:
            crowding[front_indices] = np.inf
            continue
        
        for m in range(3):  # For each objective
            sorted_indices = front_indices[np.argsort(f_min[front_indices, m])]
            min_val = f_min[sorted_indices[0], m]
            max_val = f_min[sorted_indices[-1], m]
            if max_val - min_val < 1e-12:
                continue
            crowding[sorted_indices[0]] = np.inf
            crowding[sorted_indices[-1]] = np.inf
            for k in range(1, len(sorted_indices) - 1):
                crowding[sorted_indices[k]] += (
                    f_min[sorted_indices[k + 1], m] - f_min[sorted_indices[k - 1], m]
                ) / (max_val - min_val)
    
    return levels, crowding


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty (sigma)
    uncertainty = sigma
    
    # Cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Pareto fronts
    levels, crowding = _non_dominated_front_3d(ei, uncertainty, cost_safe)
    
    # Check time
    if time.time() > deadline:
        # Fallback: return EI/cost as simple heuristic
        return ei / cost_safe
    
    # Score: higher is better
    # Lower level is better, higher crowding is better
    # Combine into a single score
    max_level = levels.max() if n > 0 else 1
    level_score = (max_level - levels) / (max_level + 1)
    
    # Normalize crowding
    max_crowding = np.max(crowding) if np.max(crowding) > 0 else 1.0
    crowding_norm = crowding / max_crowding
    
    # Final score: prioritize lower levels, then higher crowding
    # This creates a tiered selection where front 0 candidates are strongly preferred
    score = level_score * 1000 + crowding_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
