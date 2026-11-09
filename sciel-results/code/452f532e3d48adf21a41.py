"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)

def _compute_sum(indices):
    """Compute exact Fraction sum of reciprocals for given indices."""
    s = Fraction(0, 1)
    for i in indices:
        s += Fraction(1, DENOMS[i])
    return s

def _local_swap_search(indices, deadline):
    """Local search: try swapping in/out single elements to improve deficit."""
    indices = set(indices)
    current_sum = _compute_sum(indices)
    best_sum = current_sum
    best_indices = set(indices)
    
    # Precompute reciprocal values as Fractions
    recips = {i: Fraction(1, DENOMS[i]) for i in range(N)}
    
    while time.time() < deadline:
        improved = False
        # Try adding each element not in set
        for i in range(N):
            if i not in indices:
                new_sum = current_sum + recips[i]
                if new_sum < 1:
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_indices = indices | {i}
                        indices = best_indices
                        current_sum = new_sum
                        improved = True
        # Try removing each element in set (only if set has > 1 element)
        if len(indices) > 1:
            for i in list(indices):
                new_sum = current_sum - recips[i]
                if new_sum < 1 and new_sum > 0:
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_indices = indices - {i}
                        indices = best_indices
                        current_sum = new_sum
                        improved = True
        if not improved:
            break
    
    return best_indices

def _greedy_construct(deadline):
    """Greedy: start empty, add elements that keep sum < 1, prioritizing largest gains."""
    indices = set()
    current_sum = Fraction(0, 1)
    recips = {i: Fraction(1, DENOMS[i]) for i in range(N)}
    
    # Sort by reciprocal value descending
    order = sorted(range(N), key=lambda i: -DENOMS[i])  # larger denom = smaller recip, wait...
    # Actually, we want to maximize sum while staying < 1
    # Start with largest denominators (smallest reciprocals) to leave room?
    # No, greedy: add what fits. Let's try both strategies.
    
    # Strategy 1: add smallest reciprocals first (largest denominators)
    for i in range(N-1, -1, -1):  # from largest denom to smallest
        if time.time() > deadline:
            break
        if current_sum + recips[i] < 1:
            current_sum += recips[i]
            indices.add(i)
    
    # Local search to improve
    indices = _local_swap_search(indices, deadline)
    return indices

def _randomized_greedy(deadline, seed=42):
    """Randomized greedy: shuffle order, add what fits."""
    rng = random.Random(seed)
    indices = set()
    current_sum = Fraction(0, 1)
    recips = {i: Fraction(1, DENOMS[i]) for i in range(N)}
    
    order = list(range(N))
    rng.shuffle(order)
    
    for i in order:
        if time.time() > deadline:
            break
        if current_sum + recips[i] < 1:
            current_sum += recips[i]
            indices.add(i)
    
    # Local search to improve
    indices = _local_swap_search(indices, deadline)
    return indices

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Get initial solution from greedy
    best_indices = _greedy_construct(deadline)
    best_sum = _compute_sum(best_indices)
    
    # Try multiple randomized restarts
    for trial in range(50):
        if time.time() > deadline:
            break
        trial_deadline = min(deadline, time.time() + 1.0)
        trial_indices = _randomized_greedy(trial_deadline, seed=trial + 1)
        trial_sum = _compute_sum(trial_indices)
        if trial_sum > best_sum:
            best_sum = trial_sum
            best_indices = trial_indices
    
    # Final local search on best
    if time.time() < deadline:
        best_indices = _local_swap_search(best_indices, deadline)
        best_sum = _compute_sum(best_indices)
    
    # Ensure non-empty
    if not best_indices:
        best_indices = {0}
    
    return list(best_indices)
# EVOLVE-BLOCK-END
