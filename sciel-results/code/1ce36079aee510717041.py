"""Direct, structure-driven heuristic for ACO multiple-constraint knapsack."""
import time
import numpy as np

T_START = time.perf_counter()
LIMIT = 300.0   # hard wall-clock bound from evaluator
MARGIN = 10.0   # safety margin
DEADLINE = T_START + (LIMIT - MARGIN)


def _deadline_left():
    return (T_START + LIMIT) - time.perf_counter()


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Simulated Annealing Greedy Set Membership with Constraint-Aware Prizes.
    
    1. Run a simulated annealing greedy search to find high-value feasible subsets.
    2. Track which items appear in high-value solutions.
    3. Combine with constraint-aware prize signals.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shapes: prizes (5, 100) -> (100,), weights (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Deadline guard
    if _deadline_left() < 5.0:
        return np.ones(n_items) * 1e-9
    
    # --- Part 1: Simulated Annealing Greedy Search ---
    # Track frequency and value-weighted frequency of items in high-value solutions
    item_frequency = np.zeros(n_items, dtype=float)
    item_value_weighted = np.zeros(n_items, dtype=float)
    best_solution_value = 0.0
    best_solution_items = set()
    
    # Temperature schedule
    initial_temp = 10.0
    cooling_rate = 0.9
    min_temp = 0.01
    
    # Number of SA iterations (bounded by deadline)
    max_sa_iters = 200
    
    # Precompute weight metrics
    sum_w = weights.sum(axis=0)
    max_w = weights.max(axis=0)
    
    # Initial solution: greedy by prize/sum_weight
    ratio = prizes / (sum_w + 1e-9)
    order = np.argsort(-ratio)
    
    current_solution = set()
    remaining_capacity = np.full(n_constraints, capacity, dtype=float)
    for idx in order:
        if np.all(weights[:, idx] <= remaining_capacity):
            current_solution.add(idx)
            remaining_capacity -= weights[:, idx]
    
    # Compute initial value
    current_value = sum(prizes[i] for i in current_solution)
    best_solution_value = current_value
    best_solution_items = set(current_solution)
    
    temp = initial_temp
    rng = np.random.RandomState(42)
    
    for sa_iter in range(max_sa_iters):
        if _deadline_left() < 10.0:
            break
        
        if temp < min_temp:
            break
        
        # Record current solution statistics
        for idx in current_solution:
            item_frequency[idx] += 1.0
            item_value_weighted[idx] += current_value
        
        # Try a random move: add or remove an item
        if rng.random() < 0.5 and len(current_solution) < n_items:
            # Try to add an item
            candidates = [i for i in range(n_items) if i not in current_solution]
            if candidates:
                idx = candidates[rng.randint(len(candidates))]
                if np.all(weights[:, idx] <= remaining_capacity):
                    new_solution = current_solution | {idx}
                    new_value = current_value + prizes[idx]
                    # Accept if better or with probability exp((new-old)/temp)
                    if new_value >= current_value or rng.random() < np.exp((new_value - current_value) / temp):
                        current_solution = new_solution
                        remaining_capacity -= weights[:, idx]
                        current_value = new_value
                        if current_value > best_solution_value:
                            best_solution_value = current_value
                            best_solution_items = set(current_solution)
        else:
            # Try to remove an item
            if current_solution:
                idx = list(current_solution)[rng.randint(len(current_solution))]
                new_solution = current_solution - {idx}
                new_value = current_value - prizes[idx]
                # Always accept removals (simplification)
                current_solution = new_solution
                remaining_capacity += weights[:, idx]
                current_value = new_value
        
        temp *= cooling_rate
    
    # Final record
    for idx in current_solution:
        item_frequency[idx] += 1.0
        item_value_weighted[idx] += current_value
    
    # Normalize
    freq_max = item_frequency.max()
    if freq_max > 0:
        freq_norm = item_frequency / freq_max
    else:
        freq_norm = np.ones(n_items)
    
    valw_max = item_value_weighted.max()
    if valw_max > 0:
        valw_norm = item_value_weighted / valw_max
    else:
        valw_norm = np.ones(n_items)
    
    # --- Part 2: Constraint-Aware Prize Signal ---
    # For each item, compute how "easy" it is to fit given the tightest constraint
    safe_weights = np.maximum(weights, 1e-9)
    slack_ratios = (capacity - weights) / safe_weights
    slack_ratios = np.maximum(slack_ratios, 0.0)
    bottleneck_slack = slack_ratios.min(axis=0)
    
    bs_max = bottleneck_slack.max()
    if bs_max > 0:
        bs_norm = bottleneck_slack / bs_max
    else:
        bs_norm = np.ones(n_items)
    
    # Prize normalization
    p_max = prizes.max()
    if p_max > 0:
        p_norm = prizes / p_max
    else:
        p_norm = np.ones(n_items)
    
    # Efficiency: prize / max_weight (how much prize per unit of tightest resource)
    eff = prizes / (max_w + 1e-9)
    eff_max = eff.max()
    if eff_max > 0:
        eff_norm = eff / eff_max
    else:
        eff_norm = np.ones(n_items)
    
    # --- Part 3: Combine Signals ---
    # Weighted product: emphasize items that are frequently in high-value solutions,
    # have high prize, fit easily, and have good efficiency
    scores = (freq_norm ** 1.5) * (valw_norm ** 1.0) * (bs_norm ** 1.0) * (p_norm ** 1.0) * (eff_norm ** 0.5)
    
    # Add small base to avoid zeros
    scores = scores + 1e-9
    
    # Normalize to [0, 1]
    s_max = scores.max()
    if s_max > 0:
        scores = scores / s_max
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
