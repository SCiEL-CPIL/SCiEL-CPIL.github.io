import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for ACO on multiple-constraint knapsack.
    Strategy: Greedy Sequential Packing.
    1. Compute a 'value density' score for each item (e.g., prize / max_weight_component).
    2. Sort items by this score.
    3. Greedily select items that fit into the capacities (25 per constraint).
    4. Assign high heuristic scores to selected items and low scores to others.
    This biases the ACO heavily towards the items found by the greedy heuristic,
    reducing the search space to a promising region and allowing pheromones
    to refine within that set.
    """
    t0 = time.time()
    DEADLINE = t0 + 320.0  # Safety margin before 330s kill

    # Ensure correct types
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Handle potential shape variations, though contract says (5, 100)
    if prizes.ndim == 1:
        P = prizes.copy()
        W = weights if weights.ndim == 2 else weights[None, :]
    else:
        # Sum prizes across constraints? The contract says prizes shape (5,100).
        # Usually in MCKP, prize is scalar per item. Here it's a vector.
        # "highest total prize" implies summing prizes for selected items.
        # If prizes are (5,100), maybe each constraint has a prize? Or is it redundant?
        # The task says "prizes and weights, of shape (5, 100)".
        # "highest total prize found by any ant".
        # If an ant picks item i, does it get prizes[:, i] or sum(prizes[:, i])?
        # Given "mean over instances of the highest total prize", and standard knapsack,
        # it's likely the total value is the sum of the prize vector components for selected items.
        # Let's assume total prize of a set S is sum_{i in S} sum_{k} prizes[k, i].
        # So P_i = sum(prizes[:, i]).
        P = prizes.sum(axis=0)  # Shape (100,)
        W = weights             # Shape (5, 100)
        
    n_constraints, n_items = W.shape
    capacity = np.full(n_constraints, 25.0)

    # 1. Compute greedy scoring metric
    # We want items that give high prize for low weight.
    # Since weights are uniform [0,1] and capacities are 25,
    # a simple ratio P_i / (sum(W[:, i]) + epsilon) is a good start.
    # Or P_i / max(W[:, i]) to be conservative on the tightest constraint.
    # Let's use the sum of weights as the "volume" metric for sorting.
    vol = W.sum(axis=0)
    # Avoid division by zero
    vol_safe = np.where(vol > 1e-9, vol, 1.0)
    density = P / vol_safe

    # 2. Sort items by density descending
    order = np.argsort(-density)
    
    # 3. Greedy Selection
    selected = np.zeros(n_items, dtype=bool)
    current_weight = np.zeros(n_constraints)
    
    # Iterate through sorted items
    for idx in order:
        if time.time() > DEADLINE - 5.0:
            break
            
        w_item = W[:, idx]
        # Check if item fits in all constraints
        if np.all(current_weight + w_item <= capacity):
            selected[idx] = True
            current_weight += w_item
    
    # 4. Assign Heuristic Scores
    # Items in the greedy solution get a high score.
    # Others get a very low score (1e-9) to effectively exclude them from initial consideration,
    # forcing the ACO to focus on the greedy set. 
    # However, if the greedy set is suboptimal, the ACO might get stuck.
    # To allow some exploration, we can give non-selected items a small but non-zero score,
    # or rank them based on their density but scaled down significantly.
    
    # Let's try a hybrid:
    # Selected items: score = 1.0 (or high value)
    # Non-selected items: score = 1e-9 (effectively 0)
    
    # Wait, if we set non-selected to 1e-9, the ACO will ONLY pick from the greedy set initially.
    # Pheromones start at 1. Prob ~ 1 * heuristic.
    # If heuristic is 1e-9, prob is ~0.
    # This essentially forces the first few ants to replicate the greedy solution.
    # If the greedy solution is good, this is great.
    # If it's slightly suboptimal, the ACO might not escape because pheromones on non-greedy items are 0.
    # But pheromones on greedy items will build up.
    # The problem is if a non-greedy item is actually better.
    
    # Alternative: Scale the density itself, but boost the selected ones.
    # h_i = density_i * boost_factor if selected else density_i * 0.01
    
    # Let's stick to the "hard-coding" idea as per the directive, but with a slight nuance.
    # The directive says: "assign high heuristic scores exclusively to these selected items and near-zero scores to all others".
    
    h = np.full(n_items, 1e-9)
    h[selected] = 1.0
    
    # One risk: What if the greedy set is empty? (Unlikely with cap 25 and weights ~0.5 avg)
    # If no items selected, h is all 1e-9. All items equal. Random walk.
    # This is a safe fallback.
    
    # Another risk: What if the greedy set is too restrictive?
    # The ACO has 20 ants. If they all pick the same greedy set, pheromones concentrate there.
    # If a better solution exists that swaps 1 item, the pheromone on that 1 item is 0.
    # It will never be picked.
    
    # To mitigate this, we can include the "next best" candidates in the selected set.
    # Let's expand the selected set to include items that were "almost" selected.
    # Or, better yet, use the density score for all, but multiply selected by a large factor.
    
    # Let's refine: 
    # Base heuristic: density (normalized to max 1)
    # Selected items: density * 100
    # Non-selected: density * 1
    # This keeps the relative ordering but heavily favors the greedy choice.
    
    max_density = np.max(density)
    if max_density > 0:
        norm_density = density / max_density
    else:
        norm_density = np.ones(n_items)
        
    h = norm_density.copy()
    h[selected] *= 100.0
    
    # Ensure positive
    h = np.maximum(h, 1e-9)
    
    # Sanitize
    h = np.asarray(h, dtype=float)
    if h.shape != (n_items,):
        h = np.ones(n_items)
        
    h = np.where(np.isfinite(h), h, 1e-9)
    
    return h
