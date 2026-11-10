"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0 - 5.0  # Safety margin

    # Convert to numpy arrays for consistent behavior
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Handle potential 2D prizes by reducing to 1D if necessary
    if prizes.ndim == 2:
        prize_vector = np.sum(prizes, axis=0)
    else:
        prize_vector = prizes.flatten()

    # Ensure weights is (5, 100)
    n_constraints, n_items = weights.shape
    capacity = 25.0

    # --- Baseline Greedy Solution ---
    # Sort items by density (prize / total_weight) descending
    # Total weight is sum across constraints
    weight_sums = np.sum(weights, axis=0)
    safe_weight_sums = np.maximum(weight_sums, 1e-9)
    densities = prize_vector / safe_weight_sums
    
    # Indices sorted by density descending
    sorted_indices = np.argsort(densities)[::-1]
    
    # Construct baseline solution
    remaining_cap = np.full(n_constraints, capacity)
    in_baseline = np.zeros(n_items, dtype=bool)
    baseline_prize = 0.0
    
    for idx in sorted_indices:
        # Check if item fits
        if np.all(weights[:, idx] <= remaining_cap):
            in_baseline[idx] = True
            remaining_cap -= weights[:, idx]
            baseline_prize += prize_vector[idx]
    
    # --- Compute Regret Scores ---
    # For each item i, compute the drop in total prize if i is excluded from the baseline.
    # If i is in the baseline:
    #   We remove i. Then we try to fill the freed capacity with other items (greedily).
    #   Regret_i = (Baseline_Prize) - (New_Prize_without_i)
    #   Note: This is equivalent to the value of i plus the opportunity cost of the items it might have blocked.
    #   Actually, a simpler approximation: 
    #   Regret_i = Prize_i + (Value of items that could fit if i was removed but don't fit otherwise? No, that's complex)
    #   
    # Let's use a simpler "Marginal Value" approach which is a proxy for regret:
    # If i is IN baseline: Regret is high (we lose its prize). 
    # If i is NOT in baseline: Regret is 0? No, we might want to include it if it fits better.
    #
    # The directive asks for "drop in total prize of a baseline greedy solution if that item is forced to be excluded".
    # If the item is NOT in the baseline, forcing it to be excluded changes nothing. Regret = 0.
    # If the item IS in the baseline, we remove it. 
    #   New_Capacity = Baseline_Remaining_Capacity + Weights_i
    #   We then run a greedy fill on the REMAINING items (excluding i) with this new capacity.
    #   New_Prize = Sum of prizes of items added in this second pass.
    #   Regret_i = Baseline_Prize - (New_Prize + 0)  <-- Wait, Baseline_Prize included i.
    #   Actually: 
    #   Baseline = S. Prize(S) = P.
    #   S' = S \ {i}. 
    #   We can add more items to S' because we freed up capacity.
    #   Let S'' be the result of greedy filling S' with remaining items.
    #   Regret_i = P - Prize(S'')
    #   Since S'' contains S' plus possibly more, Prize(S'') >= Prize(S') = P - Prize(i).
    #   So Regret_i <= Prize(i).
    #   This measures how "irreplaceable" i is. If i is very dense, removing it and trying to replace it with less dense items (since they were lower in the sort order) will result in a lower total prize, so Regret will be close to Prize(i).
    #   If i is low density but was included because capacity was left over, removing it might allow a slightly denser item (that was previously blocked) to be added? No, the greedy order is fixed by density. Items not in S were blocked by capacity. Removing i frees capacity. The next best items (by density) that now fit will be added.
    #   So Regret_i = Prize(i) - (Prize(Added_Items) - 0)? 
    #   Let's trace:
    #   P_baseline = Sum(Prize(j) for j in S)
    #   Remove i. Capacity increases.
    #   Add items from sorted_indices (skipping those already in S\{i} and i) that fit.
    #   Let A be the set of newly added items.
    #   P_new = Sum(Prize(j) for j in S\{i}) + Sum(Prize(j) for j in A)
    #   Regret_i = P_baseline - P_new
    #   P_baseline = Prize(i) + Sum(Prize(j) for j in S\{i})
    #   Regret_i = Prize(i) - Sum(Prize(j) for j in A)
    #   This is a good signal! If i is high value and A has low total value, Regret is high.
    #   If i is low value and A has high total value (unlikely since A is from lower density), Regret could be low or negative?
    #   Since A items are lower density than i (because i was processed earlier), it's likely that Sum(Prize(A)) < Prize(i) if the capacity freed is small. But if i is huge and low density, and it blocks many small high density items? No, high density items are processed first.
    #   So this is a valid local search move evaluation.

    regret_scores = np.zeros(n_items)
    
    # Pre-compute: which items are in baseline?
    # We need to efficiently re-simulate the greedy fill for each item in the baseline.
    # n_items = 100. Baseline size is likely ~25 (capacity 25, weights ~1).
    # So we do ~25 simulations. Each simulation is O(n_items * n_constraints).
    # 25 * 100 * 5 = 12500 ops. Very fast.
    
    baseline_items = [i for i in range(n_items) if in_baseline[i]]
    
    for i in baseline_items:
        # Start with capacity freed by removing i
        current_cap = remaining_cap.copy()
        current_cap += weights[:, i]
        
        # Mark items already in baseline (excluding i) as taken
        taken = in_baseline.copy()
        taken[i] = False
        
        new_prize = 0.0
        # Iterate through items in density order
        for idx in sorted_indices:
            if taken[idx]:
                continue
            # Check if fits
            if np.all(weights[:, idx] <= current_cap):
                taken[idx] = True
                current_cap -= weights[:, idx]
                new_prize += prize_vector[idx]
        
        # Calculate regret
        # P_baseline is known.
        # P_new is new_prize.
        # Regret = P_baseline - P_new
        regret_scores[i] = baseline_prize - new_prize
        
        # Check time
        if time.time() > deadline:
            break

    # For items NOT in the baseline, the regret is 0 by definition (excluding them doesn't change the baseline).
    # However, we might want to give them a small positive heuristic if they are "almost" fitting or have high density?
    # The directive says "use this regret value as the heuristic".
    # If regret is 0, heuristic is 0. But heuristic must be positive (min 1e-9).
    # This would make ants ignore non-baseline items initially.
    # This might be bad if the baseline is poor.
    # But the baseline is greedy by density, which is a strong heuristic for knapsack.
    # Let's assume the baseline is good.
    # To be safe, let's add a small base heuristic for non-baseline items based on their density, scaled down.
    # Or just set them to 1e-9?
    # The ACO will update pheromones. If an ant picks a non-baseline item, it might find a better solution.
    # But if heuristic is 0, probability is 0.
    # Let's add a small epsilon-like value based on density for non-baseline items to allow exploration.
    # heuristic = regret + small * density
    
    # Normalize regret scores?
    # Max regret is around max prize (~5).
    # Let's scale regret to be comparable to previous heuristics.
    
    # For non-baseline items, assign a small value proportional to density
    non_baseline_mask = ~in_baseline
    if np.any(non_baseline_mask):
        # Scale density to be small compared to typical regret
        # Typical regret ~ O(1). Density ~ O(1).
        # Let's use 0.1 * density
        regret_scores[non_baseline_mask] = 0.1 * densities[non_baseline_mask]

    # Ensure positive
    heuristic = np.maximum(regret_scores, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
