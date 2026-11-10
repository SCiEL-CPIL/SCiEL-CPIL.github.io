import numpy as np
import time

_start_time = None
_GLOBAL_DEADLINE = 135.0  # Leave 15s buffer

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_time_up():
    return _get_elapsed() > _GLOBAL_DEADLINE

def _fractional_knapsack_upper_bound(weights, values, capacity):
    """
    Compute the fractional knapsack upper bound for the given items and capacity.
    This is the optimal value if we could take fractions of items.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    # Filter feasible items (weight > 0 and weight <= capacity is not strictly required for the bound calculation 
    # if we just sort and take fractions, but items with w > C contribute 0 to the bound if we are careful, 
    # or rather, the fractional part handles it. 
    # Actually, standard fractional knapsack sorts by density and takes as much as possible.
    # Items with weight > capacity can only be taken fractionally.
    
    # We need to handle weights == 0 carefully to avoid division by zero.
    # If weight is 0 and value > 0, it's infinite density? Or just take it.
    # In the context of upper bound, if w=0, v>0, we take all of them.
    
    # For the bound, we assume we can take fractions.
    # Let's sort by density v/w.
    
    # Filter out items with weight <= 0? 
    # If w <= 0 and v > 0, they are always taken. 
    # If w <= 0 and v <= 0, they are never taken.
    # The problem statement implies standard knapsack where weights are positive.
    # But to be robust:
    
    positive_weight_mask = weights > 0
    zero_weight_mask = (weights == 0) & (values > 0)
    
    bound = np.sum(values[zero_weight_mask])
    
    if not np.any(positive_weight_mask):
        return bound
        
    w_pos = weights[positive_weight_mask]
    v_pos = values[positive_weight_mask]
    
    densities = v_pos / w_pos
    order = np.argsort(-densities)
    
    rem_cap = capacity
    total_val = bound
    
    for i in order:
        if rem_cap <= 0:
            break
        if w_pos[i] <= rem_cap:
            total_val += v_pos[i]
            rem_cap -= w_pos[i]
        else:
            total_val += v_pos[i] * (rem_cap / w_pos[i])
            rem_cap = 0
            
    return total_val

def _greedy_completion(weights, values, capacity, excluded_idx):
    """
    Greedy completion: given a set of items (with one excluded), 
    greedily pick items by density until capacity is full.
    Returns the total value achieved.
    """
    if capacity <= 0:
        return 0.0
    
    # Filter out excluded item
    mask = np.ones(len(weights), dtype=bool)
    mask[excluded_idx] = False
    w = weights[mask]
    v = values[mask]
    
    if len(w) == 0:
        return 0.0
    
    # Sort by density descending
    densities = v / w
    order = np.argsort(-densities)
    sorted_w = w[order]
    sorted_v = v[order]
    
    total_val = 0.0
    remaining_cap = capacity
    
    for i in range(len(sorted_w)):
        if sorted_w[i] <= remaining_cap:
            total_val += sorted_v[i]
            remaining_cap -= sorted_w[i]
        else:
            break
            
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if _is_time_up():
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Handle zero weight positive value items first
    zero_weight_indices = np.where((weights == 0) & (values > 0))[0]
    if len(zero_weight_indices) > 0:
        return int(zero_weight_indices[0])
    
    # Filter feasible items (weight > 0 and weight <= capacity)
    feasible_mask = (weights <= remaining_capacity) & (weights > 0)
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
        
    C = remaining_capacity
    
    # Calculate densities
    densities = f_values / f_weights
    
    # Sort by density descending
    density_order = np.argsort(-densities)
    
    # Calculate the global fractional upper bound for the current state
    # This represents the "price" of the remaining capacity in terms of value
    global_bound = _fractional_knapsack_upper_bound(f_weights, f_values, C)
    
    # Strategy: Opportunity Cost based selection
    # For each candidate item, we estimate the opportunity cost.
    # The idea: If we pick item i, we lose the ability to pick other items.
    # A proxy for opportunity cost is the marginal loss in the fractional upper bound 
    # if we were to exclude item i from the consideration of the remaining capacity?
    # Or rather, the value of the capacity consumed by item i, valued at the marginal rate of the fractional solution.
    
    # Let's refine the "Combinatorial Auction" idea:
    # Price of item i = Value of the capacity it consumes, estimated by the fractional knapsack marginal value.
    # In fractional knapsack, items are taken in order of density.
    # If we take item i, we consume w_i capacity.
    # The "cost" is the value of the best fractional items that would have used that capacity if i wasn't there.
    
    # Simpler heuristic consistent with the prompt's "marginal loss in fractional upper bound":
    # Let UB(S) be the fractional bound of set S with capacity C.
    # If we pick item i, the new state is S \ {i} with capacity C - w_i.
    # The "loss" is UB(S) - (v_i + UB(S \ {i}, C - w_i)).
    # We want to minimize this loss, or maximize v_i - Loss?
    # Actually, we want to pick the item that leads to the best future.
    # The prompt suggests: select item that maximizes ratio of value to opportunity cost.
    # Opportunity Cost (OC) = UB(S) - UB(S \ {i}, C) ? No, that's the value of item i in the fractional solution.
    
    # Let's try: OC_i = Fractional Bound of remaining items (excluding i) with FULL capacity C?
    # No, that doesn't account for capacity consumption.
    
    # Let's stick to the prompt's specific phrasing: 
    # "price of an item is its opportunity cost, estimated by the marginal loss in the fractional upper bound 
    # of the remaining subproblem if that item is excluded"
    
    # Let UB_full = Fractional Bound of all items with capacity C.
    # If we exclude item i, we have items S\{i} and capacity C.
    # UB_excl_i = Fractional Bound of S\{i} with capacity C.
    # Marginal Loss L_i = UB_full - UB_excl_i.
    # This L_i is effectively the value that item i contributes to the fractional solution.
    # If L_i is high, item i is very valuable in the fractional sense.
    # The "price" is L_i.
    # We want to maximize Value / Price? That would be v_i / L_i.
    # If v_i == L_i (which is true if item i is fully taken in the fractional solution), ratio is 1.
    # If item i is partially taken, L_i < v_i? No, L_i is the fractional value contributed.
    # If item i is not taken in fractional solution, L_i = 0? Then division by zero.
    
    # This ratio v_i / L_i seems to favor items that are "efficient" in the fractional sense?
    # Actually, if L_i is small, it means excluding it doesn't hurt the fractional bound much, so it's "cheap".
    # But we want to pick items that are good for the INTEGER solution.
    
    # Alternative interpretation:
    # The "price" is the capacity cost.
    # Let's use the greedy lookahead from the current program, but refine the scoring.
    # Current program: Simulate adding item, then greedy complete.
    # Inspiration: DP or B&B.
    
    # Let's combine:
    # 1. Calculate Global Fractional Bound.
    # 2. For top-k candidates, calculate the "Opportunity Cost" as defined.
    # 3. Use a score that balances immediate value and future potential.
    
    # Let's try the specific strategy:
    # Score_i = v_i / (UB_full - UB_excl_i + epsilon)
    # But if UB_excl_i is close to UB_full, the denominator is small.
    
    # Let's look at the "marginal loss in the fractional upper bound of the REMAINING subproblem if that item is excluded".
    # This phrasing is slightly ambiguous.
    # "Remaining subproblem" usually means after we make a choice.
    # But "if that item is excluded" implies a counterfactual.
    
    # Let's assume:
    # Base State: Items S, Capacity C.
    # Candidate i:
    # If we pick i, new state: S\{i}, Capacity C-w_i.
    # The "opportunity cost" might be the loss in potential value compared to not picking i?
    
    # Let's try a simpler, robust heuristic that uses the fractional bound for pruning/scoring:
    # For each candidate i in top-k density:
    # 1. Calculate UB_after_picking_i = v_i + FractionalBound(S\{i}, C-w_i)
    # 2. This is an upper bound on the total value if we pick i.
    # 3. We want to maximize this UB? No, that's just the bound, not the actual value.
    # 4. However, picking the item with the highest UB_after_picking_i is a common heuristic (Branch and Bound style).
    # 5. The current program uses Greedy Completion. Greedy Completion is a lower bound (feasible solution).
    # 6. Let's combine: Use Fractional Bound for ranking candidates, but verify with Greedy?
    #    Or just use the Fractional Bound of the resulting state as the score.
    
    # Let's test: Score_i = v_i + _fractional_knapsack_upper_bound(S\{i}, C-w_i)
    # This is exactly the bound used in B&B.
    # The item with the highest bound is the most promising.
    
    k = min(10, f_n)
    
    best_candidate = -1
    best_score = -1.0
    
    # Pre-calculate densities for sorting candidates
    # We evaluate top-k by density first, then pick the best among them by the new score.
    
    for rank in range(k):
        idx = density_order[rank]
        
        w_i = f_weights[idx]
        v_i = f_values[idx]
        
        new_capacity = C - w_i
        if new_capacity < 0:
            continue
            
        # Calculate fractional upper bound of the remaining items (excluding idx) with new_capacity
        # We need to create arrays excluding idx for the bound calculation
        # To avoid creating new arrays every time, we can pass a mask or just slice.
        # Since k is small (10), creating small arrays is fine.
        
        mask = np.ones(f_n, dtype=bool)
        mask[idx] = False
        rem_w = f_weights[mask]
        rem_v = f_values[mask]
        
        ub_remaining = _fractional_knapsack_upper_bound(rem_w, rem_v, new_capacity)
        
        score = v_i + ub_remaining
        
        if score > best_score:
            best_score = score
            best_candidate = idx
            
    if best_candidate == -1:
        # Fallback: pick the highest density item
        best_candidate = density_order[0]
        
    orig_idx = feasible_indices[best_candidate]
    return int(orig_idx)
