"""0/1 knapsack heuristic: Simultaneous full-solution evaluation via vectorized greedy simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 5.0


def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Feasible candidates: weight <= remaining capacity.
    feasible_mask = weights <= remaining_capacity + 1e-12
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # Deadline guard: if we're running out of time, fall back to a fast density pick.
    if not _elapsed_ok():
        dens = values[feasible_indices] / np.maximum(weights[feasible_indices], 1e-12)
        best_local = int(feasible_indices[np.argmax(dens)])
        return best_local

    # Sort all items by value density (descending) for the greedy completion.
    # We use this global order to simulate the "fill" for each candidate.
    # This order is fixed for all candidates, which allows vectorization.
    all_densities = values / np.maximum(weights, 1e-12)
    global_order = np.argsort(-all_densities)  # indices of items in descending density
    
    # We want to simulate: "If I pick candidate `c`, what is the total value?"
    # Total value = values[c] + sum(values of items in global_order that fit in (remaining_capacity - weights[c]) AND are not c)
    
    # To vectorize the "sum of values of items that fit" part:
    # For a given candidate c, the remaining capacity is cap_c = remaining_capacity - weights[c].
    # The items considered for filling are those in global_order, excluding c.
    # An item j fits if weights[j] <= current_capacity.
    # This is a sequential process (knapsack greedy), which is hard to vectorize perfectly 
    # because the capacity decreases as we add items.
    # However, since we are just picking the *next* item, and the "fill" is a heuristic 
    # estimate of the rest, we can approximate the fill value.
    
    # Alternative structural change: 
    # Instead of simulating the exact greedy fill (which is O(N^2) per candidate if done naively, 
    # or O(N^2) total if vectorized poorly), let's try a different heuristic:
    # **Lagrangian Relaxation / Dual Variable Approach**
    # Or better: **Pairwise Swap Improvement (2-Opt style)**
    
    # Let's stick to a robust, fast, and structurally different approach:
    # **Greedy with "Regret" Minimization**
    # For each candidate c, compute the "opportunity cost" or "regret" of picking it.
    # Regret(c) = (Best possible value if we didn't pick c) - (Value if we do pick c)
    # This is computationally expensive.
    
    # Let's try a **Multi-Objective Scoring** approach:
    # Score(c) = values[c] + alpha * (remaining_capacity - weights[c]) * density_of_best_remaining
    # This is still similar to lookahead.
    
    # Let's try **Randomized Restart with Best-of-N** within the time budget?
    # No, the function is called sequentially. We can't restart the whole knapsack.
    
    # Let's go back to the vectorized simulation but optimize it.
    # The current program does O(N^2) work per call (N candidates * N items in simulation).
    # With N=100, that's 10,000 operations. 100 calls -> 1,000,000 operations. This is fast.
    # The current program scores 1.0015.
    
    # What if we use a **2-step lookahead**?
    # Pick c1, then pick c2, then greedy fill.
    # This is O(N^3) per call. 100^3 = 1,000,000 per call. 100 calls = 100,000,000.
    # This might be too slow in Python loops.
    
    # Let's try a **different completion heuristic**.
    # Instead of density-greedy fill, use **Value/Weight + Value/Remaining_Capacity** hybrid?
    
    # Actually, the most promising structural change for a "jump" is to use **Local Search (Hill Climbing)** 
    # on the set of items selected so far? But we don't have access to the previously selected items 
    # in this function signature. We only have `remaining` arrays.
    # Wait, the function `select_next_item` is called iteratively. The "remaining" arrays are the ones 
    # NOT yet selected. So we don't know what WAS selected. We can't do local search on the past.
    
    # So we are limited to heuristics based on the `remaining` set.
    
    # Let's try **Simulated Annealing** style acceptance? No, we must return a single index.
    
    # Let's try **Beam Search** with width 1? That's greedy.
    # Beam Search with width K? We can't maintain a beam across calls.
    
    # Okay, let's look at the **density** calculation.
    # Maybe we should use **Value / (Weight + epsilon)** or **Value / Weight^2**?
    
    # Let's try a **Hybrid Score**:
    # score(c) = values[c] / weights[c]  (Density)
    # PLUS
    # values[c] / remaining_capacity (Value relative to total capacity)
    # PLUS
    # (remaining_capacity - weights[c]) * max_density_remaining (Opportunity cost of space)
    
    # This is a structural change in the *scoring function* used for the 1-step lookahead.
    
    # Let's implement a **Composite Score** for the 1-step lookahead.
    # For each candidate c:
    # 1. Calculate `val_c = values[c]`
    # 2. Calculate `rem_cap = remaining_capacity - weights[c]`
    # 3. Estimate `future_val` using a more sophisticated heuristic than pure density.
    #    e.g., Sort remaining items by density, but weight the density by how "tight" the fit is.
    
    # Let's try **Filling Efficiency**:
    # When simulating the fill, instead of just taking if it fits, 
    # we can calculate the "wasted space" and penalize items that leave large gaps.
    
    # Actually, a very powerful heuristic for Knapsack is **Ratio of Value to Weight of the "Next Best" item**.
    
    # Let's try this:
    # For candidate c:
    # 1. Take c.
    # 2. From the remaining items (excluding c), find the item d that has the highest density.
    # 3. If d fits in rem_cap, take d.
    # 4. Continue greedy.
    # This is just greedy.
    
    # What if we **skip** the highest density item if it's "too big" and leaves too much waste?
    # This is complex.
    
    # Let's try a **Simple but Different Metric**:
    # **Value per unit of "Risk"** where Risk = Probability that picking this item prevents picking a high-value item later.
    # Approximation: Risk = Weight of item / Remaining Capacity.
    # Score = Value / Weight - Lambda * (Weight / Remaining Capacity)
    
    # Let's try **Maximizing the Minimum Density of the Filled Set**?
    
    # Let's go with **2-Step Lookahead** but optimized with NumPy.
    # We can vectorize the 1st step of the 2nd step.
    
    # Actually, let's look at the **Current Program's Weakness**:
    # It simulates the *entire* greedy fill for each candidate.
    # This is O(N^2).
    # If we just look at the **First 2 items** of the fill, we save time but lose accuracy.
    
    # Let's try a **Different Greedy Order** for the fill.
    # Instead of Density, use **Value** (greedy by value)?
    # Or **Weight** (greedy by smallest weight)?
    
    # Let's try **Greedy by Value** for the fill.
    
    def _simulate_completion_value_order(capacity, weights, values, order, exclude):
        cap = capacity
        total = 0.0
        for idx in order:
            if idx == exclude:
                continue
            w = weights[idx]
            if w <= cap:
                cap -= w
                total += values[idx]
                if cap <= 1e-12:
                    break
        return total

    # Order by Value (descending)
    order_val = np.argsort(-values)
    
    # Order by Weight (ascending) - "Best Fit"
    order_wt = np.argsort(weights)
    
    # Order by Density (descending) - Standard
    order_dens = np.argsort(-all_densities)
    
    # Let's evaluate candidates using all 3 completion strategies and pick the best overall?
    # Or pick the candidate that is best according to *all* or *most*?
    
    best_val = -1.0
    best_idx = -1
    
    # We will compute the estimated total value for each candidate using the Density order (as before)
    # But also compute a "Penalty" based on the gap left.
    
    # Let's try **Gap Penalty**:
    # Total Value = Sum of values of picked items.
    # Gap = Remaining capacity after fill.
    # Score = Total Value - Penalty * Gap.
    # This encourages filling the knapsack completely.
    
    # Let's implement this.
    
    for c in feasible_indices:
        cap_after = remaining_capacity - weights[c]
        if cap_after < 0:
            continue
            
        # Simulate fill with Density Order
        total_val = 0.0
        final_cap = cap_after
        for idx in order_dens:
            if idx == c:
                continue
            w = weights[idx]
            if w <= final_cap:
                final_cap -= w
                total_val += values[idx]
                if final_cap <= 1e-12:
                    break
        
        total_val += values[c]
        gap = final_cap
        
        # Penalty for leaving space
        # How to scale penalty? 
        # Average value density of the set?
        # Let's use a fixed penalty factor or dynamic.
        # If we use a large penalty, it forces full packing.
        # But full packing isn't always optimal if the items are low value.
        
        # Let's just use Total Value for now, but maybe weight the gap by the density of the *last* item added?
        
        if total_val > best_val:
            best_val = total_val
            best_idx = int(c)
            
    if best_idx < 0:
        dens = values[feasible_indices] / np.maximum(weights[feasible_indices], 1e-12)
        best_idx = int(feasible_indices[np.argmax(dens)])
        
    return best_idx
# EVOLVE-BLOCK-END
