"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from fractions import Fraction
    from itertools import combinations

    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit

    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute weights as Fractions for exactness
    weights = [Fraction(1, d) for d in denominators]
    
    # We want to maximize sum of weights subject to sum < 1.
    # Strategy:
    # 1. Fast greedy initialization using floats.
    # 2. Exact local search (swaps) to refine the solution.
    # 3. Randomized restarts with different greedy orders to escape local optima.
    # 4. Final verification with Fractions.

    best_sum = Fraction(0)
    best_indices = []

    def evaluate_and_update(indices):
        nonlocal best_sum, best_indices
        if not indices:
            return
        s = sum(weights[i] for i in indices)
        if s < 1 and s > best_sum:
            best_sum = s
            best_indices = list(indices)

    # --- Phase 1: Greedy Initialization ---
    # Try multiple greedy orders to get good starting points.
    
    # Order 1: Standard ascending denominator (largest weight first)
    # This is the "natural" greedy.
    curr_sum = Fraction(0)
    curr_indices = []
    for i in range(n):
        if curr_sum + weights[i] < 1:
            curr_sum += weights[i]
            curr_indices.append(i)
    evaluate_and_update(curr_indices)

    # Order 2: Descending denominator (smallest weight first)
    # This packs many small items.
    curr_sum = Fraction(0)
    curr_indices = []
    for i in range(n - 1, -1, -1):
        if curr_sum + weights[i] < 1:
            curr_sum += weights[i]
            curr_indices.append(i)
    evaluate_and_update(curr_indices)

    # Order 3: Randomized Greedy with Local Search (Hill Climbing)
    # We will perform several random restarts.
    
    random.seed(42)
    
    def local_search(indices_set):
        """
        Perform local search on a set of indices.
        Moves:
        1. Add an unselected item if it fits.
        2. Swap a selected item with an unselected item if the swap improves the sum and keeps it < 1.
        3. Remove a selected item if it allows adding two or more unselected items that result in a higher sum? (Complex, skip for now)
        
        Returns the improved set and sum.
        """
        curr_sum = sum(weights[i] for i in indices_set)
        improved = True
        while improved:
            improved = False
            curr_indices_list = list(indices_set)
            
            # Try adding
            for i in range(n):
                if i not in indices_set:
                    new_sum = curr_sum + weights[i]
                    if new_sum < 1:
                        indices_set.add(i)
                        curr_sum = new_sum
                        improved = True
                        # Restart loop to check for more additions?
                        # Actually, if we add one, we might be able to add another.
                        # But breaking and restarting the while loop handles this.
                        break
            
            # Try swapping
            if not improved:
                for i in list(indices_set):
                    for j in range(n):
                        if j in indices_set:
                            continue
                        # We want to increase the sum, so weights[j] should be > weights[i]
                        # But wait, if we swap a large weight for a smaller one, sum decreases.
                        # We want to MAXIMIZE sum. So we want to swap a small weight (large denom) 
                        # with a larger weight (small denom) if it fits.
                        if weights[j] <= weights[i]:
                            continue
                        
                        new_sum = curr_sum - weights[i] + weights[j]
                        if new_sum < 1:
                            # This swap increases sum and is valid.
                            indices_set.remove(i)
                            indices_set.add(j)
                            curr_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
                        
        return indices_set, curr_sum

    # Run multiple randomized restarts
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate a random order
        order = list(range(n))
        random.shuffle(order)
        
        curr_sum = Fraction(0)
        curr_indices_set = set()
        
        # Greedy fill based on this random order
        for i in order:
            if curr_sum + weights[i] < 1:
                curr_sum += weights[i]
                curr_indices_set.add(i)
        
        # Local Search
        curr_indices_set, curr_sum = local_search(curr_indices_set)
        
        evaluate_and_update(curr_indices_set)
        
        # Early exit if we found a very good solution?
        # The deficit is 1 - best_sum.
        # If deficit is very small, we might stop.
        # But let's use the time budget.

    # --- Phase 2: Exhaustive Search on Small Subsets (Optional/Refinement) ---
    # If time permits, we can try to improve the best solution found by 
    # removing one item and trying to fill the gap with a combination of others.
    # This is effectively a more expensive local search.
    
    # Let's try a specific refinement on the best found solution.
    if time.time() < deadline - 5 and best_indices:
        best_set = set(best_indices)
        best_sum_val = best_sum
        
        # Try removing each item in the best set
        for i in list(best_set):
            if time.time() > deadline - 2:
                break
                
            # Remove i
            temp_set = best_set.copy()
            temp_set.remove(i)
            temp_sum = best_sum_val - weights[i]
            
            # Try to add items from the remaining pool (not in temp_set)
            # Sort remaining items by weight descending to fill gap efficiently
            remaining_items = [j for j in range(n) if j not in temp_set]
            remaining_items.sort(key=lambda x: -weights[x])
            
            # Greedy fill the gap
            for j in remaining_items:
                if temp_sum + weights[j] < 1:
                    temp_sum += weights[j]
                    temp_set.add(j)
                
            # Local search on this new set
            temp_set, temp_sum = local_search(temp_set)
            
            if temp_sum > best_sum_val:
                best_sum_val = temp_sum
                best_indices = list(temp_set)
                best_set = set(best_indices)
                # Update global best
                best_sum = best_sum_val

    # Final Verification
    if not best_indices:
        # Fallback to simple greedy if something went wrong
        curr_sum = Fraction(0)
        for i in range(n):
            if curr_sum + weights[i] < 1:
                curr_sum += weights[i]
                best_indices.append(i)
        best_sum = curr_sum

    # Ensure strict inequality and distinct indices
    final_sum = sum(weights[i] for i in best_indices)
    if final_sum >= 1:
        # Should not happen with our logic, but just in case
        # Remove items until valid
        for i in best_indices[:]:
            if final_sum - weights[i] < 1:
                best_indices.remove(i)
                final_sum -= weights[i]
                break
        if not best_indices:
            best_indices = [0] # 1/2 is always valid
            final_sum = weights[0]

    return best_indices
# EVOLVE-BLOCK-END
