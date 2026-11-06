import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 58.0 # Safety margin for 60s limit
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as floats for fast evaluation during search
    reciprocals = [1.0 / d for d in denominators]
    
    # 1. Initialize with Greedy Solution (Known to be a strong baseline)
    # Greedy: Add largest available unit fraction that keeps sum < 1
    current_set = set()
    current_sum = 0.0
    
    # We want to minimize deficit (1 - sum), so we want sum as close to 1 as possible from below.
    # Greedy strategy: Sort denominators ascending (largest fractions first).
    # Try adding each. If it fits, add it.
    # This produces a valid subset.
    
    for i in range(n):
        if current_sum + reciprocals[i] < 1.0:
            current_set.add(i)
            current_sum += reciprocals[i]
            
    # Ensure non-empty (Greedy will always pick at least 1/2, so this is safe)
    if not current_set:
        current_set.add(0)
        current_sum = reciprocals[0]
        
    best_set = set(current_set)
    best_sum = current_sum
    best_deficit = 1.0 - best_sum
    
    # 2. Local Search with Add, Swap, Drop moves
    # Objective: Minimize deficit (maximize sum subject to sum < 1)
    # We use a simple hill-climbing with random restarts/annealing-like noise to escape local optima.
    
    # Precompute list of indices for random selection
    all_indices = list(range(n))
    
    # To speed up swap moves, we can precompute differences, but n=47 is small.
    # We'll use a simple loop.
    
    # Parameters for search
    # We have ~58 seconds.
    # Let's run a loop until time runs out.
    
    # Helper to check validity and update best
    def check_and_update(s, set_s):
        nonlocal best_set, best_sum, best_deficit
        if 0 < s < 1.0:
            # Verify with Fraction for the very best candidates to avoid float precision issues
            # We only do exact check if it's better than current best or close to it
            # Since we are maximizing sum, we check if s > best_sum
            if s > best_sum:
                # Exact verification
                exact_sum = Fraction(0)
                for idx in set_s:
                    exact_sum += Fraction(1, denominators[idx])
                if exact_sum < 1:
                    best_sum = float(exact_sum)
                    best_deficit = 1.0 - best_sum
                    best_set = set(set_s)
                    return True
        return False
    
    # Initial check
    check_and_update(current_sum, current_set)
    
    # Local Search Loop
    # We will use a "Best Improvement" or "First Improvement" strategy with random perturbations.
    # Given the small n, we can try all possible single moves (Add, Drop, Swap) at each step.
    # Number of moves:
    # Add: n - |S|
    # Drop: |S|
    # Swap: |S| * (n - |S|)
    # Max swaps ~ 23 * 24 ~ 550. Total moves ~ 1000.
    # 1000 moves per iteration.
    # In 50 seconds, we can do many iterations.
    
    # To avoid getting stuck in a local optimum, we will occasionally accept worse moves (Simulated Annealing style)
    # or just do random restarts from perturbed states.
    
    # Let's stick to a robust Hill Climbing with Random Restarts to keep it simple and deterministic-ish.
    
    def get_deficit(s):
        return 1.0 - s if s < 1.0 else float('inf')
        
    current_deficit = get_deficit(current_sum)
    
    # We will run iterations until timeout
    while time.time() - start_time < timeout:
        improved = False
        
        # Generate all possible single moves
        # 1. Add moves
        for i in all_indices:
            if i not in current_set:
                new_sum = current_sum + reciprocals[i]
                if new_sum < 1.0:
                    new_deficit = 1.0 - new_sum
                    if new_deficit < current_deficit:
                        current_set.add(i)
                        current_sum = new_sum
                        current_deficit = new_deficit
                        improved = True
                        check_and_update(current_sum, current_set)
                        # Break to re-evaluate from new state? Or continue?
                        # Usually in local search, we take the best move or first move.
                        # Taking the best move requires checking all.
                        # Let's collect best move.
                        break # Let's do first improvement for speed, or best?
                        # With n=47, best is safer. Let's refactor to find best move.
        
        # The "break" above is inefficient for finding the global best single step.
        # Let's implement a proper "Best Improvement" local search step.
        
        best_move_type = None
        best_move_i = None
        best_move_j = None # for swap
        best_new_sum = current_sum
        best_new_deficit = current_deficit
        
        # 1. Try Add
        for i in all_indices:
            if i not in current_set:
                new_sum = current_sum + reciprocals[i]
                if new_sum < 1.0:
                    new_def = 1.0 - new_sum
                    if new_def < best_new_deficit:
                        best_new_deficit = new_def
                        best_new_sum = new_sum
                        best_move_type = 'add'
                        best_move_i = i
                        best_move_j = None
                        
        # 2. Try Drop
        for i in list(current_set):
            new_sum = current_sum - reciprocals[i]
            if new_sum > 0: # Must be non-empty? Problem says nonempty.
                # Actually, dropping might make sum 0 if set size is 1.
                # If set size is 1, dropping makes it empty -> invalid.
                if len(current_set) > 1:
                    new_def = 1.0 - new_sum
                    if new_def < best_new_deficit:
                        best_new_deficit = new_def
                        best_new_sum = new_sum
                        best_move_type = 'drop'
                        best_move_i = i
                        best_move_j = None
            
        # 3. Try Swap
        # For each i in set, j not in set
        for i in list(current_set):
            for j in all_indices:
                if j not in current_set:
                    new_sum = current_sum - reciprocals[i] + reciprocals[j]
                    if new_sum < 1.0:
                        new_def = 1.0 - new_sum
                        if new_def < best_new_deficit:
                            best_new_deficit = new_def
                            best_new_sum = new_sum
                            best_move_type = 'swap'
                            best_move_i = i
                            best_move_j = j
                            
        # Apply best move if found
        if best_move_type is not None:
            if best_move_type == 'add':
                current_set.add(best_move_i)
            elif best_move_type == 'drop':
                current_set.remove(best_move_i)
            elif best_move_type == 'swap':
                current_set.remove(best_move_i)
                current_set.add(best_move_j)
                
            current_sum = best_new_sum
            current_deficit = best_new_deficit
            improved = True
            check_and_update(current_sum, current_set)
            
        if not improved:
            # Local optimum reached. Perturb state to escape.
            # Randomly swap 2-3 elements or add/remove random elements.
            if len(current_set) > 2:
                # Pick 2 random elements in set, 2 random not in set, swap them
                in_set = list(current_set)
                not_in_set = [i for i in all_indices if i not in current_set]
                
                if len(in_set) >= 2 and len(not_in_set) >= 2:
                    i1, i2 = random.sample(in_set, 2)
                    j1, j2 = random.sample(not_in_set, 2)
                    
                    # Apply swap
                    current_set.remove(i1)
                    current_set.remove(i2)
                    current_set.add(j1)
                    current_set.add(j2)
                    
                    # Recalculate sum
                    current_sum = sum(reciprocals[i] for i in current_set)
                    
                    # If invalid (sum >= 1), revert or fix
                    if current_sum >= 1.0:
                        # Revert
                        current_set.remove(j1)
                        current_set.remove(j2)
                        current_set.add(i1)
                        current_set.add(i2)
                        current_sum = sum(reciprocals[i] for i in current_set)
                    elif not current_set:
                        # Should not happen
                        current_set.add(0)
                        current_sum = reciprocals[0]
                        
            else:
                # Small set, just random add/remove
                if len(current_set) < 10:
                    # Try adding a random element
                    not_in_set = [i for i in all_indices if i not in current_set]
                    if not_in_set:
                        i = random.choice(not_in_set)
                        new_sum = current_sum + reciprocals[i]
                        if new_sum < 1.0:
                            current_set.add(i)
                            current_sum = new_sum
                else:
                    # Try removing a random element
                    i = random.choice(list(current_set))
                    if len(current_set) > 1:
                        current_set.remove(i)
                        current_sum = sum(reciprocals[i] for i in current_set)
                        
            # After perturbation, check if we found a better global best
            check_and_update(current_sum, current_set)
            
    # Final exact verification of best_set
    # Re-calculate exact sum for best_set to be absolutely sure
    final_exact_sum = Fraction(0)
    for idx in best_set:
        final_exact_sum += Fraction(1, denominators[idx])
        
    if final_exact_sum >= 1 or not best_set:
        # Fallback to greedy result if something went wrong
        # Re-run greedy
        selected, total = [], Fraction()
        for i, d in enumerate(range(2, 49)):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        if not selected:
            selected = [0]
        return selected
        
    return list(best_set)
