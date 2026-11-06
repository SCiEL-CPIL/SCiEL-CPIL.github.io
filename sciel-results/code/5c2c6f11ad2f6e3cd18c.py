"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [1.0 / d for d in denominators]
    
    # Greedy initialization
    def greedy_init():
        selected = set()
        current_sum = 0.0
        # Sort by denominator ascending (larger fractions first)
        indices_sorted = sorted(range(n), key=lambda i: denominators[i])
        for i in indices_sorted:
            if current_sum + recips[i] < 1.0:
                selected.add(i)
                current_sum += recips[i]
        return selected, current_sum

    best_selected, best_sum_float = greedy_init()
    best_frac_sum = sum(Fraction(1, denominators[i]) for i in best_selected)
    
    # Verify best is strictly below 1
    if best_frac_sum >= 1:
        # Fallback, though greedy should be safe
        best_selected.remove(min(best_selected, key=lambda i: denominators[i]))
        best_frac_sum = sum(Fraction(1, denominators[i]) for i in best_selected)

    # Local Search / Simulated Annealing
    # We will try to improve the sum by adding, removing, or swapping elements.
    
    current_selected, current_sum = greedy_init()
    
    # To escape local optima, we use a simple SA-like approach on swaps and additions.
    # Since adding is always good if possible, we prioritize it.
    # Swaps allow us to replace a small fraction with a larger one if it fits.
    
    temperature = 1e-6  # Small temperature for "almost" greedy, allowing some bad moves
    cooling_rate = 0.999
    
    # Precompute unselected list for fast iteration?
    # We'll just iterate over all n each step, n=47, which is small.
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add the best possible element
        best_add_idx = -1
        best_add_gain = 0.0
        for j in range(n):
            if j not in current_selected:
                gain = recips[j]
                if current_sum + gain < 1.0:
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_idx = j
        
        if best_add_idx != -1:
            current_selected.add(best_add_idx)
            current_sum += best_add_gain
            improved = True
            if current_sum > best_sum_float:
                best_sum_float = current_sum
                best_selected = set(current_selected)
            continue
        
        # 2. Try to swap
        # Find the best improving swap
        best_swap_i = -1
        best_swap_j = -1
        best_swap_gain = 0.0
        
        # Also track best "neutral" or slightly negative swap for SA
        # But let's stick to improving swaps first for speed
        
        for i in current_selected:
            for j in range(n):
                if j not in current_selected:
                    new_sum = current_sum - recips[i] + recips[j]
                    if new_sum < 1.0:
                        gain = new_sum - current_sum
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_i = i
                            best_swap_j = j
        
        if best_swap_i != -1:
            current_selected.remove(best_swap_i)
            current_selected.add(best_swap_j)
            current_sum += best_swap_gain
            improved = True
            if current_sum > best_sum_float:
                best_sum_float = current_sum
                best_selected = set(current_selected)
            continue
            
        # 3. If no improving move, try a random perturbation (SA step)
        # Pick a random i in selected and j not in selected.
        # If it fits, accept with probability exp(-delta/T) if delta < 0 (worsening)
        # Or just accept any valid move to escape local optima?
        # Let's try a random valid swap.
        
        if len(current_selected) > 0:
            i = random.choice(list(current_selected))
            unselected = [j for j in range(n) if j not in current_selected]
            if unselected:
                j = random.choice(unselected)
                new_sum = current_sum - recips[i] + recips[j]
                if new_sum < 1.0:
                    delta = new_sum - current_sum
                    # Accept if improving, or with prob exp(-delta/T) if worsening
                    if delta >= 0 or random.random() < pow(2.718281828, -delta / temperature):
                        current_selected.remove(i)
                        current_selected.add(j)
                        current_sum = new_sum
                        if current_sum > best_sum_float:
                            best_sum_float = current_sum
                            best_selected = set(current_selected)
        
        temperature *= cooling_rate
        if temperature < 1e-12:
            temperature = 1e-12 # Floor

    # Final verification with Fractions
    final_frac_sum = sum(Fraction(1, denominators[i]) for i in best_selected)
    if final_frac_sum < 1:
        return list(best_selected)
    
    # Fallback: Return greedy if something went wrong
    return list(best_selected)

# EVOLVE-BLOCK-END
