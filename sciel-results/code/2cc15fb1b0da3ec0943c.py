import math
import time
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)
ONE = Fraction(1, 1)

# Precompute float values for fast arithmetic
FLOAT_VALS = [1.0 / d for d in DENOMS]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s buffer
    
    # 1. Greedy Initialization
    # Sort by value descending (smaller denominator -> larger value)
    # We want to get close to 1 from below.
    # A simple greedy: add the largest possible value that keeps sum < 1.
    # However, this often gets stuck. 
    # Better: Start with all small denominators? No, sum is too big.
    # Start with empty set, add largest available that fits?
    # Or start with a random set?
    
    # Let's try a few greedy strategies.
    # Strategy 1: Greedy by largest value that fits.
    selected = set()
    current_sum = 0.0
    available = list(range(N))
    # Sort available by value descending
    available.sort(key=lambda i: FLOAT_VALS[i], reverse=True)
    
    for i in available:
        if current_sum + FLOAT_VALS[i] < 1.0:
            selected.add(i)
            current_sum += FLOAT_VALS[i]
            
    # This greedy might be suboptimal. Let's also try starting with the smallest denominators
    # and removing the largest ones that cause overflow?
    # Actually, the total sum is ~6.4. We need to remove ~5.4.
    # Let's try a "remove from full" strategy.
    # Start with all, sum is too high. Remove largest values until sum < 1?
    # This will leave us with a set of large denominators (small values).
    # Sum of 1/48...1/2 is ~6.4.
    # If we remove 1/2, 1/3, 1/4... we remove big chunks.
    
    selected2 = set(range(N))
    current_sum2 = sum(FLOAT_VALS)
    # Sort indices by value descending (largest first)
    order = sorted(range(N), key=lambda i: FLOAT_VALS[i], reverse=True)
    for i in order:
        if current_sum2 < 1.0:
            break
        selected2.discard(i)
        current_sum2 -= FLOAT_VALS[i]
        
    # We have two candidates. Pick the one with smaller deficit (closer to 1).
    def deficit(s):
        return 1.0 - s
        
    if deficit(current_sum) < deficit(current_sum2):
        best_selected = selected
        best_sum = current_sum
    else:
        best_selected = selected2
        best_sum = current_sum2

    # 2. Local Refinement
    # We will perform local search: Add, Remove, Swap.
    # Use float for speed.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Check Add
        best_add_idx = -1
        best_add_deficit = 1.0 - best_sum
        for j in range(N):
            if j not in best_selected:
                new_sum = best_sum + FLOAT_VALS[j]
                if new_sum < 1.0:
                    new_def = 1.0 - new_sum
                    if new_def < best_add_deficit:
                        best_add_deficit = new_def
                        best_add_idx = j
                        
        if best_add_idx != -1:
            best_selected.add(best_add_idx)
            best_sum += FLOAT_VALS[best_add_idx]
            improved = True
            continue
            
        # Check Remove
        # Only useful if we are above 1, but we enforce < 1. 
        # However, removing an item might allow adding a different item later? 
        # No, single move. If we are below 1, removing increases deficit. 
        # So Remove is never beneficial in a single step if we are strictly below 1.
        # UNLESS we consider that removing a large item might allow us to add TWO smaller items? 
        # That's a 2-swap (remove 1, add 2).
        
        # Check Swap (Remove one, Add one)
        best_swap_a = -1
        best_swap_b = -1
        best_swap_deficit = 1.0 - best_sum
        
        # Iterate over items to remove
        for a in best_selected:
            base_sum = best_sum - FLOAT_VALS[a]
            # Iterate over items to add
            for b in range(N):
                if b in best_selected:
                    continue
                new_sum = base_sum + FLOAT_VALS[b]
                if new_sum < 1.0:
                    new_def = 1.0 - new_sum
                    if new_def < best_swap_deficit:
                        best_swap_deficit = new_def
                        best_swap_a = a
                        best_swap_b = b
                        
        if best_swap_a != -1:
            best_selected.discard(best_swap_a)
            best_selected.add(best_swap_b)
            best_sum = best_sum - FLOAT_VALS[best_swap_a] + FLOAT_VALS[best_swap_b]
            improved = True
            continue
            
        # Check Double Swap (Remove two, Add two) - might be too slow?
        # N=48. Pairs of removes: ~1100. Pairs of adds: ~1100.
        # 1100 * 1100 = 1.2M operations. Might be okay if done once or few times.
        # But let's stick to single swaps first as they are O(N^2) ~ 2300 ops.
        
        # If no single move improves, try a random restart or perturbation?
        # Or accept local minimum.
        
        # To escape local minima, we can try a "2-opt" like move:
        # Remove 1, Add 2.
        # This is more expensive.
        
        # Let's try one round of 1-remove-2-add if stuck.
        best_12_a = -1
        best_12_b1 = -1
        best_12_b2 = -1
        best_12_deficit = 1.0 - best_sum
        
        for a in best_selected:
            base_sum = best_sum - FLOAT_VALS[a]
            # Find best pair b1, b2 to add
            # We want to maximize the sum added (minimize deficit) while staying < 1.
            # This is a subset sum problem for the add set of size 2.
            # We can just iterate all pairs.
            for b1 in range(N):
                if b1 in best_selected or b1 == a: # b1 != a is implied if a is removed? 
                    # Actually b1 and b2 must not be in best_selected.
                    # And b1 != b2.
                    continue
                v1 = FLOAT_VALS[b1]
                if base_sum + v1 >= 1.0:
                    continue
                for b2 in range(b1 + 1, N):
                    if b2 in best_selected:
                        continue
                    v2 = FLOAT_VALS[b2]
                    new_sum = base_sum + v1 + v2
                    if new_sum < 1.0:
                        new_def = 1.0 - new_sum
                        if new_def < best_12_deficit:
                            best_12_deficit = new_def
                            best_12_a = a
                            best_12_b1 = b1
                            best_12_b2 = b2
                            
        if best_12_a != -1:
            best_selected.discard(best_12_a)
            best_selected.add(best_12_b1)
            best_selected.add(best_12_b2)
            best_sum = best_sum - FLOAT_VALS[best_12_a] + FLOAT_VALS[best_12_b1] + FLOAT_VALS[best_12_b2]
            improved = True
            continue
            
    # Verify with Fraction
    exact_sum = sum(Fraction(1, DENOMS[i]) for i in best_selected)
    if exact_sum >= ONE:
        # Fallback: remove largest until < 1
        # This shouldn't happen if float logic was correct, but just in case.
        # Sort by value desc
        items = sorted(best_selected, key=lambda i: FLOAT_VALS[i], reverse=True)
        for i in items:
            if exact_sum < ONE:
                break
            best_selected.discard(i)
            exact_sum -= Fraction(1, DENOMS[i])
            
    return sorted(best_selected)

if __name__ == "__main__":
    idx = construct()
    s = sum(Fraction(1, DENOMS[i]) for i in idx)
    print("indices:", idx)
    print("sum:", s)
    print("deficit:", Fraction(1, 1) - s)
