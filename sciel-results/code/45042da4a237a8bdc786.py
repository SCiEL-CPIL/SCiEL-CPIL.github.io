import time
import random

def _has_distinct_subset_sums(arr):
    """Check if all subset sums are distinct."""
    sums = {0}
    for x in arr:
        new_sums = set()
        for s in sums:
            ns = s + x
            if ns in sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Initial valid set: powers of 2
    A = [1 << i for i in range(12)]
    best_A = A[:]
    best_max = max(A)
    
    random.seed(42)
    
    # Pre-calculate some useful info
    # We want to minimize max(A).
    
    # To speed up the search, we can maintain the set of sums for the current best_A?
    # No, because we change one element, the sums change significantly.
    # But 4096 sums is small. Recalculating is fast.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Strategy 1: Try to reduce the maximum element directly
        # Find the index of the maximum element
        max_val = max(best_A)
        max_idx = best_A.index(max_val)
        
        # Try to replace max_val with a smaller value
        # We try values in a random order or descending order?
        # Descending is deterministic and might get stuck.
        # Random sampling in [1, max_val-1] is better for exploration.
        
        # Let's try a few random replacements for the max element
        improved = False
        for _ in range(10): # Try 10 random candidates for the max element
            new_val = random.randint(1, max_val - 1)
            if new_val in best_A:
                continue
            test_A = best_A[:]
            test_A[max_idx] = new_val
            if _has_distinct_subset_sums(test_A):
                best_A = test_A
                best_max = new_val
                improved = True
                break
        
        if improved:
            continue

        # Strategy 2: General local search
        # Pick a random index
        idx = random.randint(0, 11)
        old_val = best_A[idx]
        
        # Generate a new value
        # Bias towards smaller values to try to reduce the max, 
        # but also allow larger values to escape local minima if needed?
        # Since we only accept if max decreases or stays same (and we track best_max),
        # we should focus on values that might lead to a smaller max.
        
        # If we change a non-max element, the max might stay the same or decrease (if it was the max? no).
        # If we change a non-max element, the max is determined by other elements.
        # So we should try to reduce the elements that are large.
        
        # Let's pick the index of the largest element, or a random one.
        # Actually, just picking a random index is fine, but we should prioritize large elements.
        # Let's pick an index proportional to its value? Or just random.
        
        # Let's try to replace the element at idx with a new value.
        # Range for new value: [1, best_max]
        # To encourage reduction, sample from [1, old_val] if old_val is large?
        
        new_val = random.randint(1, best_max)
        
        if new_val == old_val:
            continue
        if new_val in best_A:
            continue
            
        test_A = best_A[:]
        test_A[idx] = new_val
        
        if _has_distinct_subset_sums(test_A):
            new_max = max(test_A)
            if new_max < best_max:
                best_A = test_A
                best_max = new_max
            elif new_max == best_max:
                # Accept if it might lead to better future moves?
                # For simplicity, only accept if strictly better max.
                # Or accept with some probability to explore?
                # Let's accept if it's the same max but the set is "different" to allow further moves.
                # Actually, if max is same, we might be stuck.
                # Let's accept if new_max <= best_max.
                # But we need to avoid cycles.
                # Let's just accept if new_max < best_max.
                # If we get stuck, we can do a "kick" later.
                pass

    return best_A
