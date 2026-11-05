import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def _local_improve(chosen, counts, cap):
    """Try to add more elements or swap to improve the set."""
    chosen_set = set(chosen)
    chosen_np = np.array(sorted(chosen), dtype=np.int64)
    improved = False
    max_attempts = 20000
    
    # Phase 1: Try to add any remaining element
    for x in range(1, cap + 1):
        if x in chosen_set:
            continue
        if _time_left() < 5.0:
            break
        new_sums = np.concatenate((chosen_np + x, [2 * x]))
        if np.all(counts[new_sums] < 2):
            chosen_set.add(x)
            chosen_np = np.sort(np.concatenate((chosen_np, [x])))
            counts[new_sums] += 1
            improved = True
    
    # Phase 2: Try swaps - remove one element, try to add two others
    # We'll do a limited number of swap attempts
    attempts = 0
    while attempts < max_attempts and _time_left() > 10.0:
        attempts += 1
        # Try a random swap: remove a, add b and c
        if len(chosen) < 2:
            break
        
        a = random.choice(list(chosen_set))
        # Remove a temporarily
        chosen_set.remove(a)
        chosen_np_no_a = np.array(sorted(chosen_set), dtype=np.int64)
        counts_no_a = counts.copy()
        
        # Recompute counts without a
        # Actually, we need to subtract the contributions of a
        # Sums involving a: a + x for x in chosen_set, and 2*a
        for x in list(chosen_set):
            counts_no_a[a + x] -= 1
        counts_no_a[2 * a] -= 1
        
        # Now try to add two new elements
        added = []
        for b in range(1, cap + 1):
            if b in chosen_set or b == a:
                continue
            new_sums_b = np.concatenate((chosen_np_no_a + b, [2 * b]))
            if np.all(counts_no_a[new_sums_b] < 2):
                # Check if we can also add c > b
                chosen_set.add(b)
                chosen_np_with_b = np.sort(np.concatenate((chosen_np_no_a, [b])))
                counts_with_b = counts_no_a.copy()
                counts_with_b[new_sums_b] += 1
                
                for c in range(b + 1, cap + 1):
                    if c in chosen_set or c == a:
                        continue
                    new_sums_c = np.concatenate((chosen_np_with_b + c, [2 * c]))
                    if np.all(counts_with_b[new_sums_c] < 2):
                        # Success: swap a for b and c
                        chosen_set.add(c)
                        counts[new_sums_b] += 1
                        counts[new_sums_c] += 1
                        # Update counts for the remaining sums
                        # We need to be careful here
                        # Actually, let's just rebuild counts from scratch for safety
                        # But that's expensive. Let me think...
                        # The counts array was modified in place above.
                        # We added b and c, so counts should reflect that.
                        # But we need to make sure we didn't double-count.
                        # Let's just return and let the caller rebuild if needed.
                        return True
                # If we couldn't add c, undo b
                chosen_set.discard(b)
        
        # If we didn't find a good swap, put a back
        chosen_set.add(a)
        # Restore counts (they should be unchanged since we only used copies)
        # Actually counts was not modified in the loop above except in the success case
        # which returns. So counts is still valid.
    
    return improved

def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 115.0
    
    cap = 1000
    best = []
    best_len = 0
    rng = random.Random(42)
    
    while _time_left() > 5.0:
        # Random order
        order = list(range(1, cap + 1))
        rng.shuffle(order)
        
        # Greedy Build
        chosen = []
        counts = np.zeros(2 * cap + 1, dtype=np.int32)
        chosen_np = np.empty(0, dtype=np.int64)
        
        for x in order:
            # Check feasibility
            if chosen_np.size == 0:
                new_sums = np.array([2 * x], dtype=np.int64)
            else:
                new_sums = np.concatenate((chosen_np + x, [2 * x]))
            
            # Fast check: if any count is 2, adding x makes it 3 (invalid)
            # We need counts[new_sums] < 2
            # Using np.all is fast
            if np.all(counts[new_sums] < 2):
                chosen.append(x)
                chosen_np = np.concatenate((chosen_np, [x]))
                counts[new_sums] += 1
        
        # Local improvement
        if _time_left() > 10.0:
            _local_improve(chosen, counts, cap)
            # Rebuild counts after local improvement if it changed
            # For safety, rebuild counts from the final chosen set
            chosen_sorted = sorted(chosen)
            counts_final = np.zeros(2 * cap + 1, dtype=np.int32)
            for i in range(len(chosen_sorted)):
                for j in range(i, len(chosen_sorted)):
                    s = chosen_sorted[i] + chosen_sorted[j]
                    counts_final[s] += 1
        
        if len(chosen) > best_len:
            best_len = len(chosen)
            best = list(chosen)
    
    return best
