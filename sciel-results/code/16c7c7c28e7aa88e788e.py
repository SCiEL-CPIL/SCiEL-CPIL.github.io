import time
import random

def _is_valid_bitset(s):
    """
    Check if the set s has cardinality-separated subset sums using bitsets.
    Returns True if valid, False otherwise.
    """
    n = len(s)
    if n < 2:
        return True
    
    # sums[k] is a bitset where bit i is 1 if sum i is achievable by a subset of size k
    # Max sum is n * 256.
    
    sums = [0] * (n + 1)
    sums[0] = 1 # bit 0 is 1 (empty sum)
    
    for x in s:
        # Update from high k to low k (0/1 knapsack style)
        for k in range(n, 0, -1):
            sums[k] |= (sums[k-1] << x)
            
    # Check for collisions between different cardinalities
    # For all k != j, sums[k] & sums[j] must be 0.
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums[k] & sums[j]:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin for 90s limit
    
    max_val = 256
    
    # Strategy 1: Try the known optimal arithmetic progression size 31
    # This is a very strong baseline.
    # For AP [x, x+1, ..., x+k-1], condition is x > max_{j} j(k-1-j).
    # For k=31, max j(30-j) is at j=15, value 15*15=225.
    # So x must be >= 226.
    # Set is [226, ..., 256]. Size 31.
    # This is always valid.
    baseline_size = 31
    baseline_set = list(range(226, 257))
    
    # Verify baseline just in case (should be fast)
    if _is_valid_bitset(baseline_set):
        best_set = baseline_set
    else:
        # Fallback if something weird is going on (shouldn't happen)
        best_set = [1]
        
    # Strategy 2: Try to find a set larger than 31 using greedy construction
    # We use a fast incremental validity check.
    # Instead of rebuilding the whole bitset DP for each candidate,
    # we can maintain the current bitsets and check if adding x causes collisions.
    
    def _try_greedy(order, deadline):
        """
        Attempts to build a valid set by adding elements in the given order.
        Uses incremental bitset DP for fast validity checks.
        """
        max_size = 60
        masks = [0] * (max_size + 1)
        masks[0] = 1  # Sum 0 with 0 elements
        
        current_set = []
        
        for x in order:
            if len(current_set) >= max_size:
                break
            if time.time() > deadline:
                break
            
            # Check if adding x is valid
            # new_sums for cardinality k+1 come from masks[k] << x
            # These new bits in masks[k+1] must not collide with masks[j] for j != k+1
            # Note: masks[k+1] already contains some sums. Adding more is fine (same cardinality).
            # But the NEW bits (masks[k+1] ^ (masks[k+1] | (masks[k] << x))) must not intersect masks[j] for j != k+1.
            # Actually, simpler:
            # Let new_mask = masks[k] << x
            # For each j != k+1, if new_mask & masks[j] != 0, then collision.
            
            valid = True
            current_n = len(current_set)
            
            # We iterate k from current_n down to 0?
            # No, we just need to check if ANY new sum collides.
            # For each source cardinality k (0..current_n), the new sums are masks[k] << x.
            # These will be added to cardinality k+1.
            # They must not collide with existing sums of cardinality j where j != k+1.
            
            for k in range(current_n + 1):
                if masks[k] == 0:
                    continue
                
                new_sums = masks[k] << x
                target_k = k + 1
                
                # Check collision with all other cardinalities j != target_k
                # j ranges from 0 to current_n (existing cardinalities)
                # Note: target_k is at most current_n + 1.
                # If target_k == current_n + 1, it's a new cardinality, so no existing masks for it (except what we're building).
                # But masks is a fixed size array. masks[target_k] might have old data if we reused? No, we append to set, so indices grow.
                # However, we are reusing the `masks` array.
                # If we are adding the (current_n+1)-th element, target_k goes up to current_n+1.
                # masks[current_n+1] is currently 0 (unless we are reusing memory from a previous run, but we re-init masks).
                
                for j in range(current_n + 1):
                    if j == target_k:
                        continue
                    if masks[j] != 0 and (new_sums & masks[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Update masks
                # Must update in reverse order to avoid using x multiple times
                for k in range(current_n + 1, 0, -1):
                    masks[k] |= (masks[k-1] << x)
                current_set.append(x)
        
        return current_set

    # Try different greedy orders
    # 1. Descending order (largest first)
    order_desc = list(range(max_val, 0, -1))
    res = _try_greedy(order_desc, deadline)
    if len(res) > len(best_set):
        best_set = res
        
    # 2. Ascending order
    order_asc = list(range(1, max_val + 1))
    if time.time() < deadline:
        res = _try_greedy(order_asc, deadline)
        if len(res) > len(best_set):
            best_set = res
            
    # 3. Randomized orders
    # Since the bitset check is very fast, we can try many.
    attempts = 0
    while time.time() < deadline - 5:
        random_order = list(range(1, max_val + 1))
        random.shuffle(random_order)
        res = _try_greedy(random_order, deadline)
        if len(res) > len(best_set):
            best_set = res
            # If we reached max size, we can stop
            if len(best_set) >= 60:
                break
        attempts += 1
        # Limit attempts to avoid infinite loop if something is wrong, though time check handles it
        if attempts > 5000:
            break

    # Final validation of the best set found
    # It's safer to run the full exact check on the final result
    # In case the incremental logic had a subtle bug or edge case
    if not _is_valid_bitset(best_set):
        # Fallback to the known valid AP
        best_set = list(range(226, 257))
        
    # Ensure types are correct
    return list(best_set)
