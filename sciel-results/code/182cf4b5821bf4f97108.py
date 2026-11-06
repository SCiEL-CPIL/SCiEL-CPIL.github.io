import numpy as np
import time

def construct():
    """
    Finds a subset A of [0, 128] such that every integer in [128, 256] is a sum of two elements of A.
    Minimizes the largest representation multiplicity in that target interval.
    """
    n = 128
    target_min = 128
    target_max = 256
    
    # Time limit guard
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for 60s limit

    # A is a set of integers
    A = set()
    # counts[s] = number of pairs {u, v} in A such that u + v = s
    # We only care about sums up to 256
    counts = np.zeros(257, dtype=np.int32)
    
    def update_counts_add(x):
        """Add x to A and update counts."""
        A.add(x)
        for u in A:
            if u == x:
                # Pair {x, x}
                s = 2 * x
                if s <= 256:
                    counts[s] += 1
            else:
                s = u + x
                if s <= 256:
                    counts[s] += 1
        # Note: The loop above adds pairs {u, x} for all u in A (including x).
        # Since A is a set, u iterates over all current members.
        # If u < x, we add {u, x}. If u > x, we add {u, x}. If u == x, we add {x, x}.
        # This correctly counts unordered pairs.
        
    def update_counts_remove(x):
        """Remove x from A and update counts."""
        A.remove(x)
        for u in A:
            s = u + x
            if s <= 256:
                counts[s] -= 1
        # Don't forget the pair {x, x} if it was counted
        # Wait, the loop `for u in A` iterates over A AFTER removal? No, A is modified in-place.
        # So `for u in A` sees the set without x.
        # The pairs involving x are {x, x} and {x, u} for u in A_new.
        # So we need to decrement counts for {x, x} and {x, u} for all u in A_new.
        # My previous logic in add was:
        # for u in A (which includes x):
        #   if u == x: count 2x
        #   else: count u+x
        # This counts each pair exactly once.
        # In remove:
        # We need to subtract the contributions of x.
        # Pairs removed: {x, x} and {x, u} for all u in A \ {x}.
        # So:
        s_xx = 2 * x
        if s_xx <= 256:
            counts[s_xx] -= 1
        for u in A: # A is now without x
            s = u + x
            if s <= 256:
                counts[s] -= 1

    def is_valid():
        """Check if all sums in [128, 256] are covered."""
        return np.all(counts[target_min:target_max+1] > 0)

    def get_peak():
        return np.max(counts[target_min:target_max+1])

    # --- Phase 1: Greedy Construction ---
    # Start with empty set
    # While not valid, add the element that covers the most uncovered sums
    
    while not is_valid():
        if time.time() - start_time > time_limit:
            break
            
        best_x = -1
        best_gain = -1
        
        # We can limit search space if A is getting large, but n=128 is small.
        for x in range(n + 1):
            if x in A:
                continue
                
            # Calculate gain: how many currently uncovered sums will be covered by adding x?
            # A sum s is covered by x if s-x is in A (and s-x <= x for unordered? No, just s-x in A).
            # Actually, adding x creates pairs {x, u} for u in A and {x, x}.
            # These pairs produce sums s = x + u and s = 2x.
            # We want to maximize the number of s in [128, 256] such that counts[s] was 0 and becomes > 0.
            
            gain = 0
            # Check pair {x, x}
            s_xx = 2 * x
            if target_min <= s_xx <= target_max and counts[s_xx] == 0:
                gain += 1
            
            # Check pairs {x, u} for u in A
            for u in A:
                s = x + u
                if target_min <= s <= target_max and counts[s] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
                
        if best_x == -1 or best_gain == 0:
            # No single element can improve coverage?
            # This shouldn't happen if we start from empty and target is reachable.
            # But if stuck, break.
            break
            
        update_counts_add(best_x)

    # Ensure we have a valid set if we broke early due to time?
    # If not valid, we must return something valid.
    # Fallback: Use the seed solution if we failed to construct.
    if not is_valid():
        # Seed solution: A = [64, 128]
        A = set(range(64, 129))
        counts = np.zeros(257, dtype=np.int32)
        for x in A:
            for u in A:
                if x <= u:
                    s = x + u
                    if s <= 256:
                        counts[s] += 1
        # This is slow but safe fallback.
        # Actually, let's just construct the seed counts properly.
        # For A=[64..128], peak is ~65.
        # We will return this if greedy failed.
        pass

    # --- Phase 2: Local Search (Minimize Peak) ---
    # Try to remove elements
    # Try to swap elements
    
    current_peak = get_peak()
    no_improvement_count = 0
    max_no_improvement = 10000 # Prevent infinite loop if stuck in local minima with no swaps
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try removals
        # Iterate over a copy of A to allow modification
        for x in list(A):
            if time.time() - start_time > time_limit:
                break
                
            # Check if removing x keeps validity
            # We need to know if any sum s becomes 0.
            # Sums affected: 2x and x+u for u in A\{x}.
            # Let's simulate removal.
            
            # Temporarily remove x
            update_counts_remove(x)
            
            if is_valid():
                # Check if peak decreased
                new_peak = get_peak()
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                    no_improvement_count = 0
                else:
                    # Peak didn't decrease, but we removed an element.
                    # This might help future swaps or just keep the set smaller.
                    # However, if peak is the same, we might want to keep it to allow further removals?
                    # Actually, removing an element never increases peak.
                    # So we should always accept if valid, even if peak is same, to simplify the set.
                    # But wait, if we remove x, we might block a better swap later?
                    # Unlikely. Smaller sets are generally easier to manage.
                    # Let's accept if valid.
                    improved = True
                    no_improvement_count = 0
            else:
                # Restore x
                update_counts_add(x)
        
        # 2. Try swaps
        if not improved and len(A) > 0:
            # For each x in A, try swapping with y not in A
            # This is O(|A| * n * |A|) which is ~ 20 * 128 * 20 = 50k ops per full pass.
            # 50k is fast.
            
            best_swap = None
            best_new_peak = current_peak
            
            # Optimization: Only try swaps that might reduce peak.
            # A swap (x, y) changes counts.
            # Removed pairs: {x, x}, {x, u} for u in A\{x}
            # Added pairs: {y, y}, {y, u} for u in A\{x}
            
            for x in list(A):
                if time.time() - start_time > time_limit:
                    break
                
                # Calculate changes
                # We can simulate by actually doing it, but that's expensive if we revert.
                # Better to calculate delta.
                
                # Let's just do a limited random swap search if full search is too slow?
                # No, 50k is fine.
                
                for y in range(n + 1):
                    if y in A:
                        continue
                    
                    # Simulate swap
                    # Remove x
                    update_counts_remove(x)
                    # Add y
                    update_counts_add(y)
                    
                    if is_valid():
                        new_peak = get_peak()
                        if new_peak < best_new_peak:
                            best_new_peak = new_peak
                            best_swap = (x, y)
                            # If we find peak 1, we can't do better.
                            if best_new_peak == 1:
                                break
                    else:
                        # Invalid, revert
                        update_counts_remove(y)
                        update_counts_add(x)
                        continue
                    
                    # Revert for next candidate
                    update_counts_remove(y)
                    update_counts_add(x)
                    
                    if best_new_peak == 1:
                        break
                
                if best_new_peak == 1:
                    break
            
            if best_swap is not None:
                x, y = best_swap
                update_counts_remove(x)
                update_counts_add(y)
                current_peak = best_new_peak
                improved = True
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > 5:
                    # Stuck in local minimum
                    # Try a random perturbation?
                    # Or break?
                    # Let's try one random swap that maintains validity, even if peak increases, to escape.
                    # This is risky. Let's just break if we can't improve.
                    break

    # Return the set as a list
    return list(A)

# Test locally? No, just output.
