import random
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 128
    target_lo = 128
    target_hi = 256
    max_sum = 2 * n
    
    # We want to minimize the peak multiplicity in [128, 256]
    # subject to all counts in [128, 256] being >= 1.
    
    def compute_counts(A):
        """Compute the count of pairs summing to each value in [0, 2n]."""
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        A_list = list(A)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                counts[s] += 1
        return counts

    def get_metrics(counts):
        """
        Returns (uncovered_count, peak, sum_sq).
        Lower is better for all.
        """
        target_counts = counts[target_lo:target_hi + 1]
        uncovered = np.sum(target_counts == 0)
        peak = np.max(target_counts)
        sum_sq = np.sum(target_counts * target_counts)
        return (uncovered, peak, sum_sq)

    def update_counts_add(A, counts, y):
        """Update counts in-place when adding y to A."""
        # y pairs with all existing elements in A
        # Also y pairs with itself
        for a in A:
            s = a + y
            if s <= max_sum:
                counts[s] += 1
        s_2y = 2 * y
        if s_2y <= max_sum:
            counts[s_2y] += 1

    def update_counts_remove(A, counts, x):
        """Update counts in-place when removing x from A."""
        # x pairs with all remaining elements in A
        # And x paired with itself
        for a in A:
            s = a + x
            if s <= max_sum:
                counts[s] -= 1
        s_2x = 2 * x
        if s_2x <= max_sum:
            counts[s_2x] -= 1

    def update_counts_swap(A, counts, x, y):
        """Update counts in-place when swapping x (in A) with y (not in A)."""
        # Remove x
        update_counts_remove(A, counts, x)
        A.remove(x)
        
        # Add y
        A.add(y)
        update_counts_add(A, counts, y)

    # --- Phase 1: Greedy Construction ---
    # We want to build a set that covers [128, 256].
    # Strategy: Add elements that contribute to covering the most "uncovered" sums.
    # If multiple, choose the one that minimizes the peak increase.
    
    A = set()
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # We might need roughly 20-40 elements. Let's aim for a set size that allows coverage.
    # A set of size K has ~K^2/2 pairs. We need 129 distinct sums.
    # K=20 -> 210 pairs. K=30 -> 465 pairs.
    # Let's try to build up to size 50, but stop if we can't improve metrics.
    
    max_size = 60
    best_A_greedy = set()
    best_metrics_greedy = (float('inf'), float('inf'), float('inf'))
    
    # Initialize with a few random elements to break symmetry? Or start empty.
    # Starting empty is fine.
    
    for _ in range(max_size):
        if time.time() > deadline - 10.0:
            break
            
        best_y = None
        best_score = (float('inf'), float('inf'), float('inf'))
        
        # To speed up, we don't need to check ALL y in [0, 128] every time if we are smart,
        # but N=128 is small. 128 * (cost of evaluation) is fine.
        # Cost of evaluation: copy counts (128 ints), update (O(|A|)), compute metrics (O(128)).
        # This is very fast.
        
        for y in range(n + 1):
            if y in A:
                continue
            
            # Simulate adding y
            temp_counts = counts.copy()
            for a in A:
                s = a + y
                if s <= max_sum:
                    temp_counts[s] += 1
            s_2y = 2 * y
            if s_2y <= max_sum:
                temp_counts[s_2y] += 1
            
            metrics = get_metrics(temp_counts)
            
            # We prioritize reducing uncovered.
            # If uncovered is same, we prioritize lower peak.
            # If peak is same, lower sum_sq.
            # Also, we prefer smaller sets? No, we are building.
            
            if metrics < best_score:
                best_score = metrics
                best_y = y
        
        if best_y is None:
            break
            
        # Check if we actually improved
        if best_score < best_metrics_greedy:
            # Commit the change
            A.add(best_y)
            update_counts_add(A, counts, best_y)
            best_metrics_greedy = best_score
            best_A_greedy = set(A)
        else:
            # If we can't improve metrics by adding another element, stop.
            # Adding more elements will only increase multiplicity.
            break

    # --- Phase 2: Local Search (Swaps) ---
    # Now we have a valid set (hopefully). We want to reduce the peak multiplicity.
    # We perform swaps: remove x, add y.
    # Accept if:
    # 1. Coverage is maintained (uncovered == 0)
    # 2. Peak is reduced OR (Peak is same and sum_sq is reduced)
    
    if len(best_A_greedy) > 0:
        A = set(best_A_greedy)
        counts = compute_counts(A)
        curr_metrics = get_metrics(counts)
        
        best_A_local = set(A)
        best_metrics_local = curr_metrics
        
        no_improve_count = 0
        max_no_improve = 10000
        
        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            # Pick a random element to remove
            x = random.choice(list(A))
            
            # Pick a random element to add
            y = random.randint(0, n)
            if y == x or y in A:
                continue
                
            # Perform swap
            update_counts_swap(A, counts, x, y)
            new_metrics = get_metrics(counts)
            
            # Check if valid and better
            # Valid means new_metrics[0] == 0 (no uncovered sums)
            # Better means new_metrics[1] < curr_metrics[1] or (== and new_metrics[2] < curr_metrics[2])
            
            if new_metrics[0] == 0:
                if new_metrics[1] < curr_metrics[1]:
                    curr_metrics = new_metrics
                    no_improve_count = 0
                    if curr_metrics < best_metrics_local:
                        best_metrics_local = curr_metrics
                        best_A_local = set(A)
                elif new_metrics[1] == curr_metrics[1]:
                    if new_metrics[2] < curr_metrics[2]:
                        curr_metrics = new_metrics
                        no_improve_count = 0
                        if curr_metrics < best_metrics_local:
                            best_metrics_local = curr_metrics
                            best_A_local = set(A)
                    else:
                        # Revert
                        update_counts_swap(A, counts, y, x)
                        no_improve_count += 1
                else:
                    # Peak increased, revert
                    update_counts_swap(A, counts, y, x)
                    no_improve_count += 1
            else:
                # Coverage broken, revert
                update_counts_swap(A, counts, y, x)
                no_improve_count += 1

    # --- Phase 3: Try to shrink the set ---
    # If we have a valid set, can we remove an element and still be valid?
    # This would reduce the total number of pairs, potentially lowering peak.
    
    if len(best_A_local) > 0:
        A = set(best_A_local)
        counts = compute_counts(A)
        curr_metrics = get_metrics(counts)
        
        # Try removing elements
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try removing each element
            for x in list(A):
                # Simulate removal
                temp_counts = counts.copy()
                for a in A:
                    if a == x:
                        continue
                    s = a + x
                    if s <= max_sum:
                        temp_counts[s] -= 1
                s_2x = 2 * x
                if s_2x <= max_sum:
                    temp_counts[s_2x] -= 1
                
                new_metrics = get_metrics(temp_counts)
                
                if new_metrics[0] == 0:
                    # Valid removal
                    # Check if it improves peak or sum_sq
                    if new_metrics[1] < curr_metrics[1] or (new_metrics[1] == curr_metrics[1] and new_metrics[2] < curr_metrics[2]):
                        # Commit removal
                        A.remove(x)
                        # Update counts in place
                        for a in A:
                            s = a + x
                            if s <= max_sum:
                                counts[s] -= 1
                        s_2x = 2 * x
                        if s_2x <= max_sum:
                            counts[s_2x] -= 1
                        curr_metrics = new_metrics
                        improved = True
                        break # Restart loop as A changed

    # Final check: if we still have uncovered sums (which shouldn't happen if greedy worked),
    # we need to return something valid.
    # If best_A_local is empty or invalid, fall back to a simple dense set.
    
    final_A = best_A_local
    if len(final_A) == 0:
        # Fallback: A dense set usually works. E.g., all even numbers + some odds.
        # Or just a large random set.
        final_A = set(range(0, n + 1, 2))
        # Add some randoms to ensure coverage
        while True:
            counts = compute_counts(final_A)
            m = get_metrics(counts)
            if m[0] == 0:
                break
            final_A.add(random.randint(0, n))
            if time.time() > deadline - 1.0:
                # If we can't find coverage, return a very dense set
                final_A = set(range(0, n + 1))
                break

    # Ensure we return a list
    return list(final_A)

# Test helper (not part of submission)
if __name__ == "__main__":
    A = construct()
    counts = compute_counts(A)
    m = get_metrics(counts)
    print(f"Set size: {len(A)}")
    print(f"Metrics: {m}")
    print(f"Uncovered: {m[0]}, Peak: {m[1]}, SumSq: {m[2]}")
