    def search(i, current_sum, current_mask):
        nonlocal best_sum, best_mask
        
        # Update best if current partial sum is valid and better
        if current_sum < 1 and current_sum > best_sum:
            best_sum = current_sum
            best_mask = current_mask

        # Pruning: Even if we take all remaining, we can't beat best_sum
        # Note: best_sum might have just been updated to current_sum.
        # If current_sum + suffix_sums[i] <= best_sum, then 
        # suffix_sums[i] <= best_sum - current_sum.
        # If best_sum == current_sum, then suffix_sums[i] <= 0, which is false unless suffix is 0.
        # So this pruning is still valid.
        if current_sum + suffix_sums[i] <= best_sum:
            return

        if i == n:
            return

        # Option 1: Exclude item i
        search(i + 1, current_sum, current_mask)

        # Option 2: Include item i
        if current_sum + vals[i] < 1:
            search(i + 1, current_sum + vals[i], current_mask | (1 << i))
