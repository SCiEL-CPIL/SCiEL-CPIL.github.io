        # Pruning: Check if remaining sum is too large
        # Max sum of k-1 distinct unit fractions starting from d+1
        # Approx sum_{j=1}^{k-1} 1/(d+j)
        # If new_num/new_den > this sum, impossible.
        # We can compute this sum exactly or bound it.
        # For small k, exact sum is fine.
        if k > 1:
            max_sum_num = 0
            max_sum_den = 1
            for j in range(1, k):
                # Add 1/(d+j)
                # max_sum = max_sum + 1/(d+j)
                # (max_sum_num * (d+j) + max_sum_den) / (max_sum_den * (d+j))
                max_sum_num = max_sum_num * (d + j) + max_sum_den
                max_sum_den = max_sum_den * (d + j)
                # Reduce? Might get large.
                # But k is small (<=20).
            
            # Check if new_num/new_den > max_sum_num/max_sum_den
            # new_num * max_sum_den > new_den * max_sum_num
            if new_num * max_sum_den > new_den * max_sum_num:
                continue
