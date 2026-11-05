def update_counts_remove(current_A_bool, current_counts, x):
    # (x, x)
    s = 2 * x
    if min_sum <= s <= max_sum:
        current_counts[s - min_sum] -= 1
    
    # (x, y) for y in A \ {x}
    # Get y's
    y_indices = np.nonzero(current_A_bool)[0]
    # x is already False in current_A_bool, so y_indices is A \ {x}
    if len(y_indices) > 0:
        sums = x + y_indices
        mask = (sums >= min_sum) & (sums <= max_sum)
        valid_sums = sums[mask]
        idxs = valid_sums - min_sum
        current_counts[idxs] -= 1

def update_counts_add(current_A_bool, current_counts, x):
    # (x, x)
    s = 2 * x
    if min_sum <= s <= max_sum:
        current_counts[s - min_sum] += 1
    
    # (x, y) for y in A \ {x}
    # x is True in current_A_bool now.
    y_indices = np.nonzero(current_A_bool)[0]
    # Exclude x
    y_indices = y_indices[y_indices != x]
    if len(y_indices) > 0:
        sums = x + y_indices
        mask = (sums >= min_sum) & (sums <= max_sum)
        valid_sums = sums[mask]
        idxs = valid_sums - min_sum
        current_counts[idxs] += 1
