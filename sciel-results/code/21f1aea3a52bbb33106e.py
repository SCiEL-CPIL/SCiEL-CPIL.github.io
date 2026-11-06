    # Initialize with greedy solution
    greedy_sum = Fraction(0, 1)
    greedy_indices = []
    for i in sorted_indices:
        if greedy_sum + recips[sorted_indices.index(i)] < 1: # Note: recips is aligned with sorted_indices
             # Wait, recips is aligned with sorted_indices.
             # Let's just iterate sorted_indices
             pass
             
    # Let's just do a simple greedy pass on sorted_indices
    g_sum = Fraction(0, 1)
    g_idx = []
    for i in range(n):
        if g_sum + recips[i] < 1:
            g_sum += recips[i]
            g_idx.append(sorted_indices[i])
            
    if g_sum > best_sum:
        best_sum = g_sum
        best_indices = g_idx
