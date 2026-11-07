    for i in range(n):
        for j in range(i+1, n):
            # Skip if edge already exists (base or added)
            if (adj_masks[i] >> j) & 1:
                continue
            
            # Skip if it creates a triangle
            if (adj_masks[i] & adj_masks[j]) != 0:
                continue
            # ... evaluate
