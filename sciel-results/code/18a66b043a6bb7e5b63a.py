    D = np.zeros(max_x + 2, dtype=np.float64)
    
    # Contribution from k=1: f1 * x for x in [1, max_x]
    # This is a linear function, not constant on intervals
    # Actually floor(x/1) = x, so it's f1*x, not constant
    # Handle separately: M[x] += f1 * x
    
    # For k >= 2:
    for idx in range(m):
        k = domain[idx]
        fk = values[idx]
        j = 1
        while j * k <= max_x:
            lo = j * k
            hi = min((j+1) * k - 1, max_x)
            D[lo] += fk
            if hi + 1 <= max_x:
                D[hi+1] -= fk
            j += 1
    
    # Prefix sum to get M from D
    M_from_k = np.cumsum(D)
    
    # Add k=1 contribution
    M = M_from_k + f1 * np.arange(max_x + 1)
    M[0] = 0
