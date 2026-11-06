def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # next_conflict[S] = smallest y >= S such that inv[y] >= S
    # If no such y, it is n+1
    next_conflict = [n + 1] * (n + 2)
    
    # For each y, it affects S in [1, min(y, inv[y])]
    # We want the minimum y for each S
    for y in range(1, n + 1):
        limit = min(y, inv[y])
        # For S in 1..limit, y is a candidate
        # We iterate S and update min
        # Optimization: if next_conflict[S] is already <= y, no update needed
        # But we can just do it, N is small
        for S in range(1, limit + 1):
            if y < next_conflict[S]:
                next_conflict[S] = y
    
    best_start = 1
    best_length = 0
    
    for S in range(1, n + 1):
        # The interval is [S, next_conflict[S] - 1]
        # Length is next_conflict[S] - S
        # But we must ensure the interval is non-empty and valid
        # If next_conflict[S] == S, length is 0. (Self-inverse or conflict at start)
        # Wait, if inv[S] == S, then limit = S. next_conflict[S] can be S.
        # If next_conflict[S] > S, length is next_conflict[S] - S.
        length = next_conflict[S] - S
        if length > best_length:
            best_length = length
            best_start = S
            
    return [best_start, best_length]
