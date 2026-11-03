def _greedy_reduce(U, V, W):
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    best = (U, V, W, R)
    
    # Try multiple times with different orders
    import random
    for _ in range(10):
        U_curr, V_curr, W_curr, R_curr = best
        # Shuffle columns
        indices = list(range(R_curr))
        random.shuffle(indices)
        # Remap
        U_curr = [ [row[i] for i in indices] for row in U_curr ]
        V_curr = [ [row[i] for i in indices] for row in V_curr ]
        W_curr = [ [row[i] for i in indices] for row in W_curr ]
        
        # Run greedy reduction
        U_new, V_new, W_new, R_new = _greedy_reduce_inner(U_curr, V_curr, W_curr)
        if R_new < best[3]:
            best = (U_new, V_new, W_new, R_new)
            
    return best

def _greedy_reduce_inner(U, V, W):
    # Original logic
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    best = (U, V, W, R)
    for c0 in range(R):
        # ... same logic ...
        if _validate(newU, newV, newW):
            R2 = R - 1
            if R2 < best[3]:
                best = (newU, newV, newW, R2)
                U, V, W, R = newU, newV, newW, R2
    return best
