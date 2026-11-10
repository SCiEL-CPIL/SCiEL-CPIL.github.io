import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    num_points = q * q
    num_dirs = q + 1
    
    # Field arithmetic setup
    # Element i corresponds to a + b*alpha
    # a = i // p, b = i % p
    
    def add_f49(i, j):
        a1, b1 = divmod(i, p)
        a2, b2 = divmod(j, p)
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)

    def mul_f49(i, j):
        a1, b1 = divmod(i, p)
        a2, b2 = divmod(j, p)
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1a2 + a1b2*alpha + b1a2*alpha + b1b2*alpha^2
        # alpha^2 = 3
        # = (a1a2 + 3*b1b2) + (a1b2 + b1a2)*alpha
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + b1 * a2) % p
        return a * p + b

    # Precompute neighbors for each point and direction
    # neighbors[x][v][t_idx] = y
    # v in 0..48: direction (1, m) where m is F49 element v
    # v == 49: direction (0, 1)
    
    neighbors = np.zeros((num_points, num_dirs, q-1), dtype=np.int32)
    
    for x in range(num_points):
        xa, xb = divmod(x, q) # x is a point in F49^2? No.
        # Wait, x is index 0..2400.
        # x = X * 49 + Y
        X = x // q
        Y = x % q
        
        for v in range(num_dirs):
            if v < q:
                # Direction (1, m) with m = v
                # Point = (X + t, Y + m*t)
                for t_idx, t in enumerate(range(1, q)): # t != 0
                    # t is F49 element
                    # X_new = X + t
                    # Y_new = Y + m*t
                    X_new = add_f49(X, t)
                    Y_new = add_f49(Y, mul_f49(v, t))
                    y_idx = X_new * q + Y_new
                    neighbors[x][v][t_idx] = y_idx
            else:
                # Direction (0, 1)
                # Point = (X, Y + t)
                for t_idx, t in enumerate(range(1, q)):
                    X_new = X
                    Y_new = add_f49(Y, t)
                    y_idx = X_new * q + Y_new
                    neighbors[x][v][t_idx] = y_idx

    def greedy_solve(order):
        F = [ (1 << num_dirs) - 1 ] * num_points
        S = set()
        
        for x in order:
            if F[x] == 0:
                continue
            
            valid = True
            # Check if adding x causes any neighbor to lose its last direction
            for v in range(num_dirs):
                if not (F[x] & (1 << v)):
                    # x doesn't have this direction free, so adding x doesn't affect neighbors' free status for this v?
                    # Wait.
                    # If x is added to S, then for any neighbor y on line (x, v), v is no longer free for y.
                    # This happens regardless of whether v was free for x.
                    # The condition for x itself is that v was free for x BEFORE adding x.
                    # But the update to neighbors happens for ALL v.
                    pass
                
                # We must check all v, because adding x blocks v for all neighbors on line v.
                # Even if v is not in F[x], it might be the last direction for some neighbor y.
                # So we must check all v.
                
                # Optimization: If F[x] has bit v set, it's "safe" for x.
                # But we need to check neighbors.
                
                # If F[x] does NOT have bit v set, can we skip?
                # No, because x is added to S. The line L(x,v) contains x.
                # For any y on L(x,v), x is in S and on L(y,v). So v is blocked for y.
                # This is true for all v.
                
                # So we must iterate all v.
                
                # Check neighbors
                for y in neighbors[x][v]:
                    if F[y] == (1 << v):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.add(x)
                # Update F for neighbors
                for v in range(num_dirs):
                    mask = ~(1 << v)
                    for y in neighbors[x][v]:
                        F[y] &= mask
        
        return S

    best_S = set()
    start_time = time.time()
    timeout = 580 # 20s margin
    
    # Try multiple random restarts
    points_list = list(range(num_points))
    
    while time.time() - start_time < timeout:
        random.shuffle(points_list)
        S = greedy_solve(points_list)
        if len(S) > len(best_S):
            best_S = S
        
        # Optional: Try local search on best_S?
        # For now, just restarts.
        
    # Construct N
    N_points = []
    for x in range(num_points):
        if x not in best_S:
            X = x // q
            Y = x % q
            Xa, Xb = divmod(X, p)
            Ya, Yb = divmod(Y, p)
            N_points.append([[Xa, Xb], [Ya, Yb]])
            
    return np.array(N_points, dtype=np.int64)
