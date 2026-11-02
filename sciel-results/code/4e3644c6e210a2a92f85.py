def verify(U, V, W, n, m, p, R):
    # U, V, W are lists of lists
    # Convert to numpy for speed
    U_np = np.array(U).reshape(n, m, R)
    V_np = np.array(V).reshape(m, p, R)
    W_np = np.array(W).reshape(p, n, R)
    
    # Check all tuples
    for a in range(n):
        for b in range(m):
            for bp in range(m):
                for c in range(p):
                    for k in range(p):
                        for i in range(n):
                            # Sum over r
                            s = 0.0
                            for r in range(R):
                                s += U_np[a, b, r] * V_np[bp, c, r] * W_np[k, i, r]
                            
                            target = 1.0 if (a==i and c==k and b==bp) else 0.0
                            if abs(s - target) > 1e-9:
                                return False
    return True
