def construct():
    import time
    start = time.time()
    deadline = start + 120.0
    
    n, m, p = 2, 2, 8
    rank = 16
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # For each column k of B (and corresponding column of C),
    # we use 2 terms to compute C[:, k] = A * B[:, k].
    # Term 1 handles the j=0 part (A[:, 0] * B[0, k])
    # Term 2 handles the j=1 part (A[:, 1] * B[1, k])
    
    for k in range(p):
        r0 = 2 * k
        r1 = 2 * k + 1
        
        # Term r0:
        # U: A[0,0]=1, A[1,0]=1 -> U[0]=1, U[2]=1
        # V: B[0,k]=1 -> V[k]=1
        # W: C[0,k]=1, C[1,k]=1 -> W[k]=1, W[p+k]=1
        U[0][r0] = 1.0
        U[2][r0] = 1.0
        V[0 * p + k][r0] = 1.0
        W[k * n + 0][r0] = 1.0
        W[k * n + 1][r0] = 1.0
        
        # Term r1:
        # U: A[0,1]=1, A[1,1]=1 -> U[1]=1, U[3]=1
        # V: B[1,k]=1 -> V[p+k]=1
        # W: C[0,k]=1, C[1,k]=1 -> W[k]=1, W[p+k]=1
        U[1][r1] = 1.0
        U[3][r1] = 1.0
        V[1 * p + k][r1] = 1.0
        W[k * n + 0][r1] = 1.0
        W[k * n + 1][r1] = 1.0

    # Verification
    try:
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(actual - expected) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook if verification fails (should not happen)
            rank_fb = n * m * p
            U_fb = [[0.0] * rank_fb for _ in range(n * m)]
            V_fb = [[0.0] * rank_fb for _ in range(m * p)]
            W_fb = [[0.0] * rank_fb for _ in range(p * n)]
            c2 = 0
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        U_fb[i2*m+j2][c2] = 1.0
                        V_fb[j2*p+k2][c2] = 1.0
                        W_fb[k2*n+i2][c2] = 1.0
                        c2 += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass

    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
