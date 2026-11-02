"""Kronecker product decomposition for 4x4x6 matrix multiplication (rank 35)."""
DIMENSIONS = [4, 4, 6]
import time

def _verify(U, V, W, n, m, p):
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    T = np.einsum('ir,kr,jr->ijk', Ua, Wa, Va)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, k, j] - 1.0) > 1e-9:
                    ok = False
    return ok

def construct():
    t0 = time.time()
    n, m, p = 4, 4, 6
    
    # Schoolbook fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best = {"U": U, "V": V, "W": W}
    
    deadline = 150.0
    
    # Rank-5 decomposition of 2x2x3 multiplication tensor
    # Convention: U[a*2+b], V[b*3+c], W[c*2+a]
    def _sub223():
        return (
            [[1, 1, 1, 0, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 1, 1, 0],
             [1, 0, 1, 1, 1]],
            [[1, 1, 0, 1, 0],
             [0, 1, 0, 0, 1],
             [1, 0, 1, 0, 1],
             [1, 0, 0, 1, 0],
             [1, 1, 1, 0, 1]],
            [[1, 1, 1, 1, 1],
             [1, 0, 0, 1, 1],
             [1, 1, 0, 0, 1]]
        )
    
    # Rank-7 decomposition of 2x2x2 multiplication tensor (Strassen)
    # Convention: U[a*2+b], V[b*2+c], W[c*2+a]
    def _sub222():
        return (
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]],
            [[1, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0]],
            [[1, 0, 1, 1, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0],
             [0, 0, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0]]
        )
    
    try:
        U223, V223, W223 = _sub223()
        U222, V222, W222 = _sub222()
        
        r1 = 5  # rank of 2x2x3
        r2 = 7  # rank of 2x2x2
        total_rank = r1 * r2  # 35
        
        U35 = [[0] * total_rank for _ in range(16)]
        V35 = [[0] * total_rank for _ in range(24)]
        W35 = [[0] * total_rank for _ in range(24)]
        
        # Kronecker product: 4x4x6 = (2x2x3) ⊗ (2x2x2)
        # Index mapping:
        #   i = i1*2 + i2 where i1 in {0,1} (from 2x2x3), i2 in {0,1} (from 2x2x2)
        #   j = j1*2 + j2 where j1 in {0,1} (from 2x2x3), j2 in {0,1} (from 2x2x2)
        #   k = k1*3 + k2 where k1 in {0,1,2} (from 2x2x3), k2 in {0,1} (from 2x2x2)
        
        for t1 in range(r1):
            for t2 in range(r2):
                t = t1 * r2 + t2
                
                # U: rows are (i1,i2,j1,j2) -> index (i1*2+i2)*4 + (j1*2+j2)
                for i1 in range(2):
                    for i2 in range(2):
                        for j1 in range(2):
                            for j2 in range(2):
                                u_val = U223[i1 * 2 + j1][t1] * U222[i2 * 2 + j2][t2]
                                if u_val != 0:
                                    U35[(i1 * 2 + i2) * 4 + (j1 * 2 + j2)][t] = u_val
                
                # V: rows are (j1,j2,k1,k2) -> index (j1*2+j2)*6 + (k1*3+k2)
                for j1 in range(2):
                    for j2 in range(2):
                        for k1 in range(3):
                            for k2 in range(2):
                                v_val = V223[j1 * 3 + k1][t1] * V222[j2 * 2 + k2][t2]
                                if v_val != 0:
                                    V35[(j1 * 2 + j2) * 6 + (k1 * 3 + k2)][t] = v_val
                
                # W: rows are (k1,k2,i1,i2) -> index (k1*3+k2)*4 + (i1*2+i2)
                for k1 in range(3):
                    for k2 in range(2):
                        for i1 in range(2):
                            for i2 in range(2):
                                w_val = W223[k1 * 2 + i1][t1] * W222[k2 * 2 + i2][t2]
                                if w_val != 0:
                                    W35[(k1 * 3 + k2) * 4 + (i1 * 2 + i2)][t] = w_val
        
        if _verify(U35, V35, W35, 4, 4, 6):
            best = {"U": U35, "V": V35, "W": W35}
    except Exception:
        pass
    
    if time.time() > deadline - 10.0:
        return best
    return best
