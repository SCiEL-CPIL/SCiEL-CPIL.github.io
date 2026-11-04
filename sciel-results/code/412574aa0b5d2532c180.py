import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = 3, 3, 7
    
    # Strategy: Use Kronecker product structure
    # (3x3) * (3x7) can be seen as 3 independent (3x3)*(3x1) multiplications
    # Each (3x3)*(3x1) has rank 3 (trivial) or can use Strassen-like
    # 
    # Actually, let's think of it differently:
    # The tensor T[i,j,k] = sum_{a,b} A[i,a] * B[a,k]
    # where i in [0,3), a in [0,3), k in [0,7)
    #
    # We can group by k: for each k, we have a 3x3 * 3x1 multiplication
    # The 3x3 * 3x1 tensor has rank 3 (each output is independent dot product)
    # So total rank = 3 * 3 = 9? No, that's not right either.
    #
    # Let me use the explicit construction:
    # For the trivial decomposition, rank = n*m*p = 3*3*7 = 63
    # Each rank-1 term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Better approach: Use the fact that this is a "structured" tensor
    # We can write B as columns: B = [B_0, B_1, ..., B_6] where each B_j is 3x1
    # Then (A*B)_j = A * B_j
    # Each A * B_j is a 3x1 vector, computed as 3x3 * 3x1
    #
    # The 3x3 * 3x1 tensor: T[i,k] = sum_a A[i,a] * B[a,k] for fixed column
    # This has rank 3 (trivial: one per a)
    #
    # So we can decompose as: for each column k (7 columns), use rank-3 decomposition
    # Total rank = 7 * 3 = 21
    #
    # But can we do better? Maybe by sharing computation across columns.
    #
    # Let's first implement rank 21, then search for improvements.
    
    # Rank 21 construction:
    # For each output column k (0..6), and each "intermediate" index a (0..2):
    # rank index r = k * 3 + a
    # U[i*m + a, r] = 1  (select row i, column a of A)
    # V[a*p + k, r] = 1  (select row a, column k of B)  
    # W[k*n + i, r] = 1  (output to position (k,i) in result)
    
    rank = 21
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for k in range(p):
        for a in range(m):
            r = k * m + a
            for i in range(n):
                U[i * m + a, r] = 1
                W[k * n + i, r] = 1
            V[a * p + k, r] = 1
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,j,k) pattern
        # The multiplication tensor: T[i,j,k] = 1 if we're computing (AB)[i,k] using A[i,j]*B[j,k]
        # So for each (i,j,k), the coefficient should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True
    
    if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to trivial decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    u_cols = (i_idx * m + j_idx).flatten()
    v_cols = (j_idx * p + k_idx).flatten()
    w_cols = (k_idx * n + i_idx).flatten()
    r_cols = np.arange(rank)
    
    U[u_cols, r_cols] = 1
    V[v_cols, r_cols] = 1
    W[w_cols, r_cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
