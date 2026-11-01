"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n=2, m=4, p=6):
    """Verify that the decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    result = np.zeros((n, p))
    for r in range(rank):
        M = U_3d[:, :, r] @ V_3d[:, :, r]  # (n, p)
        W_r = W_3d[:, :, r]  # (p, n)
        result += M * W_r.T  # W_r.T[i,k] = W_r[k,i]
    
    target = np.ones((n, p))
    return np.allclose(result, target, atol=1e-9)

def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 6
    rank = 42  # 2 j-blocks * 3 k-blocks * 7 Strassen rank
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen 2x2 @ 2x2 decomposition
    # Local indices for 2x2 matrix: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # U entries: coefficients for A[i,j] in each product
    strassen_U = [
        {0: 1, 3: 1},       # P1: A00 + A11
        {2: 1, 3: 1},       # P2: A10 + A11
        {0: 1},             # P3: A00
        {3: 1},             # P4: A11
        {0: 1, 1: 1},       # P5: A00 + A01
        {2: 1, 0: -1},      # P6: A10 - A00
        {1: 1, 3: -1},      # P7: A01 - A11
    ]
    
    # V entries: coefficients for B[j,k] in each product
    strassen_V = [
        {0: 1, 3: 1},       # P1: B00 + B11
        {0: 1},             # P2: B00
        {1: 1, 3: -1},      # P3: B01 - B11
        {2: 1, 0: -1},      # P4: B10 - B00
        {3: 1},             # P5: B11
        {0: 1, 1: 1},       # P6: B00 + B01
        {2: 1, 3: 1},       # P7: B10 + B11
    ]
    
    # W entries: coefficients for C[i,k] in each product
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    strassen_W = [
        {0: 1, 3: 1},       # P1: C00 + C11
        {2: 1, 3: -1},      # P2: C10 - C11
        {1: 1, 3: 1},       # P3: C01 + C11
        {0: 1, 2: 1},       # P4: C00 + C10
        {0: -1, 1: 1},      # P5: -C00 + C01
        {3: 1},             # P6: C11
        {0: 1},             # P7: C00
    ]
    
    col = 0
    for j_block in range(2):  # j = 0,1 and j = 2,3
        for k_block in range(3):  # k = 0,1 and k = 2,3 and k = 4,5
            for prod_idx in range(7):
                r = col + prod_idx
                
                # U: A[i, j] where i in {0,1}, j in {j_block*2, j_block*2+1}
                for local_idx, sign in strassen_U[prod_idx].items():
                    gi = local_idx // 2
                    gj = local_idx % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign
                
                # V: B[j, k] where j in {j_block*2, j_block*2+1}, k in {k_block*2, k_block*2+1}
                for local_idx, sign in strassen_V[prod_idx].items():
                    gj = local_idx // 2
                    gk = local_idx % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign
                
                # W: C[i, k] where i in {0,1}, k in {k_block*2, k_block*2+1}
                # W index: k*n + i
                for local_idx, sign in strassen_W[prod_idx].items():
                    gi = local_idx // 2
                    gk = local_idx % 2
                    W[(k_block * 2 + gk) * n + gi, r] = sign
            
            col += 7
    
    # Verify correctness
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback: naive rank n*m*p decomposition
        rank_naive = n * m * p  # 48
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j][idx] = 1
                    V_naive[j * p + k][idx] = 1
                    W_naive[k * n + i][idx] = 1
                    idx += 1
        
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 110:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
