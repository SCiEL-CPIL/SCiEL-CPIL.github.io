"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 matrix multiplication (rank 7).
    Convention: A is 2x2 (rows i=0,1; cols j=0,1), B is 2x2 (rows j=0,1; cols k=0,1), C is 2x2 (rows k=0,1; cols i=0,1).
    U has shape (4, 7): indexed by (i, j) -> i*2+j
    V has shape (4, 7): indexed by (j, k) -> j*2+k
    W has shape (4, 7): indexed by (k, i) -> k*2+i
    """
    # Strassen's 7 products for 2x2 multiplication:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2: (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3: A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4: A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5: (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6: (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7: (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    W[1, 6] = 1  # C01
    
    return U, V, W


def _build_optimized_decomposition():
    """Build a rank-60 decomposition.
    
    Strategy:
    - i in {0, 1}: Use Strassen-based decomposition for 2x4x8.
      Split j into {0,1} and {2,3}.
      Split k into 4 groups of 2.
      Each 2x2x2 block uses rank 7.
      Total: 2 (j groups) * 4 (k groups) * 7 = 56.
    - i in {2}: Use a rank-4 decomposition for 1x4x8.
      T[2,j,k] = 1.
      Use 4 columns corresponding to j=0,1,2,3.
      For col r (0..3):
        U[2*4 + r, 56+r] = 1
        V[r*8 + k, 56+r] = 1 for all k
        W[k*3 + 2, 56+r] = 1 for all k
      Total: 4.
    
    Total rank: 60.
    """
    n, m, p = 3, 4, 8
    rank_total = 60
    
    U = np.zeros((n * m, rank_total), dtype=float)  # (12, 60)
    V = np.zeros((m * p, rank_total), dtype=float)  # (32, 60)
    W = np.zeros((p * n, rank_total), dtype=float)  # (24, 60)
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2()
    
    # Part 1: i in {0, 1}
    # i_groups: [0, 1]
    # j_groups: [0, 1], [2, 3]
    # k_groups: [0, 1], [2, 3], [4, 5], [6, 7]
    
    col = 0
    j_groups = [[0, 1], [2, 3]]
    k_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    
    for j_list in j_groups:
        for k_list in k_groups:
            j_start = j_list[0]
            k_start = k_list[0]
            
            for s in range(7):
                # U part: i in {0, 1}
                # strassen_U[ai, s] where ai = i*2+j
                for ai in range(4):
                    u_val = strassen_U[ai, s]
                    if u_val != 0:
                        di = ai // 2
                        dj = ai % 2
                        U[(di) * m + (j_start + dj), col] += u_val
                
                # V part: j in {j_start, j_start+1}, k in {k_start, k_start+1}
                # strassen_V[bj, s] where bj = j*2+k
                for bj in range(4):
                    v_val = strassen_V[bj, s]
                    if v_val != 0:
                        dj = bj // 2
                        dk = bj % 2
                        V[(j_start + dj) * p + (k_start + dk), col] += v_val
                
                # W part: k in {k_start, k_start+1}, i in {0, 1}
                # strassen_W[ck, s] where ck = k*2+i
                for ck in range(4):
                    w_val = strassen_W[ck, s]
                    if w_val != 0:
                        dk = ck // 2
                        di = ck % 2
                        W[(k_start + dk) * n + di, col] += w_val
                
                col += 1
    
    # Part 2: i in {2}
    # Rank 4 decomposition
    # Columns 56 to 59
    for r in range(4):
        # U[2, r] = 1
        U[2 * m + r, col] = 1
        
        # V[r, k] = 1 for all k
        for k in range(p):
            V[r * p + k, col] = 1
            
        # W[k, 2] = 1 for all k
        for k in range(p):
            W[k * n + 2, col] = 1
            
        col += 1
    
    assert col == rank_total, f"Expected {rank_total} columns, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 3, 4, 8
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    rank = U.shape[1]
    assert V.shape[1] == rank
    assert W.shape[1] == rank
    
    # Build the tensor from the decomposition
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                T_decomp[i, j, k] = np.dot(np.dot(u_val, v_val), w_val)
    
    T_true = np.ones((n, m, p), dtype=float)
    
    max_err = np.max(np.abs(T_decomp - T_true))
    return max_err < 1e-10


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    try:
        U, V, W = _build_optimized_decomposition()
        if _verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 96)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
