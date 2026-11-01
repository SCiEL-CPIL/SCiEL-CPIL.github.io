"""
Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.
Uses Strassen 2x2x2 blocks: split the 4 middle dimension into 2 blocks of size 2,
and the 6 output dimension into 3 blocks of size 2.
Total: 6 independent 2x2x2 multiplications, each of rank 7 (Strassen).
Total rank: 6 * 7 = 42.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x6 matrix. Output factor W follows the trace convention W[k*n+i, r].
    
    Interface: returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For A (2x2) @ B (2x2) = C (2x2):
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = (a00)(b10+b11)
    # m4 = (a11)(b01-b11)
    # m5 = (a00+a01)(b00)
    # m6 = (a10-a00)(b11)
    # m7 = (a01-a11)(b00+b10)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # U factors (2x2 for each term), V factors (2x2), W factors (2x2)
    # U_r[i,j] is the coefficient of A[i,j] in the r-th multiplication
    # V_r[j,k] is the coefficient of B[j,k] in the r-th multiplication
    # W_r[k,i] is the coefficient of C[i,k] (note: W index is [k*n+i])
    
    # Let's define the 7 terms for Strassen 2x2x2:
    # Term 1: (a00+a11)(b00+b11) -> contributes to C00 and C11
    # Term 2: (a10+a11)(b00) -> contributes to C10
    # Term 3: (a00)(b10+b11) -> contributes to C01
    # Term 4: (a11)(b01-b11) -> contributes to C00 and C11
    # Term 5: (a00+a01)(b00) -> contributes to C00
    # Term 6: (a10-a00)(b11) -> contributes to C11
    # Term 7: (a01-a11)(b00+b10) -> contributes to C00
    
    # U_r: 2x2 matrix, row i, col j
    # V_r: 2x2 matrix, row j, col k
    # W_r: 2x2 matrix, row k, col i (but stored as W[k*n+i])
    
    U_strassen = np.zeros((7, 2, 2))
    V_strassen = np.zeros((7, 2, 2))
    W_strassen = np.zeros((7, 2, 2))  # W[k, i]
    
    # Term 1: (a00+a11)(b00+b11)
    U_strassen[0] = [[1, 0], [0, 1]]
    V_strassen[0] = [[1, 0], [0, 1]]
    W_strassen[0] = [[1, 0], [0, 1]]  # C00 += 1, C11 += 1
    
    # Term 2: (a10+a11)(b00)
    U_strassen[1] = [[0, 0], [1, 1]]
    V_strassen[1] = [[1, 0], [0, 0]]
    W_strassen[1] = [[0, 0], [1, 0]]  # C10 += 1
    
    # Term 3: (a00)(b10+b11)
    U_strassen[2] = [[1, 0], [0, 0]]
    V_strassen[2] = [[0, 0], [1, 1]]
    W_strassen[2] = [[0, 1], [0, 0]]  # C01 += 1
    
    # Term 4: (a11)(b01-b11)
    U_strassen[3] = [[0, 0], [0, 1]]
    V_strassen[3] = [[0, 1], [0, -1]]
    W_strassen[3] = [[1, 0], [0, 1]]  # C00 += 1, C11 += 1
    
    # Term 5: (a00+a01)(b00)
    U_strassen[4] = [[1, 1], [0, 0]]
    V_strassen[4] = [[1, 0], [0, 0]]
    W_strassen[4] = [[1, 0], [0, 0]]  # C00 += 1
    
    # Term 6: (a10-a00)(b11)
    U_strassen[5] = [[-1, 0], [1, 0]]
    V_strassen[5] = [[0, 0], [0, 1]]
    W_strassen[5] = [[0, 0], [0, 1]]  # C11 += 1
    
    # Term 7: (a01-a11)(b00+b10)
    U_strassen[6] = [[0, 1], [0, -1]]
    V_strassen[6] = [[1, 0], [1, 0]]
    W_strassen[6] = [[1, 0], [0, 0]]  # C00 += 1
    
    # Verify Strassen 2x2x2
    def verify_2x2x2(U, V, W):
        """Verify 2x2x2 decomposition."""
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        # T[i,j,k] should be 1 for all i,j,k
        return np.allclose(T, 1.0, atol=1e-9)
    
    if not verify_2x2x2(U_strassen, V_strassen, W_strassen):
        # Fallback: use schoolbook for 2x2x2
        U_strassen = np.zeros((4, 2, 2))
        V_strassen = np.zeros((4, 2, 2))
        W_strassen = np.zeros((4, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = i * 4 + j * 2 + k
                    U_strassen[col] = 0
                    V_strassen[col] = 0
                    W_strassen[col] = 0
                    U_strassen[col][i, j] = 1
                    V_strassen[col][j, k] = 1
                    W_strassen[col][k, i] = 1
        
        if not verify_2x2x2(U_strassen, V_strassen, W_strassen):
            raise ValueError("Strassen verification failed")
    
    # Now build the full 2x4x6 decomposition
    # Split m=4 into 2 blocks of size 2: m_block = 2
    # Split p=6 into 3 blocks of size 2: p_block = 2
    # Total blocks: 2 * 3 = 6
    # Each block uses 7 (or 4) terms
    
    n_blocks_m = m // 2  # 2
    n_blocks_p = p // 2  # 3
    total_blocks = n_blocks_m * n_blocks_p  # 6
    
    rank_per_block = U_strassen.shape[0]  # 7 or 4
    total_rank = total_blocks * rank_per_block  # 42 or 24
    
    U_full = np.zeros((n * m, total_rank))  # (8, 42)
    V_full = np.zeros((m * p, total_rank))  # (24, 42)
    W_full = np.zeros((p * n, total_rank))  # (12, 42)
    
    col_offset = 0
    for mb in range(n_blocks_m):
        for pb in range(n_blocks_p):
            # Block: A[:, mb*2:(mb+1)*2] @ B[mb*2:(mb+1)*2, pb*2:(pb+1)*2]
            # A rows: 0,1 (n=2)
            # A cols: mb*2, mb*2+1 (m_block=2)
            # B rows: mb*2, mb*2+1
            # B cols: pb*2, pb*2+1 (p_block=2)
            # C rows: 0,1
            # C cols: pb*2, pb*2+1
            
            for r in range(rank_per_block):
                # U: A[i, mb*2+j]
                for i in range(n):
                    for j in range(2):
                        U_full[i * m + (mb * 2 + j), col_offset] = U_strassen[r, i, j]
                
                # V: B[mb*2+j, pb*2+k]
                for j in range(2):
                    for k in range(2):
                        V_full[(mb * 2 + j) * p + (pb * 2 + k), col_offset] = V_strassen[r, j, k]
                
                # W: C[pb*2+k, i] -> W[k*n+i] where k is local (0 or 1)
                # Global k index: pb*2 + k
                for k in range(2):
                    for i in range(n):
                        W_full[(pb * 2 + k) * n + i, col_offset] = W_strassen[r, k, i]
                
                col_offset += 1
    
    # Verify full decomposition
    def verify_full(U, V, W):
        """Verify full 2x4x6 decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        U_tensor = U_arr.reshape(n, m, rank)
        V_tensor = V_arr.reshape(m, p, rank)
        W_tensor = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_tensor, V_tensor, W_tensor)
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    if not verify_full(U_full, V_full, W_full):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p  # 48
        U_sb = np.zeros((n * m, rank_schoolbook))
        V_sb = np.zeros((m * p, rank_schoolbook))
        W_sb = np.zeros((p * n, rank_schoolbook))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
        
        def to_lists(arr):
            return [[float(x) for x in row] for row in arr]
        
        return {"U": to_lists(U_sb), "V": to_lists(V_sb), "W": to_lists(W_sb)}
    
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_lists(U_full), "V": to_lists(V_full), "W": to_lists(W_full)}

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
