"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2x2():
    """
    Return rank-4 decomposition for 2x2x2 matrix multiplication.
    C[i,k] = sum_j A[i,j]*B[j,k]
    U shape (4, 4), V shape (4, 4), W shape (4, 4)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products, but we need rank-4 for 2x2x2? 
    # Actually rank(2x2x2 matmul) = 7 (Strassen). Let me use rank 7.
    # Wait, the tensor rank of 2x2x2 matrix multiplication is 7.
    # So we need rank 7, not 4.
    
    # Let me reconsider. The rank of the 2x2x2 matrix multiplication tensor is 7.
    # So each 2x2x2 sub-problem needs rank 7.
    
    # Strassen's 7 products:
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A01) * B11
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A10) * (B00 + B11)
    # M6 = (A01 - A10) * (B10 + B11)
    # M7 = (A00 - A01) * (B01 + B11)
    #
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M7
    # C11 = M5 + M1 - M3 + M4
    #
    # U rows: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3
    # V rows: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
    # W rows: C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,2,3 (k*2+i)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = A00 * (B01 - B11)
    U[0, 0] = 1  # A[0,0]
    V[1, 0] = 1  # B[0,1]
    V[3, 0] = -1 # B[1,1]
    
    # M2 = (A00 + A01) * B11
    U[0, 1] = 1  # A[0,0]
    U[1, 1] = 1  # A[0,1]
    V[3, 1] = 1  # B[1,1]
    
    # M3 = (A10 + A11) * B00
    U[2, 2] = 1  # A[1,0]
    U[3, 2] = 1  # A[1,1]
    V[0, 2] = 1  # B[0,0]
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1 # B[0,0]
    
    # M5 = (A00 + A10) * (B00 + B11)
    U[0, 4] = 1  # A[0,0]
    U[2, 4] = 1  # A[1,0]
    V[0, 4] = 1  # B[0,0]
    V[3, 4] = 1  # B[1,1]
    
    # M6 = (A01 - A10) * (B10 + B11)
    U[1, 5] = 1  # A[0,1]
    U[2, 5] = -1 # A[1,0]
    V[2, 5] = 1  # B[1,0]
    V[3, 5] = 1  # B[1,1]
    
    # M7 = (A00 - A01) * (B01 + B11)
    U[0, 6] = 1  # A[0,0]
    U[1, 6] = -1 # A[0,1]
    V[1, 6] = 1  # B[0,1]
    V[3, 6] = 1  # B[1,1]
    
    # C00 = M5 + M4 - M2 + M6 -> W[0, r]
    # C01 = M1 + M2 -> W[1, r]
    # C10 = M3 + M7 -> W[2, r]
    # C11 = M5 + M1 - M3 + M4 -> W[3, r]
    W[0, 4] = 1  # M5
    W[0, 3] = 1  # M4
    W[0, 1] = -1 # M2
    W[0, 5] = 1  # M6
    
    W[1, 0] = 1  # M1
    W[1, 1] = 1  # M2
    
    W[2, 2] = 1  # M3
    W[2, 6] = 1  # M7
    
    W[3, 4] = 1  # M5
    W[3, 0] = 1  # M1
    W[3, 2] = -1 # M3
    W[3, 3] = 1  # M4
    
    return U, V, W


def _rank_decomp_2x1x2():
    """
    Rank decomposition for 2x1x2 matrix multiplication.
    A is 2x1, B is 1x2, C is 2x2.
    C[i,k] = A[i,0] * B[0,k]
    This is rank 4: each (i,k) pair needs its own term.
    U shape (2, 4), V shape (2, 4), W shape (4, 4)
    Indexing: U[i*1+j, r] = U[i, r], V[j*2+k, r] = V[k, r], W[k*2+i, r]
    """
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    
    for r in range(4):
        i = r % 2
        k = r // 2
        U[i, r] = 1
        V[k, r] = 1
        W[k*2 + i, r] = 1
    
    return U, V, W


def _decompose_2x5x2():
    """
    Decompose 2x5x2 matrix multiplication.
    A is 2x5, B is 5x2, C is 2x2.
    Split middle dim 5 into (2,2,1).
    Sub-tensors: 2x2x2 (rank 7), 2x2x2 (rank 7), 2x1x2 (rank 4).
    Total rank: 18.
    
    Returns U (2*5, rank), V (5*2, rank), W (2*2, rank)
    U index: i*5+j, V index: j*2+k, W index: k*2+i
    """
    s1U, s1V, s1W = _strassen_2x2x2()
    s2U, s2V, s2W = _strassen_2x2x2()
    s3U, s3V, s3W = _rank_decomp_2x1x2()
    
    total_rank = 7 + 7 + 4
    U = np.zeros((10, total_rank))
    V = np.zeros((10, total_rank))
    W = np.zeros((4, total_rank))
    
    # Sub-tensor 1: j in {0,1}, i in {0,1}, k in {0,1}
    # U rows for i=0,1 and j=0,1: indices i*5+j = 0,1,5,6
    # V rows for j=0,1 and k=0,1: indices j*2+k = 0,1,2,3
    # W rows: k*2+i
    
    for r in range(7):
        for ui in range(4):
            i, j = ui // 2, ui % 2
            U[i*5 + j, r] = s1U[ui, r]
        for vi in range(4):
            j, k = vi // 2, vi % 2
            V[j*2 + k, r] = s1V[vi, r]
        for wi in range(4):
            k, i = wi // 2, wi % 2
            W[wi, r] = s1W[wi, r]
    
    # Sub-tensor 2: j in {2,3}, i in {0,1}, k in {0,1}
    # U rows: i*5+j where j in {2,3}: indices i*5+2, i*5+3
    # V rows: j*2+k where j in {2,3}: indices 4,5,6,7
    offset = 7
    for r in range(7):
        for ui in range(4):
            i, j_local = ui // 2, ui % 2
            j = 2 + j_local
            U[i*5 + j, offset + r] = s2U[ui, r]
        for vi in range(4):
            j_local, k = vi // 2, vi % 2
            j = 2 + j_local
            V[j*2 + k, offset + r] = s2V[vi, r]
        for wi in range(4):
            k, i = wi // 2, wi % 2
            W[wi, offset + r] = s2W[wi, r]
    
    # Sub-tensor 3: j in {4}, i in {0,1}, k in {0,1}
    # U rows: i*5+4: indices 4, 9
    # V rows: 4*2+k = 8+k: indices 8, 9
    offset2 = 14
    for r in range(4):
        for ui in range(2):
            i = ui
            U[i*5 + 4, offset2 + r] = s3U[ui, r]
        for vi in range(2):
            k = vi
            V[8 + k, offset2 + r] = s3V[vi, r]
        for wi in range(4):
            k, i = wi // 2, wi % 2
            W[wi, offset2 + r] = s3W[wi, r]
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Partition 4x5x6 into 2x5x2 blocks:
    # 2 blocks along A's row dimension (rows 0-1, rows 2-3)
    # 3 blocks along B's column dimension (cols 0-1, cols 2-3, cols 4-5)
    # Total: 2 * 3 = 6 blocks, each rank 18, total rank 108
    
    # Wait, that gives 108. Let me reconsider.
    # Actually, I should try to be smarter.
    
    # Alternative: partition differently.
    # 4 = 2+2, 6 = 2+2+2
    # So 2 blocks in row dim, 3 blocks in col dim = 6 blocks of 2x5x2
    # Each 2x5x2 has rank 18, total = 6*18 = 108
    
    # Can I do better? Let me think about whether I can reduce the rank.
    
    # Actually, let me just go with 108 for now, which is better than 120.
    
    total_rank = 6 * 18
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for block_i in range(2):  # blocks along A rows
        for block_k in range(3):  # blocks along B cols
            U_sub, V_sub, W_sub = _decompose_2x5x2()
            
            # Place into global matrices
            # A rows: block_i*2 + i_local, so global row = (block_i*2 + i_local)*m + j
            # B cols: block_k*2 + k_local, so global col = j*(p) + (block_k*2 + k_local)
            # C: row k, col i -> global: (block_k*2 + k_local)*n + (block_i*2 + i_local)
            
            for r in range(18):
                # U: local index i_local*5 + j -> global (block_i*2 + i_local)*m + j
                for ui in range(10):
                    i_local, j = ui // 5, ui % 5
                    global_u_row = (block_i * 2 + i_local) * m + j
                    U[global_u_row, col_offset + r] = U_sub[ui, r]
                
                # V: local index j*2 + k_local -> global j*p + (block_k*2 + k_local)
                for vi in range(10):
                    j, k_local = vi // 2, vi % 2
                    global_v_row = j * p + (block_k * 2 + k_local)
                    V[global_v_row, col_offset + r] = V_sub[vi, r]
                
                # W: local index k_local*2 + i_local -> global (block_k*2 + k_local)*n + (block_i*2 + i_local)
                for wi in range(4):
                    k_local, i_local = wi // 2, wi % 2
                    global_w_row = (block_k * 2 + k_local) * n + (block_i * 2 + i_local)
                    W[global_w_row, col_offset + r] = W_sub[wi, r]
            
            col_offset += 18
    
    # Convert to list of lists
    def to_list(mat):
        result = []
        for row in mat:
            result.append([round(float(x), 12) for x in row])
        return result
    
    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
        return lst
    
    U_out = clean_list(to_list(U))
    V_out = clean_list(to_list(V))
    W_out = clean_list(to_list(W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

