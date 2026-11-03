"""Improved program using Strassen-based block decomposition for 3x4x7."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas for 2x2 matrix multiplication
    # C = A * B, A is 2x2, B is 2x2, C is 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U: A is 2x2, indexed as A[i,j] -> row i*2+j in U
    # V: B is 2x2, indexed as B[j,k] -> row j*2+k in V
    # W: C is 2x2, indexed as C[k,i] -> row k*2+i in W (transpose convention)
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 = A[0,0] -> U[0,0], A22 = A[1,1] -> U[3,0]
    # B11 = B[0,0] -> V[0,0], B22 = B[1,1] -> V[3,0]
    # C contribution: C = M1 contributes to C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] -= M1
    # Wait, Strassen:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for M1: contributes +1 to C11, +1 to C22, -1 to C12 (no, M5 does that), let me be careful.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W[k*2+i, c] where C[k,i] is the entry
    # C11 = C[0,0] -> W[0*2+0] = W[0]
    # C12 = C[0,1] -> W[0*2+1] = W[1]
    # C21 = C[1,0] -> W[1*2+0] = W[2]
    # C22 = C[1,1] -> W[1*2+1] = W[3]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: A11 + A22, B11 + B22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22
    
    # M2: A21 + A22, B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21
    
    # M3: A11, B21 - B22
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 and C22
    
    # M4: A22, B12 - B11
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1  # C11 and C21
    
    # M5: A11 + A12, B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1  # C11 and C12
    
    # M6: A21 - A11, B11 + B12
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22
    
    # M7: A12 - A22, B21 + B22
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1  # C11
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = int(n), int(m), int(p)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # This means the 3-tensor T_{ijk} = sum_c U_{im+j,c} V_{jp+k,c} W_{kn+i,c} should equal 1 for all i,j,k
    
    # For each (i,j,k), check the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False, val
    return True, 0.0

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = 150.0 - 10.0  # 140s deadline with 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p  # 84
    
    # Try to use Strassen for 2x2x2 blocks to reduce rank
    # Split 3x4x7 into blocks
    # n=3: 2+1
    # m=4: 2+2
    # p=7: 2+2+2+1
    
    # Strategy: use Strassen where we have 2x2x2 blocks
    # For the 2x2x2 blocks, rank 7 instead of 8 (saves 1 per block)
    # But we need to be careful about the mixed blocks (1x2x2, 2x1x2, 2x2x1, 1x2x1, etc.)
    
    # Let's try a different approach: direct search for a lower-rank decomposition
    # using the Strassen idea on the 2x2 parts
    
    # Actually, let me try to construct a decomposition by splitting:
    # A (3x4) = [A_top (2x4); A_bot (1x4)]
    # B (4x7) = [B_left (4x6) | B_right (4x1)]  where B_left is 4x6, B_right is 4x1
    # C (3x7) = A * B
    
    # C_top (2x7) = A_top (2x4) * B (4x7)
    # C_bot (1x7) = A_bot (1x4) * B (4x7)
    
    # For C_top: 2x4x7. Split B into 2x2x2 blocks:
    # B = [B1 (4x2) | B2 (4x2) | B3 (4x2) | B4 (4x1)]
    # C_top = A_top * [B1 | B2 | B3 | B4]
    # C_top1 (2x2) = A_top1 (2x2) * B1 (2x2) + A_top2 (2x2) * B1_bottom (2x2) ... this gets complicated
    
    # Let me try a cleaner block approach:
    # Split n=3 into 2+1, m=4 into 2+2, p=7 into 2+2+2+1
    # This gives 2*2*4 = 16 blocks, each of various sizes
    
    # For 2x2x2 blocks: use Strassen (rank 7 instead of 8)
    # For other blocks: use schoolbook
    
    # Number of 2x2x2 blocks: (n_blocks=2 choices for 2x1 or 1x2... wait)
    # n=3 = 2+1: 2 blocks (size 2, size 1)
    # m=4 = 2+2: 2 blocks (size 2, size 2)
    # p=7 = 2+2+2+1: 4 blocks (size 2, size 2, size 2, size 1)
    
    # Total blocks: 2*2*4 = 16
    # 2x2x2 blocks: 1*1*3 = 3 blocks (where n-block is size 2, m-block is size 2, p-block is size 2)
    #   Wait: n has one block of size 2, m has two blocks of size 2, p has three blocks of size 2
    #   So 2x2x2 blocks: 1 * 2 * 3 = 6 blocks
    # 2x2x1 blocks: 1 * 2 * 1 = 2 blocks
    # 1x2x2 blocks: 1 * 2 * 3 = 6 blocks
    # 1x2x1 blocks: 1 * 2 * 1 = 2 blocks
    # Total: 6 + 2 + 6 + 2 = 16 blocks ✓
    
    # Schoolbook rank: 6*8 + 2*2 + 6*2 + 2*1 = 48 + 4 + 12 + 2 = 66? No, that's not right.
    # Schoolbook rank for each block is n_i * m_j * p_k
    # 2x2x2: 8 each, 6 blocks: 48
    # 2x2x1: 4 each, 2 blocks: 8
    # 1x2x2: 4 each, 6 blocks: 24
    # 1x2x1: 2 each, 2 blocks: 4
    # Total schoolbook: 48 + 8 + 24 + 4 = 84 ✓
    
    # With Strassen on 2x2x2 blocks: 6*7 + 8 + 24 + 4 = 42 + 8 + 24 + 4 = 78
    # That's rank 78, better than 84!
    
    # Let's try to construct this.
    
    # Define block boundaries
    n_splits = [0, 2, 3]  # sizes: 2, 1
    m_splits = [0, 2, 4]  # sizes: 2, 2
    p_splits = [0, 2, 4, 6, 7]  # sizes: 2, 2, 2, 1
    
    n_sizes = [2, 1]
    m_sizes = [2, 2]
    p_sizes = [2, 2, 2, 1]
    
    # For each block, determine if we can use Strassen
    # Strassen works for 2x2x2 blocks
    
    rank = 0
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for ni in range(2):  # n block index
        for mj in range(2):  # m block index
            for pk in range(4):  # p block index
                n_i = n_sizes[ni]
                m_j = m_sizes[mj]
                p_k = p_sizes[pk]
                
                # Determine indices
                n_start = n_splits[ni]
                m_start = m_splits[mj]
                p_start = p_splits[pk]
                
                if n_i == 2 and m_j == 2 and p_k == 2:
                    # Use Strassen: rank 7
                    Us, Vs, Ws = _strassen_2x2x2()
                    U_blocks.append((ni, mj, pk, Us))
                    V_blocks.append((ni, mj, pk, Vs))
                    W_blocks.append((ni, mj, pk, Ws))
                    rank += 7
                else:
                    # Schoolbook for this block
                    block_rank = n_i * m_j * p_k
                    Us = np.zeros((n_i * m_j, block_rank))
                    Vs = np.zeros((m_j * p_k, block_rank))
                    Ws = np.zeros((p_k * n_i, block_rank))
                    col = 0
                    for i in range(n_i):
                        for j in range(m_j):
                            for k in range(p_k):
                                Us[i * m_j + j, col] = 1
                                Vs[j * p_k + k, col] = 1
                                Ws[k * n_i + i, col] = 1
                                col += 1
                    U_blocks.append((ni, mj, pk, Us))
                    V_blocks.append((ni, mj, pk, Vs))
                    W_blocks.append((ni, mj, pk, Ws))
                    rank += block_rank
    
    # Now assemble the full matrices
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)
    
    U_full = np.zeros((n * m, rank))
    V_full = np.zeros((m * p, rank))
    W_full = np.zeros((p * n, rank))
    
    col_offset = 0
    for (ni, mj, pk, Us), (_, _, _, Vs), (_, _, _, Ws) in zip(U_blocks, V_blocks, W_blocks):
        block_rank = Us.shape[1]
        
        # Place U block
        n_i = n_sizes[ni]
        m_j = m_sizes[mj]
        p_k = p_sizes[pk]
        
        n_start = n_splits[ni]
        m_start = m_splits[mj]
        p_start = p_splits[pk]
        
        # U rows: (n_i * m_j), starting at row (n_start * m + m_start)
        u_row_start = n_start * m + m_start
        for i in range(n_i):
            for j in range(m_j):
                u_row = u_row_start + i * m_j + j
                U_full[u_row, col_offset:col_offset + block_rank] = Us[i * m_j + j, :]
        
        # V rows: (m_j * p_k), starting at row (m_start * p + p_start)
        v_row_start = m_start * p + p_start
        for j in range(m_j):
            for k in range(p_k):
                v_row = v_row_start + j * p_k + k
                V_full[v_row, col_offset:col_offset + block_rank] = Vs[j * p_k + k, :]
        
        # W rows: (p_k * n_i), starting at row (p_start * n + n_start)
        w_row_start = p_start * n + n_start
        for k in range(p_k):
            for i in range(n_i):
                w_row = w_row_start + k * n_i + i
                W_full[w_row, col_offset:col_offset + block_rank] = Ws[k * n_i + i, :]
        
        col_offset += block_rank
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Verify
    valid, val = _verify_decomposition(n, m, p, U_full, V_full, W_full)
    if valid:
        return result
    
    # If verification failed, return schoolbook
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
