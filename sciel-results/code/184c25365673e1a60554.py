"""Strassen-based rank reduction for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    The tensor T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,9)
    must be decomposed as:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Returns dict with U, V, W as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook baseline (rank 144)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def is_valid(U, V, W, rank):
        """Check if the decomposition exactly reproduces T[i,j,k]=1 for all i,j,k."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Strassen 2x2x2 coefficients
    # u, v, w are 4-vectors for 2x2 matrices (row-major: [00, 01, 10, 11])
    strassen_u = [
        [1, 0, 0, 1],  # u1
        [0, 0, 1, 1],  # u2
        [1, 0, 0, 0],  # u3
        [0, 0, 0, 1],  # u4
        [1, 1, 0, 0],  # u5
        [1, 0, 1, 0],  # u6
        [0, 1, 0, 1],  # u7
    ]
    strassen_v = [
        [1, 0, 0, 1],  # v1
        [1, 0, 0, 0],  # v2
        [0, 1, 1, 0],  # v3
        [1, 1, 0, 0],  # v4
        [0, 0, 0, 1],  # v5
        [1, 0, 1, 0],  # v6
        [0, 1, 0, 1],  # v7
    ]
    strassen_w = [
        [1, 0, 0, 1],  # w1
        [0, 1, 0, 0],  # w2
        [1, 0, 0, 0],  # w3
        [0, 1, 0, 0],  # w4
        [1, 0, 0, 0],  # w5
        [0, 1, 0, 0],  # w6
        [1, 0, 0, 0],  # w7
    ]
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    # 4x4 = 2x2 blocks, so we have 4 block positions: (0,0), (0,1), (1,0), (1,1)
    # Each 2x2 multiplication uses 7 Strassen terms
    # Total: 4 * 7 = 28 terms for the 4x4x4 core
    
    # Actually, for 4x4 * 4x4 -> 4x4, we can use block Strassen:
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]] (each 2x2)
    # C = A*B, each Cij is 2x2
    
    # Using Strassen on the 2x2 block level gives 7 block multiplications,
    # each of which is a 2x2 * 2x2 -> 2x2, which itself needs 7 Strassen terms.
    # Total: 7 * 7 = 49 terms.
    
    # Let's construct this explicitly.
    # Block indices: bi, bj in {0,1} for row/col blocks of the 4x4
    
    # For each of the 7 Strassen block-level terms (s=0..6):
    #   The block-level U coefficient tells us which A-blocks to combine
    #   The block-level V coefficient tells us which B-blocks to combine
    #   The block-level W coefficient tells us which C-blocks to update
    
    # Within each block multiplication, we use the 7 Strassen 2x2 terms.
    
    U_44 = []
    V_44 = []
    W_44 = []
    
    for s in range(7):  # block-level Strassen term
        for t in range(7):  # within-block Strassen term
            u_vec = [0] * 16  # 4x4 row-major
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            # Block-level coefficients
            u_block = strassen_u[s]  # which A-blocks
            v_block = strassen_v[s]  # which B-blocks
            w_block = strassen_w[s]  # which C-blocks
            
            # Within-block coefficients
            u_in = strassen_u[t]  # which rows within A-block
            v_in = strassen_v[t]  # which rows within B-block  
            w_in = strassen_w[t]  # which cols within C-block
            
            # A-block positions: (0,0)=A00, (0,1)=A01, (1,0)=A10, (1,1)=A11
            # Block (bi, bj) occupies rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
            # For U (A side): index = i*4 + j where i is row, j is col
            # For V (B side): index = j*4 + k where j is row, k is col
            # For W (C side): index = k*4 + i where k is col, i is row
            
            for bi in range(2):
                for bj in range(2):
                    # A-block (bi, bj) coefficient from block-level u
                    a_coeff = u_block[bi * 2 + bj]
                    if a_coeff == 0:
                        continue
                    
                    # Within this A-block, the 2x2 Strassen term t has u_in coefficients
                    # u_in is for a 2x2 matrix: [00, 01, 10, 11]
                    for li in range(2):  # local row
                        for lj in range(2):  # local col
                            # Local index in 2x2: li*2 + lj
                            local_idx = li * 2 + lj
                            in_coeff = u_in[local_idx]
                            if in_coeff == 0:
                                continue
                            
                            # Global index in 4x4
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            gidx = gi * 4 + gj
                            u_vec[gidx] += a_coeff * in_coeff
            
            for bi in range(2):
                for bj in range(2):
                    # B-block (bi, bj) coefficient from block-level v
                    b_coeff = v_block[bi * 2 + bj]
                    if b_coeff == 0:
                        continue
                    
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            in_coeff = v_in[local_idx]
                            if in_coeff == 0:
                                continue
                            
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            gidx = gi * 4 + gj
                            v_vec[gidx] += b_coeff * in_coeff
            
            for bi in range(2):
                for bj in range(2):
                    # C-block (bi, bj) coefficient from block-level w
                    c_coeff = w_block[bi * 2 + bj]
                    if c_coeff == 0:
                        continue
                    
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            in_coeff = w_in[local_idx]
                            if in_coeff == 0:
                                continue
                            
                            # For W: index = k*4 + i where k is col, i is row
                            # C-block (bi, bj): row block bi, col block bj
                            # Global row = 2*bi + li, global col = 2*bj + lj
                            gi_row = 2 * bi + li
                            gi_col = 2 * bj + lj
                            gidx = gi_col * 4 + gi_row
                            w_vec[gidx] += c_coeff * in_coeff
            
            U_44.append(u_vec)
            V_44.append(v_vec)
            W_44.append(w_vec)
    
    # Verify 4x4x4 Strassen decomposition
    U_44_np = np.array(U_44, dtype=float).T  # (16, 49)
    V_44_np = np.array(V_44, dtype=float).T
    W_44_np = np.array(W_44, dtype=float).T
    
    valid_44 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0.0
                for r in range(49):
                    val += U_44_np[i*4+j, r] * V_44_np[j*4+k, r] * W_44_np[k*4+i, r]
                if abs(val - 1.0) > 1e-9:
                    valid_44 = False
                    break
            if not valid_44:
                break
        if not valid_44:
            break
    
    if not valid_44:
        # Fallback to schoolbook
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Now extend to 4x4 * 4x9
    # p = 9 = 4 + 4 + 1
    # For k in [0,4): use the 4x4x4 Strassen decomposition
    # For k in [4,8): use another copy (shifted)
    # For k in [8,9): use schoolbook for that single column
    
    # U has shape (16, rank), V has shape (36, rank), W has shape (36, rank)
    # U[i*m+j, r] where i in [0,4), j in [0,4)
    # V[j*p+k, r] where j in [0,4), k in [0,9)
    # W[k*n+i, r] where k in [0,9), i in [0,4)
    
    # For the first 4 k-values (k=0..3), use Strassen 4x4x4
    # For k=4..7, use another Strassen 4x4x4 (independent)
    # For k=8, use schoolbook (4 terms)
    
    rank_total = 49 + 49 + 1  # 99
    
    U_full = [[0] * rank_total for _ in range(16)]
    V_full = [[0] * rank_total for _ in range(36)]
    W_full = [[0] * rank_total for _ in range(36)]
    
    # First block: k=0..3, use Strassen 4x4x4
    for r in range(49):
        for i in range(16):
            U_full[i][r] = U_44_np[i, r]
        for j in range(4):
            for k in range(4):
                V_full[j * 9 + k][r] = V_44_np[j * 4 + k, r]
        for k in range(4):
            for i in range(4):
                W_full[k * 4 + i][r] = W_44_np[k * 4 + i, r]
    
    # Second block: k=4..7, use another Strassen 4x4x4
    offset = 49
    for r in range(49):
        for i in range(16):
            U_full[i][offset + r] = U_44_np[i, r]
        for j in range(4):
            for k in range(4):
                V_full[j * 9 + (4 + k)][offset + r] = V_44_np[j * 4 + k, r]
        for k in range(4):
            for i in range(4):
                W_full[(4 + k) * 4 + i][offset + r] = W_44_np[k * 4 + i, r]
    
    # Third block: k=8, rank-1 term (all-ones tensor slice)
    offset2 = 98
    for i in range(4):
        for j in range(4):
            U_full[i * 4 + j][offset2] = 1
    for j in range(4):
        V_full[j * 9 + 8][offset2] = 1
    for i in range(4):
        W_full[8 * 4 + i][offset2] = 1
    
    # Verify full decomposition
    if is_valid(U_full, V_full, W_full, rank_total):
        best_rank = rank_total
        best_U = U_full
        best_V = V_full
        best_W = W_full
    
    # Final validation
    if not is_valid(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        best_rank = n * m * p
        best_U = [[0] * best_rank for _ in range(n * m)]
        best_V = [[0] * best_rank for _ in range(m * p)]
        best_W = [[0] * best_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][col] = 1
                    best_V[j * p + k][col] = 1
                    best_W[k * n + i][col] = 1
                    col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
