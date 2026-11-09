"""Matrix multiplication 4x5x5 via block decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = 4, 5, 5
    
    # Total rank estimate: 
    # 2x2x2: 7
    # 2x2x1: 2
    # 2x2x3: 9
    # 2x1x2: 1
    # 2x1x3: 1
    # 2x3x2: 8
    # 2x3x3: 10
    
    # Blocks:
    # 1,3: 2x2x2 (7)
    # 2,4: 2x2x3 (9)
    # 5,7: 2x3x2 (8)
    # 6,8: 2x3x3 (10)
    
    rank = 2*7 + 2*9 + 2*8 + 2*10
    rank = 14 + 18 + 16 + 20
    rank = 68
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Define blocks: (i_start, i_end, j_start, j_end, k_start, k_end)
    blocks = [
        (0, 2, 0, 2, 0, 2), # 1
        (0, 2, 0, 2, 2, 5), # 2
        (2, 4, 0, 2, 0, 2), # 3
        (2, 4, 0, 2, 2, 5), # 4
        (0, 2, 2, 5, 0, 2), # 5
        (0, 2, 2, 5, 2, 5), # 6
        (2, 4, 2, 5, 0, 2), # 7
        (2, 4, 2, 5, 2, 5), # 8
    ]
    
    for (i0, i1, j0, j1, k0, k1) in blocks:
        ni = i1 - i0
        nj = j1 - j0
        nk = k1 - k0
        
        # Determine rank of this sub-block
        # We use the decomposition:
        # If nj == 2 and nk == 2: Rank 7 (Strassen)
        # If nj == 2 and nk == 3: Rank 9 (Strassen 2x2x2 + 2x2x1)
        # If nj == 3 and nk == 2: Rank 8 (Strassen 2x2x2 + 2x1x2)
        # If nj == 3 and nk == 3: Rank 10 (2x2x3 + 2x1x3)
        
        if nj == 2 and nk == 2:
            sub_rank = 7
        elif nj == 2 and nk == 3:
            sub_rank = 9
        elif nj == 3 and nk == 2:
            sub_rank = 8
        elif nj == 3 and nk == 3:
            sub_rank = 10
        else:
            # Fallback schoolbook
            sub_rank = ni * nj * nk
            
        # Generate sub-block matrices
        sub_U, sub_V, sub_W = _generate_sub_block(ni, nj, nk, sub_rank)
        
        # Place into global
        for r in range(sub_rank):
            for i in range(ni):
                for j in range(nj):
                    val = sub_U[i*nj+j, r]
                    if val != 0:
                        U[(i0+i)*m + (j0+j), col+r] += val
                        
            for j in range(nj):
                for k in range(nk):
                    val = sub_V[j*nk+k, r]
                    if val != 0:
                        V[(j0+j)*p + (k0+k), col+r] += val
                        
            for k in range(nk):
                for i in range(ni):
                    val = sub_W[k*ni+i, r]
                    if val != 0:
                        W[(k0+k)*n + (i0+i), col+r] += val
                        
        col += sub_rank
        
        # Time check
        if time.time() - start_time > 100:
            break
            
    U_list = [[U[i, c] for c in range(rank)] for i in range(n*m)]
    V_list = [[V[i, c] for c in range(rank)] for i in range(m*p)]
    W_list = [[W[i, c] for c in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _generate_sub_block(ni, nj, nk, target_rank):
    """
    Generates U, V, W for a sub-block of size ni x nj x nk.
    Returns matrices of shape (ni*nj, rank), (nj*nk, rank), (nk*ni, rank).
    """
    U = np.zeros((ni*nj, target_rank), dtype=float)
    V = np.zeros((nj*nk, target_rank), dtype=float)
    W = np.zeros((nk*ni, target_rank), dtype=float)
    
    col = 0
    
    # Case 1: 2x2x2 (Strassen)
    if ni == 2 and nj == 2 and nk == 2:
        sU, sV, sW = _strassen_222()
        U[:, col:col+7] = sU
        V[:, col:col+7] = sV
        W[:, col:col+7] = sW
        col += 7
        return U, V, W
        
    # Case 2: 2x2x3 (Strassen 2x2x2 + 2x2x1)
    if ni == 2 and nj == 2 and nk == 3:
        # Part 1: 2x2x2 for k=0,1
        sU, sV, sW = _strassen_222()
        U[:, col:col+7] = sU
        # V is 2x3. We only fill first 2 cols (k=0,1)
        V[:, col:col+7] = sV[:, :2] 
        # W is 3x2. We only fill first 2 rows (k=0,1)
        W[:2, col:col+7] = sW
        col += 7
        
        # Part 2: 2x2x1 for k=2
        # Rank 2
        # C_i2 = A_i0 B_02 + A_i1 B_12
        # Term 1: A_i0 B_02
        U[0, col] = 1 # A_i0
        V[0*2+2, col] = 1 # B_02 (j=0, k=2)
        W[2, col] = 1 # W_k_i, k=2, i=0? 
        # Wait, W index is k*ni + i.
        # For term A_i0 B_02, contribution to C_i2 is 1.
        # W[2*2+0, col] = 1? No, W is shared for all i?
        # Rank 1 term: U_i = delta_{i0}, V_jk = delta_{j0}delta_{k2}, W_ki = delta_{k2}delta_{i0}?
        # Let's use the standard basis.
        # T1: U[0]=1, V[2]=1 (j=0,k=2), W[4]=1 (k=2,i=0)?
        # C_02 = U_0 V_2 W_4 = 1*1*1 = 1.
        # C_12 = U_1 V_2 W_6 = 0*1*1 = 0.
        # We need C_12 = A_10 B_02.
        # So we need another term for i=1?
        # Yes, 2x2x1 has rank 2.
        # Term 1: i=0. U[0]=1, V[0*2+2]=1, W[2*2+0]=1.
        # Term 2: i=1. U[1]=1, V[0*2+2]=1, W[2*2+1]=1.
        # Wait, V is same?
        # T1: U=(1,0), V=(0,0,1) (for j=0,k=2), W=(0,0,1,0) (for k=2,i=0).
        # T2: U=(0,1), V=(0,0,1) (for j=0,k=2), W=(0,0,0,1) (for k=2,i=1).
        # This handles A_i0 B_02.
        # We also need A_i1 B_12.
        # T3: U=(1,0) for A_i0? No, U is for A.
        # U[0] corresponds to A_00? No, U index is i*nj+j.
        # U[0*2+0] = A_00. U[0*2+1] = A_01.
        # U[1*2+0] = A_10. U[1*2+1] = A_11.
        
        # Let's just use schoolbook for the 2x2x1 part to be safe.
        # 2x2x1 rank 2? No, schoolbook is 4.
        # But we can do it with 2?
        # C_i2 = A_i0 B_02 + A_i1 B_12.
        # M1 = A_i0 B_02.
        # M2 = A_i1 B_12.
        # C_i2 = M1 + M2.
        # M1 is rank 1?
        # U_i = A_i0. V_jk = B_02 if j=0,k=2 else 0. W_ki = 1 if k=2 else 0?
        # No, W must be independent of A, B.
        # The tensor is fixed.
        # T_{i, j, k} = 1 if j=0, k=2? No.
        # The sub-tensor for j=0, k=2 is T_{i, 0, 2} = 1.
        # This is a matrix of all 1s (2x1).
        # Rank of a 2x1 all-ones matrix is 1.
        # So T_{i, 0, 2} can be represented by 1 rank-1 term?
        # U_i = 1. V_{0,2} = 1. W_{2,i} = 1.
        # Sum_r U_i V_{0,2} W_{2,i} = 1 * 1 * 1 = 1.
        # Yes!
        # So for each (j,k) pair, if the slice T_{i, j, k} is all 1s, it's rank 1.
        # For 2x2x1 (j=0,1; k=2):
        # Slice j=0, k=2: All 1s. Rank 1.
        # Slice j=1, k=2: All 1s. Rank 1.
        # Total Rank 2.
        
        # So for 2x2x1 part (k=2):
        # Term 1: j=0, k=2.
        U[:, col] = 1.0 # All i
        V[0*2+2, col] = 1.0 # j=0, k=2
        W[2, col] = 1.0 # k=2, i=0? No, W index k*ni+i.
        # W[2*2+0] = 1. W[2*2+1] = 1.
        W[4, col] = 1.0
        W[5, col] = 1.0
        col += 1
        
        # Term 2: j=1, k=2.
        U[:, col] = 1.0
        V[1*2+2, col] = 1.0 # j=1, k=2
        W[4, col] = 1.0
        W[5, col] = 1.0
        col += 1
        
        return U, V, W
        
    # Case 3: 2x3x2
    if ni == 2 and nj == 3 and nk == 2:
        # Part 1: 2x2x2 (j=0,1; k=0,1)
        sU, sV, sW = _strassen_222()
        U[:, col:col+7] = sU
        V[:, col:col+7] = sV[:, :2] # j=0,1
        W[:, col:col+7] = sW
        col += 7
        
        # Part 2: 2x1x2 (j=2; k=0,1)
        # Slice j=2, k=0: All 1s. Rank 1.
        U[:, col] = 1.0
        V[2*2+0, col] = 1.0 # j=2, k=0
        W[0, col] = 1.0 # k=0, i=0
        W[1, col] = 1.0 # k=0, i=1
        col += 1
        
        # Slice j=2, k=1: All 1s. Rank 1.
        U[:, col] = 1.0
        V[2*2+1, col] = 1.0 # j=2, k=1
        W[2, col] = 1.0 # k=1, i=0
        W[3, col] = 1.0 # k=1, i=1
        col += 1
        
        return U, V, W
        
    # Case 4: 2x3x3
    if ni == 2 and nj == 3 and nk == 3:
        # Part 1: 2x2x3 (j=0,1; k=0,1,2)
        # Reuse logic from Case 2?
        # 2x2x2 (j=0,1; k=0,1) Rank 7
        sU, sV, sW = _strassen_222()
        U[:, col:col+7] = sU
        V[:, col:col+7] = sV[:, :2]
        W[:2, col:col+7] = sW
        col += 7
        
        # 2x2x1 (j=0,1; k=2) Rank 2
        # Slice j=0, k=2: Rank 1
        U[:, col] = 1.0
        V[0*3+2, col] = 1.0
        W[2, col] = 1.0
        W[3, col] = 1.0
        col += 1
        
        # Slice j=1, k=2: Rank 1
        U[:, col] = 1.0
        V[1*3+2, col] = 1.0
        W[2, col] = 1.0
        W[3, col] = 1.0
        col += 1
        
        # Part 2: 2x1x3 (j=2; k=0,1,2)
        # Slice j=2, k=0: Rank 1
        U[:, col] = 1.0
        V[2*3+0, col] = 1.0
        W[0, col] = 1.0
        W[1, col] = 1.0
        col += 1
        
        # Slice j=2, k=1: Rank 1
        U[:, col] = 1.0
        V[2*3+1, col] = 1.0
        W[2, col] = 1.0
        W[3, col] = 1.0
        col += 1
        
        # Slice j=2, k=2: Rank 1
        U[:, col] = 1.0
        V[2*3+2, col] = 1.0
        W[4, col] = 1.0
        W[5, col] = 1.0
        col += 1
        
        return U, V, W

    # Fallback
    return U, V, W

def _strassen_222():
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 3] = 1; W[1, 0] = -1; W[2, 3] = -1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W
# EVOLVE-BLOCK-END
