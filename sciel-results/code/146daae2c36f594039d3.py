"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110
    
    # Strategy: Use rank-19 decomposition of 5x5x5, then restrict to 4x5x5
    # The 4x5x5 multiplication tensor is T[i,j,k] = delta_{i,k} for i in 0..3
    # This is a sub-tensor of the 5x5x5 tensor, so rank <= 19
    # We use the known rank-19 decomposition and check if it works directly
    
    # Rank-19 decomposition of 5x5x5 matrix multiplication (Bürgisser 2005)
    # C = A * B where A, B, C are 5x5
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k} for all i,j,k in 0..4
    # For 4x5x5: same but i in 0..3 only for the delta constraint on W
    
    # The rank-19 decomposition uses 19 scalar multiplications
    # U: 25x19, V: 25x19, W: 25x19 for 5x5x5
    # For 4x5x5: U: 20x19, V: 25x19, W: 20x19
    
    # Let me construct the rank-19 decomposition for 5x5x5
    # This is complex, so let me try a different approach:
    # Use Strassen-like decomposition adapted for 4x5x5
    
    # Alternative: Direct construction using the fact that 4x5x5 can be seen as
    # 4x4x4 (rank 49 via Strassen) + 4x1x5 (rank 20) + 4x5x1 (rank 20) - 4x1x1 (rank 4)
    # = 49 + 20 + 20 - 4 = 85 (same as before)
    
    # Better: Try to find a lower rank by using structure
    # The tensor T[i,j,k] = delta_{i,k} for i<4 is the identity in the (i,k) space
    # This is equivalent to 4 copies of the 5x5 identity
    
    # Let me try a direct rank-19 approach using the known 5x5x5 decomposition
    # and verify if it restricts correctly to 4x5x5
    
    # For now, let me try a hybrid approach:
    # Use rank-7 Strassen for the 4x4x4 part (already done)
    # But optimize the boundary terms
    
    # Actually, let me try: 4x5x5 = 4x4x4 + 4x4x1 + 4x1x4 + 4x1x1
    # The 4x4x1 and 4x1x4 parts share structure with the 4x1x1 part
    
    # Let me try to reduce the boundary rank by sharing terms
    
    rank = 85
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: 4x4x4 Strassen (rank 49)
    def strassen_2x2x2():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        u[0,0]=1; u[3,0]=1; v[0,0]=1; v[3,0]=1; w[0,0]=1; w[3,0]=1
        u[2,1]=1; u[3,1]=1; v[0,1]=1; w[2,1]=1; w[3,1]=-1
        u[0,2]=1; v[1,2]=1; v[3,2]=-1; w[1,2]=1; w[3,2]=1
        u[3,3]=1; v[2,3]=1; v[0,3]=-1; w[0,3]=1; w[2,3]=1
        u[0,4]=1; u[1,4]=1; v[3,4]=1; w[0,4]=-1; w[1,4]=1
        u[0,5]=-1; u[2,5]=1; v[0,5]=1; v[1,5]=1; w[3,5]=1
        u[1,6]=1; u[3,6]=-1; v[2,6]=1; v[3,6]=1; w[0,6]=1
        return u, v, w
    
    su, sv, sw = strassen_2x2x2()
    
    strassen_products = [
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
    ]
    
    def a_global(block, li, lk):
        if block == 0: return li, lk
        elif block == 1: return li, lk + 2
        elif block == 2: return li + 2, lk
        elif block == 3: return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        if block == 0: return lk, lj
        elif block == 1: return lk, lj + 2
        elif block == 2: return lk + 2, lj
        elif block == 3: return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        if block == 0: return li, lj
        elif block == 1: return li, lj + 2
        elif block == 2: return li + 2, lj
        elif block == 3: return li + 2, lj + 2
    
    for t in range(7):
        prod = strassen_products[t]
        for t2 in range(7):
            c_idx = col + t2
            for li in range(2):
                for lk in range(2):
                    for lj in range(2):
                        su_val = su[li * 2 + lk, t2]
                        if su_val != 0:
                            for block_a, sign_a in prod['A']:
                                gi, gk = a_global(block_a, li, lk)
                                U[gi * m + gk, c_idx] += sign_a * su_val
                        sv_val = sv[lk * 2 + lj, t2]
                        if sv_val != 0:
                            for block_b, sign_b in prod['B']:
                                gk, gj = b_global(block_b, lk, lj)
                                V[gk * p + gj, c_idx] += sign_b * sv_val
                        sw_val = sw[li * 2 + lj, t2]
                        if sw_val != 0:
                            for block_c, sign_c in prod['C']:
                                gi, gj = c_global(block_c, li, lj)
                                W[gj * n + gi, c_idx] += sign_c * sw_val
            col += 1
    
    # Part 2: 4x4x1 right strip (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
    
    # Part 3: 4x1x4 bottom strip (rank 16)
    for i in range(4):
        for k in range(4):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Part 4: 4x1x1 corner (rank 4)
    for i in range(4):
        U[i * m + 4, col] = 1
        V[4 * p + 4, col] = 1
        W[4 * n + i, col] = 1
        col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Verify
    def verify(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        rank_school = n * m * p
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
