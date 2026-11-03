"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition (7 terms)
    # Each term c has vectors u_c (2x2), v_c (2x2), w_c (2x2)
    # u_c[i,j], v_c[j,k], w_c[k,i]
    strassen_u = [
        [[1, 0], [0, 1]],   # c=0
        [[0, 0], [1, 1]],   # c=1
        [[0, 1], [0, 1]],   # c=2
        [[1, 0], [0, 0]],   # c=3
        [[0, 0], [1, -1]],  # c=4
        [[1, 1], [0, 0]],   # c=5
        [[0, 1], [-1, 1]],  # c=6
    ]
    strassen_v = [
        [[1, 0], [0, 1]],   # c=0
        [[0, 0], [1, 0]],   # c=1
        [[0, 0], [1, 0]],   # c=2
        [[1, 0], [0, 0]],   # c=3
        [[0, 0], [1, 0]],   # c=4
        [[1, 0], [1, 0]],   # c=5
        [[1, 0], [0, -1]],  # c=6
    ]
    strassen_w = [
        [[1, 0], [0, 1]],   # c=0
        [[0, 0], [1, 0]],   # c=1
        [[0, 1], [0, 0]],   # c=2
        [[1, 0], [0, 0]],   # c=3
        [[0, 0], [1, 0]],   # c=4
        [[0, 1], [0, 1]],   # c=5
        [[0, 0], [1, 0]],   # c=6
    ]
    
    # Verify Strassen 2x2x2
    def verify_strassen():
        T = np.zeros((2, 2, 2, 2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T[i, j, j, k, k, i] = 1
        S = np.zeros_like(T)
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        S[i, j, j, k, k, i] += strassen_u[c][i][j] * strassen_v[c][j][k] * strassen_w[c][k][i]
        return np.allclose(T, S)
    
    assert verify_strassen(), "Strassen 2x2x2 verification failed"
    
    # Build 4x4x8 decomposition using Strassen
    # We use block decomposition:
    # A is 4x4, B is 4x8, C is 4x8
    # Split A into 2x2 blocks of 2x2, B into 2x4 blocks (2x2 each), C into 2x4 blocks
    # Use Strassen on 2x2x2 blocks
    
    # For 4x4x4 (n=4,m=4,p=4), rank would be 7^2 = 49 using Strassen
    # For 4x4x8, we need to handle the 8 dimension
    # Split B (4x8) into two 4x4 blocks, compute C1 = A*B1 and C2 = A*B2
    # Each uses rank 49, total rank 98
    
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each of the two 4x4 blocks of B
    for b_block in range(2):
        base_rank = b_block * 49
        
        # For 4x4x4 using Strassen:
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # C = [[C11, C12], [C21, C22]] where each is 2x2
        
        # Strassen for 4x4x4 uses 7x7 = 49 terms
        # Term (c1, c2) where c1 is for first Strassen level, c2 for second
        
        # Actually, let's think more carefully.
        # For 4x4x4:
        # A is 4x4, split into 2x2 blocks of 2x2
        # B is 4x4, split into 2x2 blocks of 2x2
        # C is 4x4, split into 2x2 blocks of 2x2
        #
        # The tensor T for 4x4x4 is in V_A^* ⊗ V_B^* ⊗ V_C
        # We can use the Kronecker product structure.
        #
        # Let's index A by (i1,i2,j1,j2), B by (j1,j2,k1,k2), C by (k1,k2,i1,i2)
        # where i1,j1,k1 ∈ {0,1} are block indices, i2,j2,k2 ∈ {0,1} are within-block indices
        #
        # T_{i1,i2,j1,j2, j1,j2,k1,k2, k1,k2,i1,i2} = 1 if consistent
        #
        # We can decompose using Strassen on the block level:
        # The block-level tensor is 2x2x2, which has rank 7
        # The within-block tensor is 2x2x2, which has rank 7
        # So total rank is 7*7 = 49
        
        for c1 in range(7):  # block-level Strassen term
            for c2 in range(7):  # within-block Strassen term
                r = base_rank + c1 * 7 + c2
                
                # U index: i*m + j where i = i1*2 + i2, j = j1*2 + j2
                # u_c1 is for block level: u_c1[i1][j1]
                # u_c2 is for within-block: u_c2[i2][j2]
                # U[i1*2+i2, j1*2+j2, r] = u_c1[i1][j1] * u_c2[i2][j2]
                
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U[i * m + j][r] = strassen_u[c1][i1][j1] * strassen_u[c2][i2][j2]
                
                # V index: j*p + k where j = j1*2 + j2, k = k1*2 + k2
                # But wait, p=8, so k goes up to 7
                # For b_block, k1 is determined by b_block
                # k = b_block * 4 + k1*2 + k2
                
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = b_block * 4 + k1 * 2 + k2
                                V[j * p + k][r] = strassen_v[c1][j1][k1] * strassen_v[c2][j2][k2]
                
                # W index: k*n + i where k = b_block*4 + k1*2 + k2, i = i1*2 + i2
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = b_block * 4 + k1 * 2 + k2
                                i = i1 * 2 + i2
                                W[k * n + i][r] = strassen_w[c1][k1][i1] * strassen_w[c2][k2][i2]
    
    # Convert to numpy and verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify: sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # for the correct indices, 0 otherwise
    
    # The tensor T[i,j,k] = 1 for all i,j,k? No.
    # T is the matrix multiplication tensor.
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity tensor for this purpose)
    
    # Actually, let me reconsider. The tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the multiplication is defined
    # For matrix multiplication, T_{i,j,k} = 1 for all valid i,j,k
    
    # Let's verify
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    err += abs(val - 1.0)
                    if err > 1.0:
                        break
        if err > 1.0:
            break
        if err > 1.0:
            break
    
    if err > 1e-6:
        # Fallback to schoolbook if Strassen verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
