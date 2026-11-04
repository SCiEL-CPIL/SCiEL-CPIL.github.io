"""Tensor decomposition for 2x4 * 4x4 matrix multiplication using Strassen's algorithm.

The approach splits the problem into four 2x2 * 2x2 sub-problems and uses
Strassen's rank-7 decomposition for each. Total rank: 4 * 7 = 28.
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Strassen's 7 rank-1 terms for 2x2 * 2x2 matrix multiplication.
    # For 2x2 matrices A, B with C = A*B:
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #
    # The 7 terms (u, v, w) where each is a vector of length 4:
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c01, c10, c11
    # M2 = (a10+a11)b00       -> contributes to c10, c11
    # M3 = a00(b01-b11)       -> contributes to c01, c11
    # M4 = a11(b10-b00)       -> contributes to c00, c10
    # M5 = (a00+a01)b11       -> contributes to c00, c10
    # M6 = (a10-a00)(b00+b01) -> contributes to c01, c11
    # M7 = (a01-a11)(b10+b11) -> contributes to c00, c10
    #
    # Output: c00 = M1 + M4 - M5 + M7
    #         c01 = M3 + M5
    #         c10 = M2 + M4
    #         c11 = M1 - M2 + M3 + M6
    
    # Each term: (u_vec, v_vec, w_vec) where vectors have length 4
    # u_vec[i*2+j] corresponds to A[i][j]
    # v_vec[i*2+j] corresponds to B[i][j]
    # w_vec[i*2+j] corresponds to C[i][j]
    
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11)
        # u = [1, 0, 0, 1], v = [1, 0, 0, 1]
        # w: contributes +1 to c00, +1 to c01, +1 to c10, +1 to c11 -> [1, 1, 1, 1]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
        
        # M2 = (a10+a11)b00
        # u = [0, 0, 1, 1], v = [1, 0, 0, 0]
        # w: contributes +1 to c10, +1 to c11 -> [0, 0, 1, 1]
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        
        # M3 = a00(b01-b11)
        # u = [1, 0, 0, 0], v = [0, 1, 0, -1]
        # w: contributes +1 to c01, +1 to c11 -> [0, 1, 0, 1]
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        
        # M4 = a11(b10-b00)
        # u = [0, 0, 0, 1], v = [-1, 0, 1, 0]
        # w: contributes +1 to c00, +1 to c10 -> [1, 0, 1, 0]
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        
        # M5 = (a00+a01)b11
        # u = [1, 1, 0, 0], v = [0, 0, 0, 1]
        # w: contributes -1 to c00, +1 to c10 -> [-1, 0, 1, 0]
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        
        # M6 = (a10-a00)(b00+b01)
        # u = [-1, 0, 1, 0], v = [1, 1, 0, 0]
        # w: contributes +1 to c01, +1 to c11 -> [0, 1, 0, 1]
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]),
        
        # M7 = (a01-a11)(b10+b11)
        # u = [0, 1, 0, -1], v = [0, 0, 1, 1]
        # w: contributes +1 to c00, +1 to c10 -> [1, 0, 1, 0]
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 1, 0]),
    ]
    
    # Verify the Strassen decomposition for 2x2*2x2
    # Tensor T[i,j,k] = 1 for all i,j,k in {0,1} (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Sum over terms of u[t][i*2+j] * v[t][j*2+k] * w[t][k*2+i] should equal 1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for u, v, w in strassen_terms:
                    total += u[i*2+j] * v[j*2+k] * w[k*2+i]
                assert total == 1, f"Strassen verification failed at ({i},{j},{k}): got {total}"
    
    # Now build the full decomposition for 2x4 * 4x4
    # A is 2x4, B is 4x4, C is 2x4
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # Split into 4 blocks:
    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    # Block 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    # Block 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    # Block 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    #
    # For each block, we use the 7 Strassen terms.
    # Global indices:
    # U index: i*m + j = i*4 + j, where i in {0,1}, j in {0,1,2,3}
    # V index: j*p + k = j*4 + k, where j in {0,1,2,3}, k in {0,1,2,3}
    # W index: k*n + i = k*2 + i, where k in {0,1,2,3}, i in {0,1}
    
    rank = 28  # 4 blocks * 7 terms each
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    term_offset = 0
    # Blocks: (a_col_start, b_row_start, b_col_start)
    blocks = [
        (0, 0, 0),  # A[:,0:2] * B[0:2,0:2]
        (2, 2, 0),  # A[:,2:4] * B[2:4,0:2]
        (0, 0, 2),  # A[:,0:2] * B[0:2,2:4]
        (2, 2, 2),  # A[:,2:4] * B[2:4,2:4]
    ]
    
    for a_col_start, b_row_start, b_col_start in blocks:
        for u_local, v_local, w_local in strassen_terms:
            # u_local has 4 entries for local A indices (r,c) where r in {0,1}, c in {0,1}
            # Local A[r][c] maps to global A[r][a_col_start + c]
            # U index: r*4 + (a_col_start + c)
            for r in range(2):
                for c in range(2):
                    val = u_local[r*2 + c]
                    if val != 0:
                        U[r * 4 + (a_col_start + c), term_offset] += val
            
            # v_local has 4 entries for local B indices (r,c) where r in {0,1}, c in {0,1}
            # Local B[r][c] maps to global B[b_row_start + r][b_col_start + c]
            # V index: (b_row_start + r)*4 + (b_col_start + c)
            for r in range(2):
                for c in range(2):
                    val = v_local[r*2 + c]
                    if val != 0:
                        V[(b_row_start + r) * 4 + (b_col_start + c), term_offset] += val
            
            # w_local has 4 entries for local C indices (r,c) where r in {0,1}, c in {0,1}
            # Local C[r][c] maps to global C[r][b_col_start + c]
            # W index: (b_col_start + c)*2 + r
            for r in range(2):
                for c in range(2):
                    val = w_local[r*2 + c]
                    if val != 0:
                        W[(b_col_start + c) * 2 + r, term_offset] += val
            
            term_offset += 1
    
    # Verify the full decomposition
    # For random A and B, check that sum_t U[:,t] * V[:,t] * W[:,t] = A*B
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for t in range(rank):
        # U[:,t] has entries U[i*4+j, t] for i in {0,1}, j in {0,1,2,3}
        # V[:,t] has entries V[j*4+k, t] for j in {0,1,2,3}, k in {0,1,2,3}
        # W[:,t] has entries W[k*2+i, t] for k in {0,1,2,3}, i in {0,1}
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    total += U[i*4 + j, t] * V[j*4 + k, t] * W[k*2 + i, t]
                C_computed[i, k] += total
    
    assert np.allclose(C_computed, C_expected), f"Verification failed!\nComputed:\n{C_computed}\nExpected:\n{C_expected}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
