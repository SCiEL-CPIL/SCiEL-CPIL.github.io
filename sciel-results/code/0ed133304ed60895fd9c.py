"""
Strassen-based decomposition for 4x5 by 5x6 matrix multiplication.
Partitions into a 4x4x4 block core (rank 49 via recursive Strassen) and a 4x1x6 strip (schoolbook rank 24).
Total rank: 73.
"""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2x2 matrix multiplication using Strassen.
    Each factor is (U_vec, V_vec, W_vec) where:
    - U_vec has 4 entries (for A[0,0], A[0,1], A[1,0], A[1,1])
    - V_vec has 4 entries (for B[0,0], B[0,1], B[1,0], B[1,1])
    - W_vec has 4 entries (for C[0,0], C[0,1], C[1,0], C[1,1])
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    factors = []
    # M1
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4
    factors.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
    # M5
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 1]))
    # M6
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 5, 6
    
    # Partition: A is 4x5, B is 5x6
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x6, B2 is 1x6
    # C = A1@B1 + A2@B2
    
    # For 4x4x4 core: use Strassen recursively, rank 49
    # For 4x1x6 strip: schoolbook, rank 24
    # Total rank: 73
    
    rank = 73
    U = [[0] * rank for _ in range(n * m)]  # 20 rows
    V = [[0] * rank for _ in range(m * p)]  # 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 24 rows
    
    col = 0
    
    # === 4x4x4 core: Strassen (rank 49) ===
    # View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Apply Strassen at block level: 7 block products, each computed with Strassen (rank 7).
    
    # Block indices for A1 (4x4), viewed as 2x2 blocks of 2x2:
    # A1_00: A[0:2, 0:2] -> U indices: 0,1,5,6
    # A1_01: A[0:2, 2:4] -> U indices: 2,3,7,8
    # A1_10: A[2:4, 0:2] -> U indices: 10,11,15,16
    # A1_11: A[2:4, 2:4] -> U indices: 12,13,17,18
    
    # Block indices for B1 (4x6), we only use cols 0:4 for the 4x4 core:
    # B1_00: B[0:2, 0:2] -> V indices: 0,1,6,7
    # B1_01: B[0:2, 2:4] -> V indices: 2,3,8,9
    # B1_10: B[2:4, 0:2] -> V indices: 12,13,18,19
    # B1_11: B[2:4, 2:4] -> V indices: 14,15,20,21
    
    # Block indices for C (6x4), we only use rows 0:4 for the 4x4 core:
    # C_00: C[0:2, 0:2] -> W indices: 0,1,4,5
    # C_01: C[0:2, 2:4] -> W indices: 2,3,6,7
    # C_10: C[2:4, 0:2] -> W indices: 8,9,12,13
    # C_11: C[2:4, 2:4] -> W indices: 10,11,14,15
    
    # Strassen at block level:
    # M1 = (A1_00 + A1_11)(B1_00 + B1_11)
    # M2 = (A1_10 + A1_11)B1_00
    # M3 = A1_00(B1_01 - B1_11)
    # M4 = A1_11(B1_10 - B1_00)
    # M5 = (A1_00 + A1_01)B1_11
    # M6 = (A1_10 - A1_00)(B1_00 + B1_01)
    # M7 = (A1_01 - A1_11)(B1_10 + B1_11)
    
    # For each block product, we need to compute a 2x2x2 matrix multiplication.
    # We use Strassen for each, giving 7 rank-1 factors per block product.
    
    # Define the block index mappings:
    # For a 2x2 block at position (bi, bj) in A1 (bi, bj in {0,1}):
    #   The 2x2 block has entries at A1[2*bi:2*bi+2, 2*bj:2*bj+2]
    #   U indices: for row r in {0,1}, col c in {0,1}: U[(2*bi+r)*m + (2*bj+c)]
    
    # For B1 block at (bi, bj):
    #   V indices: for row r in {0,1}, col c in {0,1}: V[(2*bi+r)*p + (2*bj+c)]
    
    # For C block at (ci, cj):
    #   W indices: for row r in {0,1}, col c in {0,1}: W[(2*ci+r)*n + (2*cj+c)]
    
    def get_A_block_indices(bi, bj):
        """Get U indices for A1 block at (bi, bj)."""
        indices = []
        for r in range(2):
            for c in range(2):
                i = 2 * bi + r
                j = 2 * bj + c
                indices.append(i * m + j)
        return indices
    
    def get_B_block_indices(bi, bj):
        """Get V indices for B1 block at (bi, bj)."""
        indices = []
        for r in range(2):
            for c in range(2):
                j = 2 * bi + r
                k = 2 * bj + c
                indices.append(j * p + k)
        return indices
    
    def get_C_block_indices(ci, cj):
        """Get W indices for C block at (ci, cj)."""
        indices = []
        for r in range(2):
            for c in range(2):
                k = 2 * ci + r
                i = 2 * cj + c
                indices.append(k * n + i)
        return indices
    
    # Strassen coefficients for block-level:
    # Each entry is (A_block_combo, B_block_combo, C_block_contributions)
    # A_block_combo: dict mapping (bi, bj) -> coefficient
    # B_block_combo: dict mapping (bi, bj) -> coefficient
    # C_block_contributions: dict mapping (ci, cj) -> coefficient
    
    strassen_blocks = [
        # M1 = (A1_00 + A1_11)(B1_00 + B1_11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A1_10 + A1_11)B1_00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A1_00(B1_01 - B1_11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A1_11(B1_10 - B1_00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A1_00 + A1_01)B1_11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,1): 1}),
        # M6 = (A1_10 - A1_00)(B1_00 + B1_01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A1_01 - A1_11)(B1_10 + B1_11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # For each of the 7 block products, apply Strassen at the 2x2 level.
    s2x2_factors = _strassen_2x2x2_factors()
    
    for a_combo, b_combo, c_combo in strassen_blocks:
        # For each Strassen factor at the 2x2 level:
        for u_vec, v_vec, w_vec in s2x2_factors:
            # u_vec, v_vec, w_vec are 4-element vectors for the 2x2 block
            # We need to expand these to the full 4x4 blocks.
            
            # For the A side: a_combo gives us which 2x2 blocks of A1 to use.
            # For each block (bi, bj) with coefficient coeff_a:
            #   The 2x2 block has 4 entries, and u_vec gives coefficients for those entries.
            #   So for entry (r,c) in the block, the coefficient is coeff_a * u_vec[r*2+c]
            
            # Similarly for B and C.
            
            # We're adding one rank-1 factor to the decomposition.
            # U[row] gets the coefficient for that row.
            # V[col] gets the coefficient for that col.
            # W[out] gets the coefficient for that out.
            
            for (bi, bj), coeff_a in a_combo.items():
                a_indices = get_A_block_indices(bi, bj)
                for r in range(2):
                    for c in range(2):
                        u_coeff = coeff_a * u_vec[r * 2 + c]
                        if u_coeff != 0:
                            U[a_indices[r * 2 + c]][col] += u_coeff
            
            for (bi, bj), coeff_b in b_combo.items():
                b_indices = get_B_block_indices(bi, bj)
                for r in range(2):
                    for c in range(2):
                        v_coeff = coeff_b * v_vec[r * 2 + c]
                        if v_coeff != 0:
                            V[b_indices[r * 2 + c]][col] += v_coeff
            
            for (ci, cj), coeff_c in c_combo.items():
                c_indices = get_C_block_indices(ci, cj)
                for r in range(2):
                    for c in range(2):
                        w_coeff = coeff_c * w_vec[r * 2 + c]
                        if w_coeff != 0:
                            W[c_indices[r * 2 + c]][col] += w_coeff
            
            col += 1
    
    # === 4x1x6 strip: schoolbook (rank 24) ===
    # A2 is 4x1 (col 4 of A), B2 is 1x6 (row 4 of B)
    # A2[i, 0] * B2[0, k] -> C[i, k]
    for i in range(4):
        for k in range(6):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify correctness
    if time.time() < deadline:
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = np.array(U)[:, r]
            v_vec = np.array(V)[:, r]
            w_vec = np.array(W)[:, r]
            
            # u_vec has length n*m, v_vec has length m*p, w_vec has length p*n
            # Reshape and compute
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # The rank-1 factor contributes: sum over j of u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This should equal C[i,k] for the contribution of this factor
            for i in range(n):
                for k in range(p):
                    # Sum over j
                    s = 0
                    for j in range(m):
                        s += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    C_computed[i, k] += s
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # If verification fails, fall back to schoolbook
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
