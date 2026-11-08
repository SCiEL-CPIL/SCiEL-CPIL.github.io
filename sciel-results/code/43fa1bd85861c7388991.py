"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen44_decomposition():
    """
    Rank-49 tensor decomposition for 4x4 x 4x4 -> 4x4 matrix multiplication
    using two levels of Strassen (7 block-mults, each expanded to 7 scalar mults).
    
    Returns list of 49 (coeff_a, coeff_b, coeff_c) dicts.
    coeff_a: {(i,j): val} for A[i,j]
    coeff_b: {(j,k): val} for B[j,k]
    coeff_c: {(k,i): val} for C[i,k] (trace convention W[k*n+i])
    """
    terms = []
    
    # 7 block-level Strassen products for 4x4x4
    # Each: (a_blocks, b_blocks, c_map)
    # a_blocks/b_blocks: {(block_row, block_col): sign}
    # c_map: {(p_row, p_col): sign} where P is the 2x2 result of the block product
    # P[pr,pc] contributes sign * P[pr,pc] to C block at position (br_c, bc_c)
    # but actually c_map maps P-position to which C-blocks it goes to.
    
    # Let me define more carefully:
    # The 7 block products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B00)
    # M4 = (A11)(B01-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M3 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 - M6
    #
    # Each Mi is a 2x2 matrix product. Within each, we use Strassen (7 scalar mults).
    # The 2x2 result P of Mi contributes to C blocks as specified.
    
    block_products = [
        # M1: A=(A00+A11), B=(B00+B11)
        # C00 += P, C11 += P
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1},
         {0: [(0,0,1)], 1: [(0,0,1)], 2: [(0,0,1)], 3: [(0,0,1)],
          4: [(1,1,1)], 5: [(1,1,1)], 6: [(1,1,1)], 7: [(1,1,1)]}),
        # Wait, this doesn't work. P is 2x2, so P[0,0], P[0,1], P[1,0], P[1,1]
        # C00 is a 2x2 block: C00[0,0]=C[0,0], C00[0,1]=C[0,1], C00[1,0]=C[1,0], C00[1,1]=C[1,1]
        # So P[0,0]->C[0,0], P[0,1]->C[0,1], P[1,0]->C[1,0], P[1,1]->C[1,1]
        # And similarly for C11: P[0,0]->C[2,2], P[0,1]->C[2,3], P[1,0]->C[3,2], P[1,1]->C[3,3]
        
        # Let me redefine c_map: for each P-position (pr,pc) [0-3], list of (C_row, C_col, sign)
    ]
    
    # P positions: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    block_products = [
        # M1: A00+A11 times B00+B11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1},
         {0: [(0,0,1),(2,2,1)], 1: [(0,1,1),(2,3,1)],
          2: [(1,0,1),(3,2,1)], 3: [(1,1,1),(3,3,1)]}),
        # M2: A10+A11 times B00
        ({(1,0):1, (1,1):1}, {(0,0):1},
         {0: [(2,0,1),(2,2,-1)], 1: [(2,1,1),(2,3,-1)],
          2: [(3,0,1),(3,2,-1)], 3: [(3,1,1),(3,3,-1)]}),
        # M3: A00 times B10-B00
        ({(0,0):1}, {(1,0):1, (0,0):-1},
         {0: [(0,0,-1),(0,2,1)], 1: [(0,1,-1),(0,3,1)],
          2: [(1,0,-1),(1,2,1)], 3: [(1,1,-1),(1,3,1)]}),
        # M4: A11 times B01-B10
        ({(1,1):1}, {(0,1):1, (1,0):-1},
         {0: [(0,0,-1),(2,0,1)], 1: [(0,1,-1),(2,1,1)],
          2: [(1,0,-1),(3,0,1)], 3: [(1,1,-1),(3,1,1)]}),
        # M5: A00+A01 times B11
        ({(0,0):1, (0,1):1}, {(1,1):1},
         {0: [(0,2,1),(2,2,1)], 1: [(0,3,1),(2,3,1)],
          2: [(1,2,1),(3,2,1)], 3: [(1,3,1),(3,3,1)]}),
        # M6: A10-A00 times B00+B01
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1},
         {0: [(2,2,-1)], 1: [(2,3,-1)],
          2: [(3,2,-1)], 3: [(3,3,-1)]}),
        # M7: A01-A11 times B10+B11
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1},
         {0: [(0,0,1)], 1: [(0,1,1)],
          2: [(1,0,1)], 3: [(1,1,1)]}),
    ]
    
    # For each block product, expand the 2x2 matrix product into 7 scalar Strassen mults
    # X[di,dj] = sum over (br,bc) in a_blocks: a_blocks[(br,bc)] * A[br*2+di, bc*2+dj]
    # Y[di,dj] = sum over (br,bc) in b_blocks: b_blocks[(br,bc)] * B[br*2+di, bc*2+dj]
    #
    # Strassen 2x2:
    # m1=(X00+X11)(Y00+Y11)
    # m2=(X10+X11)(Y00)
    # m3=(X00)(Y10-Y00)
    # m4=(X11)(Y01-Y10)
    # m5=(X00+X01)(Y11)
    # m6=(X10-X00)(Y00+Y01)
    # m7=(X01-X11)(Y10+Y11)
    # P[0,0]=m1-m3-m4+m7
    # P[0,1]=m3+m5
    # P[1,0]=m2+m4
    # P[1,1]=m1-m2+m5-m6
    
    # c_contribs: for each mult index k (0-6), dict of P-position -> sign
    c_contribs = [
        {0: 1, 3: 1},      # m1: P[0,0]+1, P[1,1]+1
        {2: 1, 3: -1},     # m2: P[1,0]+1, P[1,1]-1
        {0: -1, 1: 1},     # m3: P[0,0]-1, P[0,1]+1
        {0: -1, 2: 1},     # m4: P[0,0]-1, P[1,0]+1
        {1: 1, 3: 1},      # m5: P[0,1]+1, P[1,1]+1
        {3: -1},           # m6: P[1,1]-1
        {0: 1},            # m7: P[0,0]+1
    ]
    
    for a_blocks, b_blocks, c_map in block_products:
        # Build X and Y coefficient matrices
        X = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in a_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    X[di][dj][idx] = sign
        
        Y = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in b_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    Y[di][dj][idx] = sign
        
        def merge(*dicts):
            result = {}
            for d in dicts:
                for k, v in d.items():
                    result[k] = result.get(k, 0) + v
            return {k: v for k, v in result.items() if v != 0}
        
        def negate(d):
            return {k: -v for k, v in d.items()}
        
        # 7 scalar Strassen products
        scalar_mults = [
            (merge(X[0][0], X[1][1]), merge(Y[0][0], Y[1][1])),           # m1
            (merge(X[1][0], X[1][1]), merge(Y[0][0])),                     # m2
            (merge(X[0][0]), merge(Y[1][0], negate(Y[0][0]))),             # m3
            (merge(X[1][1]), merge(Y[0][1], negate(Y[1][0]))),             # m4
            (merge(X[0][0], X[0][1]), merge(Y[1][1])),                     # m5
            (merge(X[1][0], negate(X[0][0])), merge(Y[0][0], Y[0][1])),    # m6
            (merge(X[0][1], negate(X[1][1])), merge(Y[1][0], Y[1][1])),    # m7
        ]
        
        for k, (a_coefs, b_coefs) in enumerate(scalar_mults):
            if not a_coefs or not b_coefs:
                continue
            
            # Determine C coefficients
            c_coefs = {}
            for p_pos, s_sign in c_contribs[k].items():
                pr, pc = divmod(p_pos, 2)
                for (c_row, c_col, c_sign) in c_map[p_pos]:
                    # C entry at (c_row, c_col) in the output matrix C[i,k]
                    # In trace convention, W index is (k, i) = (c_col, c_row)
                    c_idx = (c_col, c_row)  # (k, i)
                    total_sign = s_sign * c_sign
                    c_coefs[c_idx] = c_coefs.get(c_idx, 0) + total_sign
            
            c_coefs = {k: v for k, v in c_coefs.items() if v != 0}
            if c_coefs:
                terms.append((a_coefs, b_coefs, c_coefs))
    
    return terms


def construct():
    """
    Matrix multiplication 4x4 by 4x9.
    Returns U (16 x rank), V (36 x rank), W (36 x rank) with integer entries.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    DEADLINE = start_time + 100  # safety margin
    
    # Strategy: Use Strassen-49 for 4x4x4 multiplications
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # rank(A*B1) = 49 (Strassen)
    # rank(A*B2) = 49 (Strassen)
    # rank(A*B3) = 16 (schoolbook for 4x4 x 4x1)
    # Total rank: 49 + 49 + 16 = 114
    # Score: 144/114 ≈ 1.263
    
    # Get Strassen-49 terms for 4x4x4
    strassen_terms = _strassen44_decomposition()
    
    # Verify we got 49 terms
    assert len(strassen_terms) == 49, f"Expected 49 terms, got {len(strassen_terms)}"
    
    # Build the full decomposition
    total_rank = 49 + 49 + 16
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Part 1: A * B1 where B1 = B[:, 0:4]
    for c, (a_coefs, b_coefs, c_coefs) in enumerate(strassen_terms):
        for (i, j), val in a_coefs.items():
            U[i*m + j, c] += val
        for (j, k), val in b_coefs.items():
            V[j*p + k, c] += val  # k is 0..3 for B1
        for (k, i), val in c_coefs.items():
            W[k*n + i, c] += val
    
    # Part 2: A * B2 where B2 = B[:, 4:8]
    offset = 49
    for c, (a_coefs, b_coefs, c_coefs) in enumerate(strassen_terms):
        for (i, j), val in a_coefs.items():
            U[i*m + j, offset + c] += val
        for (j, k), val in b_coefs.items():
            V[j*p + (k + 4), offset + c] += val  # k shifted by 4 for B2
        for (k, i), val in c_coefs.items():
            W[k*n + i, offset + c] += val
    
    # Part 3: A * B3 where B3 = B[:, 8:9] (4x1)
    offset2 = 98
    for i in range(n):
        for j in range(m):
            for k in range(1):  # only 1 column
                col = offset2 + i*m + j
                U[i*m + j, col] = 1
                V[j*p + (8 + k), col] = 1
                W[(8 + k)*n + i, col] = 1
    
    # Verify the decomposition
    if time.time() < DEADLINE:
        # Check: for random matrices, U @ diag(V) @ W should give A*B
        np.random.seed(42)
        A_test = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B_test = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for c in range(total_rank):
            u_vec = U[:, c]
            v_vec = V[:, c]
            w_vec = W[:, c]
            # Reshape u_vec to (n, m), v_vec to (m, p), w_vec to (p, n)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            # Contribution: u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # C[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += np.dot(u_mat[i, :], v_mat[:, k] * w_mat[k, i])
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook if verification fails
            rank_fb = n * m * p
            U_fb = np.zeros((n*m, rank_fb), dtype=int)
            V_fb = np.zeros((m*p, rank_fb), dtype=int)
            W_fb = np.zeros((p*n, rank_fb), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m + j, col] = 1
                        V_fb[j*p + k, col] = 1
                        W_fb[k*n + i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
