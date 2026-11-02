"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # Check a few random samples or all if small
    # For n=4, m=4, p=9, total 144 checks.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)] # 2x2 = 4 rows
    V = [[0] * rank for _ in range(4)] # 2x2 = 4 rows
    W = [[0] * rank for _ in range(4)] # 2x2 = 4 rows
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Term 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1 # C00, C01, C10, C11 all get M1?
    # Wait, C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's assign W coefficients based on the formulas above.
    
    # M1
    W[0][0] += 1  # C00
    W[1][0] += 0  # C01
    W[2][0] += 0  # C10
    W[3][0] += 1  # C11
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] += 0
    W[1][1] += 0
    W[2][1] += 1  # C10
    W[3][1] += -1 # C11
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] += 0
    W[1][2] += 1  # C01
    W[2][2] += 0
    W[3][2] += 1  # C11
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] += -1 # C00
    W[1][3] += 0
    W[2][3] += 1  # C10
    W[3][3] += 0
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] += 1  # C00
    W[1][4] += 1  # C01
    W[2][4] += 0
    W[3][4] += 0
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] += 0
    W[1][5] += 0
    W[2][5] += 0
    W[3][5] += 1  # C11
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] += 1  # C00
    W[1][6] += 0
    W[2][6] += 0
    W[3][6] += 0
    
    return {"U": U, "V": V, "W": W}

def _combine_decomps(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions for M_{n,m,p1} and M_{n,m,p2} into M_{n,m,p1+p2}.
    """
    r1 = len(decomp1["U"][0])
    r2 = len(decomp2["U"][0])
    r = r1 + r2
    
    U = [row + [0]*r2 for row in decomp1["U"]]
    # U is same for both parts?
    # In M_{n,m,p}, U depends on (i,j). It does not depend on k.
    # So U is identical for both parts?
    # Yes, U[i,j,r] is the coefficient for the r-th term.
    # If we concatenate, the first r1 terms correspond to part 1, next r2 to part 2.
    # U for part 1 is in first r1 cols, part 2 in next r2 cols.
    
    # V depends on (j,k).
    # For part 1, k is in [0, p1). For part 2, k is in [p1, p1+p2).
    # So V rows for part 1 are in the first p1*n rows? No, V is (m*p, rank).
    # Rows are indexed by (j,k).
    # We need to place V1 in the rows corresponding to k < p1, and V2 in k >= p1.
    
    V = [[0]*r for _ in range(m * (p1 + p2))]
    for j in range(m):
        for k in range(p1):
            # Row index in V: j*p2_total + k? No, j*(p1+p2) + k
            # V1 row index: j*p1 + k
            V[j*(p1+p2) + k] = decomp1["V"][j*p1 + k][:r1] + [0]*r2
        for k in range(p2):
            # Row index in V: j*(p1+p2) + p1 + k
            # V2 row index: j*p2 + k
            V[j*(p1+p2) + p1 + k] = [0]*r1 + decomp2["V"][j*p2 + k]
            
    # W depends on (k,i).
    # Rows indexed by (k,i).
    # Part 1: k in [0, p1). Part 2: k in [p1, p1+p2).
    W = [[0]*r for _ in range((p1 + p2) * n)]
    for k in range(p1):
        for i in range(n):
            # Row index: k*n + i
            W[k*n + i] = decomp1["W"][k*n + i][:r1] + [0]*r2
    for k in range(p2):
        for i in range(n):
            # Row index: (p1+k)*n + i
            W[(p1+k)*n + i] = [0]*r1 + decomp2["W"][k*n + i]
            
    return {"U": U, "V": V, "W": W}

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2x2 Strassen blocks.
    Rank 49.
    """
    # We have 7 Strassen terms for the outer 2x2 block multiplication.
    # Each term is a 2x2 matrix multiplication (M_{2,2,2}).
    # We use the rank-7 decomposition for each inner multiplication.
    
    # Outer Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will build U, V, W for 4x4x4.
    # U: 16 rows (4x4), 49 cols
    # V: 36 rows (4x9? No, 4x4), 49 cols
    # W: 16 rows (4x4), 49 cols
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    inner = _strassen_2x2()
    
    # Map outer block indices to 4x4 indices
    # A00: rows 0,1; cols 0,1
    # A01: rows 0,1; cols 2,3
    # A10: rows 2,3; cols 0,1
    # A11: rows 2,3; cols 2,3
    
    # We define the 7 outer terms.
    # Each term t has coefficients for A blocks and B blocks.
    # coeff_A[t][block] and coeff_B[t][block]
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    
    outer_A = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    
    outer_B = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Result C blocks coefficients
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    outer_C = [
        [1, 0, 0, 1],  # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, C11(-1)
        [0, 1, 0, 1],  # M3 -> C01, C11
        [-1, 0, 1, 0], # M4 -> C00(-1), C10
        [1, 1, 0, 0],  # M5 -> C00, C01
        [0, 0, 0, 1],  # M6 -> C11
        [1, 0, 0, 0]   # M7 -> C00
    ]
    
    # Now, for each outer term t (0..6), we have an inner M_{2,2,2} decomposition.
    # The inner decomposition has rank 7.
    # Total rank 7*7 = 49.
    # Column offset for term t is t*7.
    
    for t in range(7):
        col_offset = t * 7
        
        # For each inner term r (0..6) in the Strassen decomposition of M_{2,2,2}:
        for r in range(7):
            col = col_offset + r
            
            # Inner U, V, W entries for term r
            # inner["U"] has 4 rows (2x2 blocks).
            # We need to map these to the 4x4 matrix indices.
            
            # Inner U row indices: 0,1,2,3 correspond to A blocks 0,1,2,3?
            # No, inner U is for M_{2,2,2}.
            # The "A" in inner is a 2x2 matrix.
            # The "B" in inner is a 2x2 matrix.
            # The "C" in inner is a 2x2 matrix.
            
            # We are computing M_t = X_t Y_t.
            # X_t is a 2x2 matrix formed by linear combo of A blocks.
            # Y_t is a 2x2 matrix formed by linear combo of B blocks.
            
            # Actually, the inner decomposition computes C_inner = X_inner Y_inner.
            # Here X_inner corresponds to the "A" part of the term, Y_inner to "B".
            
            # The inner U[r] gives coefficients for the elements of X_inner.
            # The inner V[r] gives coefficients for the elements of Y_inner.
            # The inner W[r] gives coefficients for the elements of C_inner.
            
            # We need to expand X_inner (2x2) into the 4x4 A.
            # X_inner[0,0] = sum_{b} outer_A[t][b] * A[b, 0,0] + ...?
            # No.
            # The outer term M_t = (sum c_ab A_b) * (sum d_cb B_c).
            # Let X = sum c_ab A_b.
            # Let Y = sum d_cb B_c.
            # Then M_t = X Y.
            # We decompose M_t using inner Strassen.
            # Inner Strassen decomposes C = X Y.
            # Inner U[r] gives coefficients for X's elements.
            # So X[0,0] = sum_r inner_U[0][r] * (something)?
            # No. The decomposition is C = sum_r U_r \otimes V_r \otimes W_r.
            # This means C[i,j] = sum_r U[i,j,r] V[i,j,r] W[i,j,r]?
            # No.
            # C[i,j] = sum_k A[i,k] B[k,j].
            # Decomposition: C[i,j] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]?
            # The standard form is:
            # T = sum_r U_r \otimes V_r \otimes W_r.
            # U_r is a vector of size n*m.
            # V_r is a vector of size m*p.
            # W_r is a vector of size p*n.
            
            # In our inner decomposition for M_{2,2,2}:
            # inner_U is 4x7.
            # inner_V is 4x7.
            # inner_W is 4x7.
            
            # For the inner term r, we have:
            # X_inner[i,k] contribution?
            # Actually, the inner decomposition computes C_inner = X_inner Y_inner.
            # The "A" matrix in the inner problem is X_inner.
            # The "B" matrix in the inner problem is Y_inner.
            
            # So, for the global A matrix (4x4):
            # The element A_global[i,j] appears in X_inner if it is part of the block.
            
            # Let's map the inner 2x2 indices to global 4x4 indices.
            # Inner index (0,0) -> Global block 0, offset (0,0)
            # Inner index (0,1) -> Global block 0, offset (0,1)
            # Inner index (1,0) -> Global block 0, offset (1,0)
            # Inner index (1,1) -> Global block 0, offset (1,1)
            # ...
            
            # For a given outer term t and inner term r:
            # The coefficient for A_global[i,j] in U is:
            # sum_{b} outer_A[t][b] * inner_U[local_i, local_j, r]
            # where (b, local_i, local_j) maps to global (i,j).
            
            # Similarly for V and W.
            
            for local_i in range(2):
                for local_j in range(2):
                    # Global indices for A
                    # Block b contains A[b, local_i, local_j]
                    # We sum over all blocks b
                    val_U = 0
                    for b in range(4):
                        # Global row: b_row * 2 + local_i
                        # Global col: b_col * 2 + local_j
                        b_row = b // 2
                        b_col = b % 2
                        g_i = b_row * 2 + local_i
                        g_j = b_col * 2 + local_j
                        val_U += outer_A[t][b] * inner["U"][local_i*2 + local_j][r]
                    U[g_i*4 + g_j][col] = val_U
                    
                    # Global indices for B
                    val_V = 0
                    for c in range(4):
                        c_row = c // 2
                        c_col = c % 2
                        g_j_B = c_row * 2 + local_i # B row is inner col?
                        # Wait. B is 4x4.
                        # Inner Y is 2x2.
                        # Y[k, l] corresponds to B blocks.
                        # Y[local_k, local_l].
                        # Global B index: (c_row*2 + local_k, c_col*2 + local_l).
                        # So for V, we need B[global_row, global_col].
                        # V is indexed by (j, k) in M_{n,m,p}.
                        # Here m=4, p=4.
                        # V[j, k, r].
                        # j is column of A / row of B.
                        # k is column of B.
                        
                        # In inner problem, V is indexed by (j, k) where j,k in 0..1.
                        # Inner V[local_j, local_k, r].
                        # This corresponds to B[global_row, global_col].
                        # global_row = c_row*2 + local_j?
                        # global_col = c_col*2 + local_k?
                        
                        # Let's re-verify.
                        # Inner B is 2x2.
                        # Inner B[0,0] is from Block 0.
                        # Inner B[0,1] is from Block 1? No.
                        # The blocks are A00, A01, A10, A11.
                        # A00 is top-left.
                        # A01 is top-right.
                        # A10 is bottom-left.
                        # A11 is bottom-right.
                        
                        # So Block 0 (A00) covers rows 0,1 and cols 0,1.
                        # Block 1 (A01) covers rows 0,1 and cols 2,3.
                        # Block 2 (A10) covers rows 2,3 and cols 0,1.
                        # Block 3 (A11) covers rows 2,3 and cols 2,3.
                        
                        # Inner B[local_row, local_col].
                        # local_row 0 -> Global Row 0,1 (Block 0 or 1)
                        # local_row 1 -> Global Row 2,3 (Block 2 or 3)
                        # local_col 0 -> Global Col 0,1 (Block 0 or 2)
                        # local_col 1 -> Global Col 2,3 (Block 1 or 3)
                        
                        # So for B:
                        # b_row = local_row
                        # b_col = local_col
                        # Block index c = b_row * 2 + b_col.
                        # Global row = c_row * 2 + local_offset_row?
                        # No.
                        # If b_row=0, global rows are 0,1.
                        # If b_row=1, global rows are 2,3.
                        # The local offset within the block is local_row?
                        # Yes.
                        
                        # So for V[j, k, r]:
                        # j is the column of A / row of B.
                        # k is the column of B.
                        
                        # Inner V[local_j, local_k, r].
                        # local_j corresponds to the row of B (which is col of A).
                        # local_k corresponds to the col of B.
                        
                        # For a fixed local_j, local_k:
                        # b_row = local_j
                        # b_col = local_k
                        # c = b_row * 2 + b_col
                        # c_row = c // 2
                        # c_col = c % 2
                        # Global row of B (j) = c_row * 2 + local_j?
                        # No.
                        # If b_row=0, global rows are 0,1. The local offset is local_j?
                        # Wait. Inner B is 2x2.
                        # Inner B[0,0] is the top-left element of the 2x2 block matrix.
                        # This element comes from A00[0,0]?
                        # Yes.
                        # Inner B[0,1] comes from A01[0,0]?
                        # Yes.
                        # Inner B[1,0] comes from A10[0,0]?
                        # Yes.
                        # Inner B[1,1] comes from A11[0,0]?
                        # Yes.
                        
                        # So Inner B[local_row, local_col] corresponds to:
                        # Block c = local_row * 2 + local_col.
                        # Offset within block: (0,0)?
                        # No.
                        # Inner B is a 2x2 matrix of *blocks*.
                        # Each element of Inner B is a 2x2 block.
                        # So Inner B[0,0] is Block 0.
                        # Inner B[0,1] is Block 1.
                        # Inner B[1,0] is Block 2.
                        # Inner B[1,1] is Block 3.
                        
                        # So for Inner V[local_j, local_k, r]:
                        # This term contributes to B elements in Block c = local_j * 2 + local_k.
                        # The coefficient is inner_V[local_j, local_k, r].
                        # This coefficient applies to ALL 4 elements of that block?
                        # No.
                        # The inner decomposition is for M_{2,2,2}.
                        # It assumes the inputs are 2x2 matrices.
                        # So Inner B is a 2x2 matrix.
                        # The elements of Inner B are scalars.
                        # But here, the "scalars" are 2x2 matrices.
                        
                        # So Inner B[local_j, local_k] is a 2x2 matrix (the block).
                        # The inner decomposition uses the *values* of the 2x2 matrix elements?
                        # No.
                        # The inner decomposition computes C = X Y.
                        # X and Y are 2x2 matrices.
                        # If X and Y are matrices of blocks, then C is a matrix of blocks.
                        # The decomposition of M_{2,2,2} applies to the *structure* of the 2x2 multiplication.
                        # It doesn't care that the elements are themselves matrices.
                        # So, Inner U[local_i, local_j, r] is the coefficient for X[local_i, local_j].
                        # X[local_i, local_j] is a 2x2 block.
                        # So the coefficient applies to the entire block?
                        # No.
                        # X[local_i, local_j] is a single element in the 2x2 "meta-matrix".
                        # But that element is a 2x2 matrix.
                        # So the coefficient multiplies the entire 2x2 matrix?
                        # Yes.
                        
                        # So for U:
                        # Inner U[local_i, local_j, r] multiplies Block b = local_i * 2 + local_j?
                        # No.
                        # X is a 2x2 matrix of blocks.
                        # X[0,0] = A00.
                        # X[0,1] = A01.
                        # X[1,0] = A10.
                        # X[1,1] = A11.
                        
                        # So Inner U[local_i, local_j, r] is the coefficient for the block at (local_i, local_j).
                        # Block index b = local_i * 2 + local_j.
                        # This coefficient applies to all 4 elements of that block.
                        
                        # So for U[g_i*4 + g_j][col]:
                        # We sum over local_i, local_j.
                        # b = local_i * 2 + local_j.
                        # coeff = outer_A[t][b] * inner_U[local_i*2 + local_j][r].
                        # This coeff applies to all 4 elements of Block b.
                        # Block b covers rows [2*(b//2), 2*(b//2)+1] and cols [2*(b%2), 2*(b%2)+1].
                        
                        # So we can just assign this coeff to all 4 positions in the block.
                        
                        pass # Logic handled below

            # Simplified Loop:
            # For each outer term t and inner term r:
            # For each local block position (li, lj) in 2x2:
            #   b = li*2 + lj
            #   coeff = outer_A[t][b] * inner_U[li*2 + lj][r]
            #   For each element (ei, ej) in 2x2 block b:
            #     g_i = 2*(b//2) + ei
            #     g_j = 2*(b%2) + ej
            #     U[g_i*4 + g_j][col] = coeff
            
            # Same for V and W.
            
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner["U"][li*2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j][col] = coeff_U
                            
                    coeff_V = outer_B[t][b] * inner["V"][li*2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_j_B = 2 * (b // 2) + ei # Row of B
                            g_k_B = 2 * (b % 2) + ej   # Col of B
                            # V is indexed by (j, k). j is row of B.
                            V[g_j_B * 4 + g_k_B][col] = coeff_V
                            
                    coeff_W = outer_C[t][b] * inner["W"][li*2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_k_C = 2 * (b // 2) + ei # Row of C (k)
                            g_i_C = 2 * (b % 2) + ej   # Col of C (i)
                            # W is indexed by (k, i).
                            W[g_k_C * 4 + g_i_C][col] = coeff_W

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Decompose 9 into 4 + 4 + 1
    d1 = _decomp_4x4x4()
    d2 = _decomp_4x4x4()
    d3 = _schoolbook(4, 4, 1)
    
    # Combine d1 and d2
    d12 = _combine_decomps(d1, d2, 4, 4, 4, 4)
    
    # Combine d12 and d3
    d_total = _combine_decomps(d12, d3, 4, 4, 8, 1)
    
    # Verify
    if _verify_fast(d_total["U"], d_total["V"], d_total["W"], n, m, p):
        return d_total
    else:
        # Fallback to schoolbook if construction failed
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
