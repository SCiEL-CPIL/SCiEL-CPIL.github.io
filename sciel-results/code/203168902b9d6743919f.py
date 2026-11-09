"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We will build a decomposition of rank 36.
    # C = A * B, where A is 2x4, B is 4x5, C is 2x5.
    # Split B into B_left (4x4) and B_right (4x1).
    # C = A * B_left + A * B_right.
    # A * B_right is rank 8 (schoolbook).
    # A * B_left is rank 28 (Strassen-like for 2x4 by 4x4).
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: A * B_right (rank 8)
    # B_right columns are indices 4..4 (since p=5, last col).
    # A is 2x4, B_right is 4x1, C_right is 2x1.
    # C_right[i][0] = sum_{j=0}^{3} A[i][j] * B_right[j][0]
    # B_right[j][0] corresponds to B[j][4].
    # C_right[i][0] corresponds to C[i][4].
    # W index for C[i][4] is 4*n + i = 4*2 + i = 8 + i.
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1.0
            V[j*p + 4, col] = 1.0  # B[j][4]
            W[4*n + i, col] = 1.0  # C[i][4]
            col += 1
    # col is now 8.
    
    # Part 2: A * B_left (rank 28)
    # B_left is 4x4 (cols 0..3).
    # We split A into A_top (2x2, cols 0,1) and A_bot (2x2, cols 2,3).
    # We split B_left into 4 blocks of 2x2:
    # B_LL (rows 0,1; cols 0,1), B_LR (rows 0,1; cols 2,3)
    # B_BL (rows 2,3; cols 0,1), B_BR (rows 2,3; cols 2,3)
    # C_left = A_top * B_LL + A_top * B_LR + A_bot * B_BL + A_bot * B_BR
    # Each term is 2x2 by 2x2 multiplication, rank 7 (Strassen).
    # Total rank 4 * 7 = 28.
    
    def strassen_decompose(U_rows, V_rows, W_rows, base_col):
        """
        Add Strassen decomposition for 2x2 by 2x2 multiplication.
        U_rows: list of 4 row indices for A (top-left, top-right, bot-left, bot-right)
        V_rows: list of 4 row indices for B
        W_rows: list of 4 row indices for C
        """
        # A = [[a, b], [c, d]]
        # B = [[e, f], [g, h]]
        # C = [[i, j], [k, l]]
        # a = U_rows[0], b = U_rows[1], c = U_rows[2], d = U_rows[3]
        # e = V_rows[0], f = V_rows[1], g = V_rows[2], h = V_rows[3]
        # i = W_rows[0], j = W_rows[1], k = W_rows[2], l = W_rows[3]
        
        # Strassen intermediates:
        # M1 = (a+d)(e+h) -> a(e+h) + d(e+h)
        # M2 = (c+d)e -> ce + de
        # M3 = a(f-h) -> af - ah
        # M4 = d(g-e) -> dg - de
        # M5 = (a+b)h -> ah + bh
        # M6 = (c-a)(e+f) -> c(e+f) - a(e+f)
        # M7 = (b-d)(g+h) -> b(g+h) - d(g+h)
        
        # This requires 7 multiplications, but each multiplication is a rank-1 tensor.
        # Wait, Strassen reduces multiplications of scalars.
        # The tensor rank of 2x2x2 is 7.
        # The decomposition is:
        # M1 = a(e+h) + d(e+h) -> This is not a single rank-1 term.
        # The rank-1 terms correspond to the 7 products in the standard Strassen scheme,
        # but we must express C entries as sums of these.
        
        # Let's use the standard 7-term decomposition for M_2,2,2.
        # The 7 rank-1 tensors are:
        # 1. (a+d)(e+h) -> contributes to C
        # 2. (c+d)e
        # 3. a(f-h)
        # 4. d(g-e)
        # 5. (a+b)h
        # 6. (c-a)(e+f)
        # 7. (b-d)(g+h)
        
        # Actually, it is easier to just list the 7 terms (U, V, W) vectors.
        # Term 1: (a+d)(e+h) = ae + ah + de + dh
        #   U: [1, 0, 0, 1] (a, d)
        #   V: [1, 1, 0, 0] (e, h) -- Wait, indices.
        #   V indices: e=0, f=1, g=2, h=3.
        #   V: [1, 0, 0, 1] (e, h)
        #   C contribution: C00 += 1, C01 += 1, C10 += 1, C11 += 1
        #   W: [1, 1, 1, 1]
        
        # Let's verify M1 = (a+d)(e+h).
        # a=e*? No.
        # M1 = (a+d)(e+h) = ae + ah + de + dh.
        # C00 = ae + bg. C11 = ch + df.
        # C00 + C11 = ae + bg + ch + df.
        # M1 + M2 - M4 - M5 = (ae+ah+de+dh) + (ce+de) - (dg-de) - (ah+bh)
        # = ae + ah + de + dh + ce + de - dg + de - ah - bh
        # = ae + ce + 3de + dh - dg - bh. This is getting messy.
        
        # Let's use a known explicit 7-term decomposition for 2x2 matrix mult.
        # Reference: 
        # C00 = M1 - M2 - M4 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M3 + M5 - M6
        # Where:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # We need to decompose each Mk into rank-1 terms? No, Mk IS a product of two linear forms.
        # So each Mk is a rank-1 tensor in the space of (A, B, C).
        # U for M1: a=1, b=0, c=0, d=1 => [1,0,0,1]
        # V for M1: e=1, f=0, g=0, h=1 => [1,0,0,1]
        # W for M1: C00=1, C01=0, C10=0, C11=-1 => [1,0,0,-1]  (from C11 = M1 ... - M6, and C00 = M1...)
        # Wait, the linear combination coefficients are in W.
        
        # Let's map the indices.
        # A indices: 0:a, 1:b, 2:c, 3:d
        # B indices: 0:e, 1:f, 2:g, 3:h
        # C indices: 0:C00, 1:C01, 2:C10, 3:C11
        
        # M1 = (a+d)(e+h)
        # U1 = [1, 0, 0, 1]
        # V1 = [1, 0, 0, 1]
        # W1 = [1, 0, 0, -1]  # C00 += 1*M1, C11 += -1*M1
        
        # M2 = (c+d)e
        # U2 = [0, 0, 1, 1]
        # V2 = [1, 0, 0, 0]
        # W2 = [-1, 0, 1, 0]   # C00 -= M2, C10 += M2
        
        # M3 = a(f-h)
        # U3 = [1, 0, 0, 0]
        # V3 = [0, 1, 0, -1]
        # W3 = [0, 1, 0, -1]   # C01 += M3, C11 -= M3
        
        # M4 = d(g-e)
        # U4 = [0, 0, 0, 1]
        # V4 = [0, 0, 1, -1]
        # W4 = [0, 1, 0, 0]    # C01 += M4
        
        # M5 = (a+b)h
        # U5 = [1, 1, 0, 0]
        # V5 = [0, 0, 0, 1]
        # W5 = [0, 0, 1, 1]    # C10 += M5, C11 += M5
        
        # M6 = (c-a)(e+f)
        # U6 = [0, 0, 1, -1]  # c - a
        # V6 = [1, 1, 0, 0]  # e + f
        # W6 = [0, 0, 0, -1] # C11 -= M6
        
        # M7 = (b-d)(g+h)
        # U7 = [0, 1, 0, -1]  # b - d
        # V7 = [0, 0, 1, 1]   # g + h
        # W7 = [1, 0, 0, 0]   # C00 += M7
        
        terms = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,-1]),
            ([0,0,1,1], [1,0,0,0], [-1,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,-1]),
            ([0,0,0,1], [0,0,1,-1], [0,1,0,0]),
            ([1,1,0,0], [0,0,0,1], [0,0,1,1]),
            ([0,0,1,-1], [1,1,0,0], [0,0,0,-1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0])
        ]
        
        for t in terms:
            u_vec, v_vec, w_vec = t
            for idx in range(4):
                if u_vec[idx] != 0:
                    U[U_rows[idx], base_col + col - base_col] = u_vec[idx] # Placeholder
                if v_vec[idx] != 0:
                    V[V_rows[idx], base_col + col - base_col] = v_vec[idx]
                if w_vec[idx] != 0:
                    W[W_rows[idx], base_col + col - base_col] = w_vec[idx]
            col += 1
        return col

    # We need to carefully manage the global column index.
    # Let's just use a local counter for the 28 terms and add to the global 'col'.
    
    # Block 1: A_top * B_LL
    # A_top rows: 0,1 (in A, which is 2x4). A indices 0,1.
    # B_LL rows: 0,1. Cols: 0,1.
    # C rows: 0,1. Cols: 0,1.
    
    # U_rows for A_top: [0, 1, 2, 3]? No.
    # A is 2x4. Row 0 has cols 0,1,2,3. Row 1 has cols 0,1,2,3.
    # A_top is A[:, 0:2]. So elements A[0,0], A[0,1], A[1,0], A[1,1].
    # In U (flattened 2x4), these are indices:
    # A[0,0] -> 0
    # A[0,1] -> 1
    # A[1,0] -> 2
    # A[1,1] -> 3
    # So U_rows = [0, 1, 2, 3].
    
    # B_LL is B[0:2, 0:2]. Elements B[0,0], B[0,1], B[1,0], B[1,1].
    # In V (flattened 4x5), these are indices:
    # B[0,0] -> 0
    # B[0,1] -> 1
    # B[1,0] -> 5
    # B[1,1] -> 6
    # V_rows = [0, 1, 5, 6].
    
    # C_LL is C[0:2, 0:2]. Elements C[0,0], C[0,1], C[1,0], C[1,1].
    # In W (flattened 5x2, k*n+i), indices are:
    # C[0,0] -> 0*2+0 = 0
    # C[0,1] -> 1*2+0 = 2
    # C[1,0] -> 0*2+1 = 1
    # C[1,1] -> 1*2+1 = 3
    # W_rows = [0, 2, 1, 3].
    
    col = strassen_decompose([0,1,2,3], [0,1,5,6], [0,2,1,3], col)
    
    # Block 2: A_top * B_LR
    # B_LR is B[0:2, 2:4]. Elements B[0,2], B[0,3], B[1,2], B[1,3].
    # V indices:
    # B[0,2] -> 2
    # B[0,3] -> 3
    # B[1,2] -> 7
    # B[1,3] -> 8
    # V_rows = [2, 3, 7, 8].
    
    # C_LR is C[0:2, 2:4]. Elements C[0,2], C[0,3], C[1,2], C[1,3].
    # W indices:
    # C[0,2] -> 2*2+0 = 4
    # C[0,3] -> 3*2+0 = 6
    # C[1,2] -> 2*2+1 = 5
    # C[1,3] -> 3*2+1 = 7
    # W_rows = [4, 6, 5, 7].
    
    col = strassen_decompose([0,1,2,3], [2,3,7,8], [4,6,5,7], col)
    
    # Block 3: A_bot * B_BL
    # A_bot is A[:, 2:4]. Elements A[0,2], A[0,3], A[1,2], A[1,3].
    # U indices:
    # A[0,2] -> 2
    # A[0,3] -> 3
    # A[1,2] -> 4
    # A[1,3] -> 5
    # U_rows = [2,3,4,5].
    
    # B_BL is B[2:4, 0:2]. Elements B[2,0], B[2,1], B[3,0], B[3,1].
    # V indices:
    # B[2,0] -> 10
    # B[2,1] -> 11
    # B[3,0] -> 15
    # B[3,1] -> 16
    # V_rows = [10, 11, 15, 16].
    
    # C_BL is C[0:2, 0:2]. Same W_rows as Block 1.
    # W_rows = [0, 2, 1, 3].
    
    col = strassen_decompose([2,3,4,5], [10,11,15,16], [0,2,1,3], col)
    
    # Block 4: A_bot * B_BR
    # A_bot U_rows = [2,3,4,5].
    # B_BR is B[2:4, 2:4]. Elements B[2,2], B[2,3], B[3,2], B[3,3].
    # V indices:
    # B[2,2] -> 12
    # B[2,3] -> 13
    # B[3,2] -> 17
    # B[3,3] -> 18
    # V_rows = [12, 13, 17, 18].
    
    # C_BR is C[0:2, 2:4]. Same W_rows as Block 2.
    # W_rows = [4, 6, 5, 7].
    
    col = strassen_decompose([2,3,4,5], [12,13,17,18], [4,6,5,7], col)
    
    assert col == 36, f"Expected rank 36, got {col}"
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
