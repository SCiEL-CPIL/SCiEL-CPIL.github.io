import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 tensor contribution
    # u_idx, v_idx, w_idx are the indices into U, V, W
    # u_val, v_val, w_val are the coefficients
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
        
    # Strassen coefficients for 2x2 * 2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We need to map local indices (a_row, a_col, b_row, b_col, c_row, c_col) 
    # to global indices.
    # A is 2x4. A_L is cols 0,1. A_R is cols 2,3.
    # B is 4x5. 
    # B_T is rows 0,1. B_B is rows 2,3.
    # B_T is 2x5. B_TL is cols 0,1,2,3. B_TR is col 4.
    # B_B is 2x5. B_BL is cols 0,1,2,3. B_BR is col 4.
    
    # Process A_L * B_T
    # A_L is A[0:2, 0:2]
    # B_T is B[0:2, 0:5]
    
    # Sub-part 1: A_L * B_TL (2x2 * 2x4)
    # B_TL is B[0:2, 0:4]
    # Split B_TL into B_TL1 (cols 0,1) and B_TL2 (cols 2,3)
    
    # Strassen for A_L * B_TL1
    # A_L indices: (0,0), (0,1), (1,0), (1,1) -> Global U: i*m + j
    # B_TL1 indices: (0,0), (0,1), (1,0), (1,1) -> Global V: j*p + k
    # C indices: (0,0), (0,1), (1,0), (1,1) -> Global W: k*n + i
    
    def strassen_block(a_base_i, a_base_j, b_base_j, b_base_k, c_base_k, c_base_i):
        # a_base_i, a_base_j: top-left corner of A block in global A
        # b_base_j, b_base_k: top-left corner of B block in global B
        # c_base_k, c_base_i: top-left corner of C block in global C (W index k*n+i)
        
        # M1: (A00+A11)(B00+B11)
        # U terms: A00, A11
        # V terms: B00, B11
        # W terms: C00, C01, C10, C11 (with signs)
        
        # We add 7 terms for the 7 multiplications.
        # Each multiplication is a rank-1 tensor? 
        # No, Strassen's algorithm computes linear combinations.
        # The "multiplications" are $M_x = L_A \cdot L_B$.
        # $L_A$ is a linear form in A. $L_B$ is a linear form in B.
        # This corresponds to a rank-1 tensor in the decomposition?
        # Yes, if we expand $L_A L_B = \sum u_i v_j$.
        # But in the tensor decomposition $U, V, W$, each column $r$ corresponds to one multiplication $u_r \cdot v_r \cdot w_r$?
        # No, the standard decomposition $U, V, W$ has entries that are the coefficients.
        # If we use Strassen, we have 7 "virtual" multiplications.
        # Each virtual multiplication $M_x$ contributes to multiple $C_{ik}$.
        # So for each $M_x$, we need to add terms to U, V, W?
        # Wait. The rank is the number of columns in U, V, W.
        # If we use Strassen, we replace the 8 scalar multiplications of schoolbook with 7 "complex" multiplications.
        # But in the tensor decomposition context, a "multiplication" $A_{ij} B_{jk}$ is a rank-1 tensor.
        # Strassen's algorithm uses linear combinations.
        # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11}) = A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # This is a sum of 4 rank-1 tensors.
        # So Strassen does NOT reduce the rank of the tensor decomposition in the simple sense of "number of scalar multiplications" if we are constrained to the $U, V, W$ format where each column is a rank-1 tensor $u \otimes v \otimes w$.
        # UNLESS we allow the entries of U, V, W to be arbitrary linear combinations.
        # The problem asks for "factor matrices ... with integer/half-integer entries that exactly decompose the matrix multiplication tensor".
        # This implies we are looking for the tensor rank.
        # The tensor rank of $M_{2,2,2}$ is 7.
        # This means there exist vectors $u_r, v_r, w_r$ such that $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
        # Strassen's algorithm provides this decomposition!
        # The "7 multiplications" correspond to the 7 rank-1 tensors.
        # The coefficients in U, V, W are the values of the linear forms.
        
        # So, for Strassen's M1:
        # $u_1$ has 1 at A00 and 1 at A11.
        # $v_1$ has 1 at B00 and 1 at B11.
        # $w_1$ has coefficients corresponding to how M1 contributes to C.
        # M1 contributes to C00 (+1), C01 (0), C10 (0), C11 (-1)?
        # Let's check Strassen reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1, the W vector has:
        # W[C00] += 1
        # W[C11] += -1
        
        # We will implement this logic.
        pass

    # We need to carefully map the indices.
    
    # Let's define a function to add a Strassen decomposition for a specific 2x2 A block and 2x2 B block.
    def add_strassen(a_i0, a_j0, b_j0, b_k0):
        # A block starts at row a_i0, col a_j0
        # B block starts at row b_j0, col b_k0
        # C block starts at row a_i0 (implied), col b_k0
        # W index for C[i, k] is k*n + i
        
        # M1
        # U: A00 (a_i0, a_j0), A11 (a_i0+1, a_j0+1)
        # V: B00 (b_j0, b_k0), B11 (b_j0+1, b_k0+1)
        # W: C00 (b_k0, a_i0) -> +1, C11 (b_k0+1, a_i0+1) -> -1
        nonlocal col
        # We need to add a column for M1
        # U entries
        U[a_i0 * m + a_j0][col] = 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][col] = 1
        # V entries
        V[b_j0 * p + b_k0][col] = 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][col] = 1
        # W entries
        W[b_k0 * n + a_i0][col] = 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][col] = -1
        col += 1
        
        # M2: (A10+A11)B00
        # U: A10 (a_i0+1, a_j0), A11 (a_i0+1, a_j0+1)
        # V: B00 (b_j0, b_k0)
        # W: C10 (b_k0, a_i0+1) -> +1, C11 (b_k0+1, a_i0+1) -> -1
        U[(a_i0 + 1) * m + a_j0][col] = 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][col] = 1
        V[b_j0 * p + b_k0][col] = 1
        W[b_k0 * n + (a_i0 + 1)][col] = 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        # U: A00 (a_i0, a_j0)
        # V: B01 (b_j0, b_k0+1), B11 (b_j0+1, b_k0+1)
        # W: C01 (b_k0+1, a_i0) -> +1, C11 (b_k0+1, a_i0+1) -> +1
        U[a_i0 * m + a_j0][col] = 1
        V[b_j0 * p + (b_k0 + 1)][col] = 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][col] = -1
        W[(b_k0 + 1) * n + a_i0][col] = 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        # U: A11 (a_i0+1, a_j0+1)
        # V: B10 (b_j0+1, b_k0), B00 (b_j0, b_k0)
        # W: C00 (b_k0, a_i0) -> +1, C10 (b_k0, a_i0+1) -> +1
        U[(a_i0 + 1) * m + (a_j0 + 1)][col] = 1
        V[(b_j0 + 1) * p + b_k0][col] = 1
        V[b_j0 * p + b_k0][col] = -1
        W[b_k0 * n + a_i0][col] = 1
        W[b_k0 * n + (a_i0 + 1)][col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        # U: A00 (a_i0, a_j0), A01 (a_i0, a_j0+1)
        # V: B11 (b_j0+1, b_k0+1)
        # W: C01 (b_k0+1, a_i0) -> +1, C00 (b_k0, a_i0) -> -1
        U[a_i0 * m + a_j0][col] = 1
        U[a_i0 * m + (a_j0 + 1)][col] = 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][col] = 1
        W[(b_k0 + 1) * n + a_i0][col] = 1
        W[b_k0 * n + a_i0][col] = -1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        # U: A10 (a_i0+1, a_j0), A00 (a_i0, a_j0)
        # V: B00 (b_j0, b_k0), B01 (b_j0, b_k0+1)
        # W: C11 (b_k0+1, a_i0+1) -> +1
        U[(a_i0 + 1) * m + a_j0][col] = 1
        U[a_i0 * m + a_j0][col] = -1
        V[b_j0 * p + b_k0][col] = 1
        V[b_j0 * p + (b_k0 + 1)][col] = 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        # U: A01 (a_i0, a_j0+1), A11 (a_i0+1, a_j0+1)
        # V: B10 (b_j0+1, b_k0), B11 (b_j0+1, b_k0+1)
        # W: C00 (b_k0, a_i0) -> +1, C11 (b_k0+1, a_i0+1) -> +1
        U[a_i0 * m + (a_j0 + 1)][col] = 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][col] = -1
        V[(b_j0 + 1) * p + b_k0][col] = 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][col] = 1
        W[b_k0 * n + a_i0][col] = 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][col] = 1
        col += 1

    # Process A_L (rows 0,1; cols 0,1) * B_TL1 (rows 0,1; cols 0,1)
    add_strassen(0, 0, 0, 0)
    
    # Process A_L (rows 0,1; cols 0,1) * B_TL2 (rows 0,1; cols 2,3)
    add_strassen(0, 0, 0, 2)
    
    # Process A_L (rows 0,1; cols 0,1) * B_TR (rows 0,1; col 4)
    # This is 2x2 * 2x1. Rank 4.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # B_TR is B[0:2, 4:5]. Rows 0,1. Col 4.
    # B0 is B[0,4]. B1 is B[1,4].
    # A00 is A[0,0]. A01 is A[0,1].
    # A10 is A[1,0]. A11 is A[1,1].
    # W index for C0 (row 0, col 4) is 4*n + 0 = 8.
    # W index for C1 (row 1, col 4) is 4*n + 1 = 9.
    
    # Term 1: A00 B0 -> C0
    U[0 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 2: A01 B1 -> C0
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 3: A10 B0 -> C1
    U[1 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # Term 4: A11 B1 -> C1
    U[1 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # Now Process A_R (rows 0,1; cols 2,3) * B_B
    # B_B is rows 2,3.
    # B_BL1 is rows 2,3; cols 0,1.
    # B_BL2 is rows 2,3; cols 2,3.
    # B_BR is rows 2,3; col 4.
    
    # A_R * B_BL1
    add_strassen(0, 2, 2, 0)
    
    # A_R * B_BL2
    add_strassen(0, 2, 2, 2)
    
    # A_R * B_BR (2x2 * 2x1)
    # B_BR is B[2:4, 4:5]. Rows 2,3. Col 4.
    # B0 is B[2,4]. B1 is B[3,4].
    # A00 is A[0,2]. A01 is A[0,3].
    # A10 is A[1,2]. A11 is A[1,3].
    # W index for C0 (row 0, col 4) is 8.
    # W index for C1 (row 1, col 4) is 9.
    
    # Term 5: A00 B0 -> C0
    U[0 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 6: A01 B1 -> C0
    U[0 * m + 3][col] = 1
    V[3 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 7: A10 B0 -> C1
    U[1 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # Term 8: A11 B1 -> C1
    U[1 * m + 3][col] = 1
    V[3 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
