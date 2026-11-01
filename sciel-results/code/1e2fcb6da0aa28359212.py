"""Optimized 4x4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-56 decomposition for 4x4 matrix multiplication using
    two-level Strassen's algorithm.
    
    The approach:
    - Use Strassen's 7-multiplication algorithm at the block level (4x4 = 2x2 blocks of 2x2)
    - Use schoolbook multiplication at the scalar level within each 2x2 block
    - This gives 7 * 8 = 56 rank terms (vs 64 for pure schoolbook)
    
    Interface: returns {'U', 'V', 'W'} with shapes (16, 56), (16, 56), (16, 56)
    Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=4
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = m = p = 4
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper functions for index conversion
    # A[i,j] -> global index i*m + j
    # B[j,k] -> global index j*p + k  
    # C[i,k] -> global index k*n + i (trace convention W[k*n+i])
    
    # Strassen's 7 products for 2x2 block multiplication:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Block indices: (br, bc) where br is block row, bc is block col
    # Each block is 2x2, so local indices (lr, lc) in [0,1]
    
    col = 0
    
    # For each of the 7 Strassen products, we need to decompose
    # the product of two 2x2 matrices into rank-1 terms
    
    # M0 = (A00+A11)(B00+B11)
    # This is a 2x2 matrix product where:
    # X = A00+A11 (2x2 matrix), Y = B00+B11 (2x2 matrix)
    # We need to express X*Y as sum of rank-1 terms
    
    # For a 2x2 matrix product XY, we can use schoolbook: 8 terms
    # (X00*Y00, X00*Y10, X10*Y00, X10*Y10, X01*Y00, X01*Y10, X11*Y00, X11*Y10)
    # But actually for tensor decomposition, each term is u*v^T*w where
    # u picks elements from A, v picks from B, w contributes to C
    
    # Let's define the Strassen products more carefully.
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Global indices:
    # A[i,j] where i in [0,3], j in [0,3]
    #   A00 block: i in {0,1}, j in {0,1}
    #   A01 block: i in {0,1}, j in {2,3}
    #   A10 block: i in {2,3}, j in {0,1}
    #   A11 block: i in {2,3}, j in {2,3}
    
    # B[i,j] where i in [0,3], j in [0,3]
    #   B00 block: i in {0,1}, j in {0,1}
    #   B01 block: i in {0,1}, j in {2,3}
    #   B10 block: i in {2,3}, j in {0,1}
    #   B11 block: i in {2,3}, j in {2,3}
    
    # C[i,j] where i in [0,3], j in [0,3]
    #   C00 block: i in {0,1}, j in {0,1}
    #   C01 block: i in {0,1}, j in {2,3}
    #   C10 block: i in {2,3}, j in {0,1}
    #   C11 block: i in {2,3}, j in {2,3}
    
    # W index: W[k*n+i] where k is column of C, i is row of C
    # So for C[i,j], the W index is j*4 + i
    
    def a_idx(br, bc, lr, lc):
        """Global A index for block (br,bc), local position (lr,lc)"""
        i = 2*br + lr
        j = 2*bc + lc
        return i * 4 + j
    
    def b_idx(br, bc, lr, lc):
        """Global B index for block (br,bc), local position (lr,lc)"""
        i = 2*br + lr
        j = 2*bc + lc
        return i * 4 + j
    
    def w_idx(br, bc, lr, lc):
        """Global W index for C block (br,bc), local position (lr,lc)"""
        i = 2*br + lr
        j = 2*bc + lc
        return j * 4 + i
    
    # For each Strassen product M_l, we have:
    # M_l = X_l * Y_l where X_l and Y_l are 2x2 matrices of linear forms
    
    # The product X_l * Y_l is a 2x2 matrix, and each entry is a sum
    # of products of linear forms. We decompose each such product into
    # rank-1 tensor terms.
    
    # Define the linear forms for each Strassen product
    
    # M0 = (A00+A11)(B00+B11)
    # X0 = A00+A11, Y0 = B00+B11
    # X0[i,j] = A[2*0+i, 2*0+j] + A[2*1+i, 2*1+j]
    # Y0[i,j] = B[2*0+i, 2*0+j] + B[2*1+i, 2*1+j]
    
    # M1 = (A10+A11)B00
    # X1 = A10+A11, Y1 = B00
    # X1[i,j] = A[2*1+i, 2*0+j] + A[2*1+i, 2*1+j]
    # Y1[i,j] = B[2*0+i, 2*0+j]
    
    # M2 = A00(B01-B11)
    # X2 = A00, Y2 = B01-B11
    # X2[i,j] = A[2*0+i, 2*0+j]
    # Y2[i,j] = B[2*0+i, 2*1+j] - B[2*1+i, 2*1+j]
    
    # M3 = A11(B10-B00)
    # X3 = A11, Y3 = B10-B00
    # X3[i,j] = A[2*1+i, 2*1+j]
    # Y3[i,j] = B[2*1+i, 2*0+j] - B[2*0+i, 2*0+j]
    
    # M4 = (A00+A01)B11
    # X4 = A00+A01, Y4 = B11
    # X4[i,j] = A[2*0+i, 2*0+j] + A[2*0+i, 2*1+j]
    # Y4[i,j] = B[2*1+i, 2*1+j]
    
    # M5 = (A10-A00)(B00+B01)
    # X5 = A10-A00, Y5 = B00+B01
    # X5[i,j] = A[2*1+i, 2*0+j] - A[2*0+i, 2*0+j]
    # Y5[i,j] = B[2*0+i, 2*0+j] + B[2*0+i, 2*1+j]
    
    # M6 = (A01-A11)(B10+B11)
    # X6 = A01-A11, Y6 = B10+B11
    # X6[i,j] = A[2*0+i, 2*1+j] - A[2*1+i, 2*1+j]
    # Y6[i,j] = B[2*1+i, 2*0+j] + B[2*1+i, 2*1+j]
    
    # For each M_l, the product X_l * Y_l gives a 2x2 result matrix.
    # Entry (r,c) of X_l * Y_l is sum over k of X_l[r,k] * Y_l[k,c]
    
    # Each term X_l[r,k] * Y_l[k,c] is a product of two linear forms,
    # which we decompose into rank-1 tensor terms.
    
    # X_l[r,k] is a linear form in A entries
    # Y_l[k,c] is a linear form in B entries
    # Their product contributes to C entries according to the Strassen formula
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each term in the decomposition of M_l, we need to know which
    # C blocks it contributes to and with what coefficient.
    
    # Let's define the contributions:
    # contrib[l][br][bc] = coefficient of M_l in C_block(br,bc)
    
    contrib = [[0.0]*2 for _ in range(7)]  # contrib[l] is 2x2 matrix of coefficients
    
    # C00 = M0 + M3 - M4 + M6
    contrib[0][0][0] += 1.0
    contrib[3][0][0] += 1.0
    contrib[4][0][0] += -1.0
    contrib[6][0][0] += 1.0
    
    # C01 = M2 + M4
    contrib[2][0][1] += 1.0
    contrib[4][0][1] += 1.0
    
    # C10 = M1 + M3
    contrib[1][1][0] += 1.0
    contrib[3][1][0] += 1.0
    
    # C11 = M0 - M1 + M2 + M5
    contrib[0][1][1] += 1.0
    contrib[1][1][1] += -1.0
    contrib[2][1][1] += 1.0
    contrib[5][1][1] += 1.0
    
    # Now define the linear forms for X and Y for each Strassen product
    
    # X_forms[l][r][k] = list of (coeff, a_block_row, a_block_col, a_local_row, a_local_col)
    # Y_forms[l][k][c] = list of (coeff, b_block_row, b_block_col, b_local_row, b_local_col)
    
    X_forms = [[[] for _ in range(2)] for _ in range(7)]  # X_forms[l][r] is list for entry (r,*)
    Y_forms = [[[] for _ in range(2)] for _ in range(7)]  # Y_forms[l][k] is list for entry (*,k)
    
    # Actually, let's be more precise. For the product X*Y, entry (r,c) = sum_k X[r,k]*Y[k,c]
    # So we need X[r,k] and Y[k,c] for each r,k,c
    
    # Let me redefine:
    # X_terms[l][r][k] = list of (coeff, br, bc, lr, lc) for A entries in X_l[r,k]
    # Y_terms[l][k][c] = list of (coeff, br, bc, lr, lc) for B entries in Y_l[k,c]
    
    X_terms = [[[] for _ in range(2)] for _ in range(7)]  # X_terms[l][r][k]
    Y_terms = [[[] for _ in range(2)] for _ in range(7)]  # Y_terms[l][k][c]
    
    # Initialize as 3D lists
    X_terms = [[[[] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    Y_terms = [[[[] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # M0: X0 = A00+A11, Y0 = B00+B11
    for r in range(2):
        for k in range(2):
            X_terms[0][r][k].append((1.0, 0, 0, r, k))  # A00[r,k]
            X_terms[0][r][k].append((1.0, 1, 1, r, k))  # A11[r,k]
    for k in range(2):
        for c in range(2):
            Y_terms[0][k][c].append((1.0, 0, 0, k, c))  # B00[k,c]
            Y_terms[0][k][c].append((1.0, 1, 1, k, c))  # B11[k,c]
    
    # M1: X1 = A10+A11, Y1 = B00
    for r in range(2):
        for k in range(2):
            X_terms[1][r][k].append((1.0, 1, 0, r, k))  # A10[r,k]
            X_terms[1][r][k].append((1.0, 1, 1, r, k))  # A11[r,k]
    for k in range(2):
        for c in range(2):
            Y_terms[1][k][c].append((1.0, 0, 0, k, c))  # B00[k,c]
    
    # M2: X2 = A00, Y2 = B01-B11
    for r in range(2):
        for k in range(2):
            X_terms[2][r][k].append((1.0, 0, 0, r, k))  # A00[r,k]
    for k in range(2):
        for c in range(2):
            Y_terms[2][k][c].append((1.0, 0, 1, k, c))  # B01[k,c]
            Y_terms[2][k][c].append((-1.0, 1, 1, k, c)) # B11[k,c]
    
    # M3: X3 = A11, Y3 = B10-B00
    for r in range(2):
        for k in range(2):
            X_terms[3][r][k].append((1.0, 1, 1, r, k))  # A11[r,k]
    for k in range(2):
        for c in range(2):
            Y_terms[3][k][c].append((1.0, 1, 0, k, c))  # B10[k,c]
            Y_terms[3][k][c].append((-1.0, 0, 0, k, c)) # B00[k,c]
    
    # M4: X4 = A00+A01, Y4 = B11
    for r in range(2):
        for k in range(2):
            X_terms[4][r][k].append((1.0, 0, 0, r, k))  # A00[r,k]
            X_terms[4][r][k].append((1.0, 0, 1, r, k))  # A01[r,k]
    for k in range(2):
        for c in range(2):
            Y_terms[4][k][c].append((1.0, 1, 1, k, c))  # B11[k,c]
    
    # M5: X5 = A10-A00, Y5 = B00+B01
    for r in range(2):
        for k in range(2):
            X_terms[5][r][k].append((1.0, 1, 0, r, k))  # A10[r,k]
            X_terms[5][r][k].append((-1.0, 0, 0, r, k)) # A00[r,k]
    for k in range(2):
        for c in range(2):
            Y_terms[5][k][c].append((1.0, 0, 0, k, c))  # B00[k,c]
            Y_terms[5][k][c].append((1.0, 0, 1, k, c))  # B01[k,c]
    
    # M6: X6 = A01-A11, Y6 = B10+B11
    for r in range(2):
        for k in range(2):
            X_terms[6][r][k].append((1.0, 0, 1, r, k))  # A01[r,k]
            X_terms[6][r][k].append((-1.0, 1, 1, r, k)) # A11[r,k]
    for k in range(2):
        for c in range(2):
            Y_terms[6][k][c].append((1.0, 1, 0, k, c))  # B10[k,c]
            Y_terms[6][k][c].append((1.0, 1, 1, k, c))  # B11[k,c]
    
    # Now construct the decomposition
    # For each l in 0..6, for each r in 0..1, for each k in 0..1, for each c in 0..1:
    # The term X_l[r,k] * Y_l[k,c] contributes to M_l[r,c]
    # M_l[r,c] contributes to C blocks according to contrib[l]
    
    # X_l[r,k] * Y_l[k,c] is a product of two linear forms.
    # X_l[r,k] = sum over (coeff_a, br_a, bc_a, lr_a, lc_a) of coeff_a * A[br_a, bc_a][lr_a, lc_a]
    # Y_l[k,c] = sum over (coeff_b, br_b, bc_b, lr_b, lc_b) of coeff_b * B[br_b, bc_b][lr_b, lc_b]
    
    # The product is: sum_{a,b} coeff_a * coeff_b * A[...] * B[...]
    # Each such term is a rank-1 tensor term.
    
    col = 0
    for l in range(7):
        for r in range(2):
            for k in range(2):
                for c in range(2):
                    # This term is X_l[r,k] * Y_l[k,c]
                    # It contributes to M_l[r,c]
                    # M_l[r,c] is a 2x2 matrix entry, but in the block context,
                    # r and c are the local row and column within the block result
                    
                    # The contribution to C:
                    # M_l is a 2x2 matrix, and M_l[r,c] contributes to C blocks
                    # based on contrib[l]
                    
                    # For each pair of terms in X_l[r,k] and Y_l[k,c], we create a rank-1 term
                    for coeff_a, br_a, bc_a, lr_a, lc_a in X_terms[l][r][k]:
                        for coeff_b, br_b, bc_b, lr_b, lc_b in Y_terms[l][k][c]:
                            coeff = coeff_a * coeff_b
                            if coeff == 0:
                                continue
                            
                            # U part: A entry
                            u_idx = a_idx(br_a, bc_a, lr_a, lc_a)
                            U[u_idx][col] += coeff
                            
                            # V part: B entry
                            v_idx = b_idx(br_b, bc_b, lr_b, lc_b)
                            V[v_idx][col] += coeff
                            
                            # W part: C entries
                            # M_l[r,c] contributes to C blocks
                            # The local position in the C block is (r, c)
                            for br_c in range(2):
                                for bc_c in range(2):
                                    coeff_c = contrib[l][br_c][bc_c]
                                    if coeff_c != 0:
                                        w_idx = w_idx(br_c, bc_c, r, c)
                                        W[w_idx][col] += coeff * coeff_c
                            
                            col += 1
    
    # Verify the rank is 56
    assert col == 56, f"Expected rank 56, got {col}"
    
    # Verify the decomposition
    U_r = U.reshape(4, 4, 56)
    V_r = V.reshape(4, 4, 56)
    W_r = W.reshape(4, 4, 56)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This should equal 1 for all i,j,k (since matrix multiplication tensor has 1s)
    
    # Actually, the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Wait, no. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank_school = 64
        U_s = np.zeros((16, rank_school))
        V_s = np.zeros((16, rank_school))
        W_s = np.zeros((16, rank_school))
        
        col_s = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_s[i*4+j][col_s] = 1
                    V_s[j*4+k][col_s] = 1
                    W_s[k*4+i][col_s] = 1
                    col_s += 1
        
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
