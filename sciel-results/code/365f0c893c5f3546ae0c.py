import numpy as np
import time

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 28 Strassen decomposition
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A1 (2x2), A2 (2x2)
    # Split B into B11, B12, B21, B22 (all 2x2)
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # We need to express C[i, k] = sum_j A[i, j] B[j, k]
    # i in 0,1
    # j in 0,1,2,3
    # k in 0,1,2,3
    
    # A[i, j] corresponds to A1[i, j] if j<2 else A2[i, j-2]
    # B[j, k] corresponds to B11[j, k] if j<2, k<2
    #               B12[j, k-2] if j<2, k>=2
    #               B21[j-2, k] if j>=2, k<2
    #               B22[j-2, k-2] if j>=2, k>=2
    
    # We use Strassen for 2x2x2.
    # Strassen decomposition for M = X*Y where X, Y are 2x2.
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We have 4 such products:
    # P1 = A1 * B11
    # P2 = A2 * B21
    # P3 = A1 * B12
    # P4 = A2 * B22
    
    # C11 = P1 + P2
    # C12 = P3 + P4
    
    # Each Pi has 7 terms. Total 28 terms.
    # We construct U, V, W for each term.
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen_term(U, V, W, col, A_idx, B_idx, coeff=1):
        # A_idx: (i, j) in A
        # B_idx: (j, k) in B
        # We are computing A[i, j] * B[j, k]
        # But Strassen computes a 2x2 product.
        # This function is a placeholder. I will build the full U, V, W directly.
        pass

    # Let's build the 28 terms explicitly.
    # Term 0-6: P1 = A1 * B11
    # Term 7-13: P2 = A2 * B21
    # Term 14-20: P3 = A1 * B12
    # Term 21-27: P4 = A2 * B22
    
    # Strassen coefficients for M1..M7
    # M1: (X11+X22)(Y11+Y22)
    # M2: (X21+X22)Y11
    # M3: X11(Y12-Y22)
    # M4: X22(Y21-Y11)
    # M5: (X11+X12)Y22
    # M6: (X21-X11)(Y11+Y12)
    # M7: (X12-X22)(Y21+Y22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # For P1 = A1 * B11:
    # A1 is A[0:2, 0:2], B11 is B[0:2, 0:2]
    # Result is C[0:2, 0:2]
    
    # M1 = (A11+A22)(B11+B22)
    # A11 = A[0,0], A22 = A[1,1]
    # B11 = B[0,0], B22 = B[1,1]
    # U[0*4+0, c] += 1, U[1*4+1, c] += 1
    # V[0*4+0, c] += 1, V[1*4+1, c] += 1
    # W[0*2+0, c] += 1 (C00), W[1*2+0, c] += 1 (C10) -> Wait, C11 is C[0,0]
    # W index is k*n + i.
    # C[0,0] -> k=0, i=0 -> W[0, c]
    # C[1,0] -> k=0, i=1 -> W[2, c]
    # C[0,1] -> k=1, i=0 -> W[1, c]
    # C[1,1] -> k=1, i=1 -> W[3, c]
    
    # Let's write a helper to add a Strassen block.
    
    def add_block(U, V, W, col_offset, a_row, a_col, b_row, b_col, c_row, c_col, signs):
        # signs: list of 7 tuples (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict (i, j) -> val
        # v_coeffs: dict (j, k) -> val
        # w_coeffs: dict (k, i) -> val
        
        # M1
        U[a_row*4+a_col, col_offset+0] += 1
        U[(a_row+1)*4+(a_col+1), col_offset+0] += 1
        V[b_row*4+b_col, col_offset+0] += 1
        V[(b_row+1)*4+(b_col+1), col_offset+0] += 1
        W[(c_row)*2+(c_col), col_offset+0] += 1 # C[c_row, c_col] is M1? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1 contributes to C11 and C22
        W[c_row*2+c_col, col_offset+0] += 1 # C11
        W[(c_row+1)*2+(c_col+1), col_offset+0] += 1 # C22
        
        # M2
        U[(a_row+1)*4+a_col, col_offset+1] += 1
        U[(a_row+1)*4+(a_col+1), col_offset+1] += 1
        V[b_row*4+b_col, col_offset+1] += 1
        W[(c_row+1)*2+c_col, col_offset+1] += 1 # C21
        
        # M3
        U[a_row*4+a_col, col_offset+2] += 1
        V[b_row*4+(b_col+1), col_offset+2] += 1
        V[(b_row+1)*4+(b_col+1), col_offset+2] -= 1
        W[c_row*2+(c_col+1), col_offset+2] += 1 # C12
        W[(c_row+1)*2+(c_col+1), col_offset+2] += 1 # C22
        
        # M4
        U[(a_row+1)*4+(a_col+1), col_offset+3] += 1
        V[(b_row+1)*4+b_col, col_offset+3] += 1
        V[b_row*4+b_col, col_offset+3] -= 1
        W[c_row*2+c_col, col_offset+3] += 1 # C11
        W[(c_row+1)*2+c_col, col_offset+3] += 1 # C21
        
        # M5
        U[a_row*4+a_col, col_offset+4] += 1
        U[a_row*4+(a_col+1), col_offset+4] += 1
        V[(b_row+1)*4+(b_col+1), col_offset+4] += 1
        W[c_row*2+c_col, col_offset+4] -= 1 # C11
        W[c_row*2+(c_col+1), col_offset+4] += 1 # C12
        
        # M6
        U[(a_row+1)*4+a_col, col_offset+5] += 1
        U[a_row*4+a_col, col_offset+5] -= 1
        V[b_row*4+b_col, col_offset+5] += 1
        V[b_row*4+(b_col+1), col_offset+5] += 1
        W[(c_row+1)*2+(c_col+1), col_offset+5] += 1 # C22
        
        # M7
        U[a_row*4+(a_col+1), col_offset+6] += 1
        U[(a_row+1)*4+(a_col+1), col_offset+6] -= 1
        V[(b_row+1)*4+b_col, col_offset+6] += 1
        V[(b_row+1)*4+(b_col+1), col_offset+6] += 1
        W[c_row*2+c_col, col_offset+6] += 1 # C11
        W[(c_row+1)*2+c_col, col_offset+6] += 1 # C22

    # P1: A1 * B11 -> C11 (rows 0,1; cols 0,1)
    # a_row=0, a_col=0, b_row=0, b_col=0, c_row=0, c_col=0
    add_block(U, V, W, 0, 0, 0, 0, 0, 0, 0, None)
    
    # P2: A2 * B21 -> C11 (rows 0,1; cols 0,1)
    # A2 is A[0:2, 2:4], B21 is B[2:4, 0:2]
    # a_row=0, a_col=2, b_row=2, b_col=0, c_row=0, c_col=0
    add_block(U, V, W, 7, 0, 2, 2, 0, 0, 0, None)
    
    # P3: A1 * B12 -> C12 (rows 0,1; cols 2,3)
    # A1 is A[0:2, 0:2], B12 is B[0:2, 2:4]
    # a_row=0, a_col=0, b_row=0, b_col=2, c_row=0, c_col=2
    add_block(U, V, W, 14, 0, 0, 0, 2, 0, 2, None)
    
    # P4: A2 * B22 -> C12 (rows 0,1; cols 2,3)
    # A2 is A[0:2, 2:4], B22 is B[2:4, 2:4]
    # a_row=0, a_col=2, b_row=2, b_col=2, c_row=0, c_col=2
    add_block(U, V, W, 21, 0, 2, 2, 2, 0, 2, None)
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
