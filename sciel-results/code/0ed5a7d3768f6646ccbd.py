"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 by 4x6 matrix.
    
    Strategy: Use Strassen's algorithm for 2x2 matrix multiplication on sub-blocks.
    Split A (2x4) into two 2x2 blocks: A0, A1.
    Split B (4x6) into six 2x2 blocks: B00, B01, B02, B10, B11, B12.
    C = A*B is 2x6, split into three 2x2 blocks: C0, C1, C2.
    Ck = A0*B0k + A1*B1k for k=0,1,2.
    Each 2x2 product uses Strassen (rank 7), so each Ck uses 2*7=14.
    Total rank = 3*14 = 42.
    
    The output factor follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank) = (8, 42)
    V has shape (m*p, rank) = (24, 42)
    W has shape (p*n, rank) = (12, 42)
    """
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # Given X (2x2) and Y (2x2), compute Z = X*Y using 7 multiplications.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # We need to compute Ck = A0*B0k + A1*B1k for k=0,1,2.
    # For each k, we compute A0*B0k and A1*B1k separately using Strassen.
    # Then add the results.
    
    # Let's define the Strassen decomposition for a single 2x2*2x2 product.
    # For X*Y where X has elements at positions (i,j) and Y at (j,k),
    # we need to map these to the U, V, W indices.
    
    # U[i*m+j, c] - i is row of A, j is col of A
    # V[j*p+k, c] - j is row of B (which is col of A), k is col of B
    # W[k*n+i, c] - k is col of C (col of B), i is row of C (row of A)
    
    # For a sub-block multiplication, say A0 * B00:
    # A0 has rows 0,1 and cols 0,1 of A. So i in {0,1}, j in {0,1}.
    # B00 has rows 0,1 and cols 0,1 of B. So j in {0,1}, k in {0,1}.
    # C00 has rows 0,1 and cols 0,1 of C. So i in {0,1}, k in {0,1}.
    
    # Strassen for X*Y (2x2):
    # The 7 "multiplications" are bilinear forms.
    # For X = A0 (2x2), Y = B00 (2x2):
    # M1 = (A0[0,0]+A0[1,1]) * (B00[0,0]+B00[1,1])
    # M2 = (A0[1,0]+A0[1,1]) * B00[0,0]
    # M3 = A0[0,0] * (B00[0,1]-B00[1,1])
    # M4 = A0[1,1] * (B00[1,0]-B00[0,0])
    # M5 = (A0[0,0]+A0[0,1]) * B00[1,1]
    # M6 = (A0[1,0]-A0[0,0]) * (B00[0,0]+B00[0,1])
    # M7 = (A0[0,1]-A0[1,1]) * (B00[1,0]+B00[1,1])
    
    # C00[0,0] = M1+M4-M5+M7
    # C00[0,1] = M3+M5
    # C00[1,0] = M2+M4
    # C00[1,1] = M1-M2+M3+M6
    
    # We'll implement a helper to add a Strassen decomposition for X*Y
    # where X is a 2x2 sub-block of A and Y is a 2x2 sub-block of B,
    # and the result goes into a 2x2 sub-block of C.
    
    def add_strassen(x_row_offset, x_col_offset, y_row_offset, y_col_offset, 
                     c_row_offset, c_col_offset, col_start, coeff=1):
        """
        Add the Strassen decomposition for the product of a 2x2 block of A
        at (x_row_offset, x_col_offset) times a 2x2 block of B at 
        (y_row_offset, y_col_offset), adding the result to the 2x2 block of C
        at (c_row_offset, c_col_offset) with the given coefficient.
        Uses 7 columns starting from col_start.
        """
        # Strassen's 7 bilinear forms
        # M1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
        U[x_row_offset*m + x_col_offset][col_start] = coeff
        U[(x_row_offset+1)*m + (x_col_offset+1)][col_start] = coeff
        V[y_row_offset*p + y_col_offset][col_start] = coeff
        V[(y_row_offset+1)*p + (y_col_offset+1)][col_start] = coeff
        W[c_col_offset*n + c_row_offset][col_start] = coeff
        W[(c_col_offset+1)*n + (c_row_offset+1)][col_start] = coeff
        W[c_col_offset*n + (c_row_offset+1)][col_start] = coeff
        W[(c_col_offset+1)*n + c_row_offset][col_start] = coeff
        
        # M2 = (X[1,0]+X[1,1]) * Y[0,0]
        U[(x_row_offset+1)*m + x_col_offset][col_start+1] = coeff
        U[(x_row_offset+1)*m + (x_col_offset+1)][col_start+1] = coeff
        V[y_row_offset*p + y_col_offset][col_start+1] = coeff
        W[(c_col_offset+1)*n + (c_row_offset+1)][col_start+1] = coeff
        W[(c_col_offset+1)*n + c_row_offset][col_start+1] = coeff
        
        # M3 = X[0,0] * (Y[0,1]-Y[1,1])
        U[x_row_offset*m + x_col_offset][col_start+2] = coeff
        V[y_row_offset*p + (y_col_offset+1)][col_start+2] = coeff
        V[(y_row_offset+1)*p + (y_col_offset+1)][col_start+2] = -coeff
        W[c_col_offset*n + c_row_offset][col_start+2] = coeff
        W[c_col_offset*n + (c_row_offset+1)][col_start+2] = coeff
        W[(c_col_offset+1)*n + c_row_offset][col_start+2] = coeff
        
        # M4 = X[1,1] * (Y[1,0]-Y[0,0])
        U[(x_row_offset+1)*m + (x_col_offset+1)][col_start+3] = coeff
        V[(y_row_offset+1)*p + y_col_offset][col_start+3] = coeff
        V[y_row_offset*p + y_col_offset][col_start+3] = -coeff
        W[(c_col_offset+1)*n + (c_row_offset+1)][col_start+3] = coeff
        W[c_col_offset*n + c_row_offset][col_start+3] = coeff
        W[(c_col_offset+1)*n + c_row_offset][col_start+3] = coeff
        
        # M5 = (X[0,0]+X[0,1]) * Y[1,1]
        U[x_row_offset*m + x_col_offset][col_start+4] = coeff
        U[x_row_offset*m + (x_col_offset+1)][col_start+4] = coeff
        V[(y_row_offset+1)*p + (y_col_offset+1)][col_start+4] = coeff
        W[(c_col_offset+1)*n + (c_row_offset+1)][col_start+4] = coeff
        W[(c_col_offset+1)*n + c_row_offset][col_start+4] = coeff
        W[c_col_offset*n + c_row_offset][col_start+4] = coeff
        
        # M6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1])
        U[(x_row_offset+1)*m + x_col_offset][col_start+5] = coeff
        U[x_row_offset*m + x_col_offset][col_start+5] = -coeff
        V[y_row_offset*p + y_col_offset][col_start+5] = coeff
        V[y_row_offset*p + (y_col_offset+1)][col_start+5] = coeff
        W[(c_col_offset+1)*n + (c_row_offset+1)][col_start+5] = coeff
        W[c_col_offset*n + (c_row_offset+1)][col_start+5] = coeff
        W[(c_col_offset+1)*n + c_row_offset][col_start+5] = coeff
        
        # M7 = (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1])
        U[x_row_offset*m + (x_col_offset+1)][col_start+6] = coeff
        U[(x_row_offset+1)*m + (x_col_offset+1)][col_start+6] = -coeff
        V[(y_row_offset+1)*p + y_col_offset][col_start+6] = coeff
        V[(y_row_offset+1)*p + (y_col_offset+1)][col_start+6] = coeff
        W[c_col_offset*n + c_row_offset][col_start+6] = coeff
        W[(c_col_offset+1)*n + (c_row_offset+1)][col_start+6] = coeff
        W[(c_col_offset+1)*n + c_row_offset][col_start+6] = coeff
    
    # For each k in {0,1,2}, compute Ck = A0*B0k + A1*B1k
    # A0 is at row_offset=0, col_offset=0 (2x2)
    # A1 is at row_offset=0, col_offset=2 (2x2)
    # B0k is at row_offset=0, col_offset=2*k (2x2)
    # B1k is at row_offset=2, col_offset=2*k (2x2)
    # Ck is at row_offset=0, col_offset=2*k (2x2)
    
    col = 0
    for k in range(3):
        # A0 * B0k
        add_strassen(0, 0, 0, 2*k, 0, 2*k, col)
        col += 7
        # A1 * B1k
        add_strassen(0, 2, 2, 2*k, 0, 2*k, col)
        col += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

