"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Direct construction using known low-rank decompositions.
    We use a block-based approach where we decompose the 4x4x5 multiplication
    into smaller blocks and use known optimal decompositions for each.
    
    For 4x4 * 4x5:
    - Split into 2x2 blocks for A and B
    - Use Strassen's algorithm (rank 7) for 2x2 * 2x2 multiplications
    - Use direct multiplication (rank 4) for 2x2 * 2x1 multiplications
    
    Total: 8 * 7 + 4 * 4 = 56 + 16 = 72
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    deadline = time.time() + 140  # Safety margin from 150s limit
    
    # U: shape (n*m, rank) = (16, rank)
    # V: shape (m*p, rank) = (20, rank)  
    # W: shape (p*n, rank) = (20, rank)
    
    # First, let's try to construct a valid decomposition
    # Using the block structure
    
    rank_total = 72
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col_offset = 0
    
    # Helper: Add Strassen terms for 2x2 * 2x2 multiplication
    # A block at (row_block_a, col_block_a), B block at (row_block_b, col_block_b)
    # Result goes to (row_block_a, col_block_b) in the result
    def add_strassen_2x2(ra, ca, rb, cb):
        """Add 7 Strassen terms for 2x2 * 2x2."""
        global col_offset
        # A indices: rows 2*ra, 2*ra+1; cols 2*ca, 2*ca+1
        # B indices: rows 2*rb, 2*rb+1; cols 2*cb, 2*cb+1
        # Result indices: rows 2*ra, 2*ra+1; cols 2*cb, 2*cb+1
        
        # Strassen's 7 products for C = A*B where A, B, C are 2x2:
        # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)*(e+h)
        # M2 = (c+d)*e  
        # M3 = a*(f-h)
        # M4 = d*(g-e)
        # M5 = (a+b)*h
        # M6 = (c-a)*(e+f)
        # M7 = (b-d)*(g+h)
        # C = [[M1+M4-M5+M7, M3+M5],
        #      [M2+M4,        M1-M2+M3+M6]]
        
        # U entries for each M term (coefficients on A)
        # V entries for each M term (coefficients on B)
        # W entries for each M term (coefficients on C)
        
        # M1: (a+d)*(e+h)
        U[2*ra * m + 2*ca, col_offset] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col_offset] = 1.0
        V[2*rb * p + 2*cb, col_offset] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col_offset] = 1.0
        # C11 += M1, C22 -= M1 (wait, let me recalculate)
        # Actually: C11 = M1 + M4 - M5 + M7
        #            C12 = M3 + M5
        #            C21 = M2 + M4
        #            C22 = M1 - M2 + M3 + M6
        W[2*cb * n + 2*ra, col_offset] = 1.0    # C11
        W[(2*cb+1) * n + (2*ra+1), col_offset] = 1.0  # C22
        col_offset += 1
        
        # M2: (c+d)*e
        U[(2*ra+1) * m + 2*ca, col_offset] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col_offset] = 1.0
        V[2*rb * p + 2*cb, col_offset] = 1.0
        W[2*cb * n + (2*ra+1), col_offset] = 1.0    # C21
        W[(2*cb+1) * n + (2*ra+1), col_offset] = -1.0  # C22
        col_offset += 1
        
        # M3: a*(f-h)
        U[2*ra * m + 2*ca, col_offset] = 1.0
        V[2*rb * p + (2*cb+1), col_offset] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col_offset] = -1.0
        W[(2*cb+1) * n + 2*ra, col_offset] = 1.0    # C12
        W[(2*cb+1) * n + (2*ra+1), col_offset] = 1.0  # C22
        col_offset += 1
        
        # M4: d*(g-e)
        U[(2*ra+1) * m + (2*ca+1), col_offset] = 1.0
        V[(2*rb+1) * p + 2*cb, col_offset] = 1.0
        V[2*rb * p + 2*cb, col_offset] = -1.0
        W[2*cb * n + 2*ra, col_offset] = 1.0    # C11
        W[2*cb * n + (2*ra+1), col_offset] = 1.0  # C21
        col_offset += 1
        
        # M5: (a+b)*h
        U[2*ra * m + 2*ca, col_offset] = 1.0
        U[2*ra * m + (2*ca+1), col_offset] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col_offset] = 1.0
        W[2*cb * n + 2*ra, col_offset] = -1.0    # C11
        W[(2*cb+1) * n + 2*ra, col_offset] = 1.0  # C12
        col_offset += 1
        
        # M6: (c-a)*(e+f)
        U[(2*ra+1) * m + 2*ca, col_offset] = 1.0
        U[2*ra * m + 2*ca, col_offset] = -1.0
        V[2*rb * p + 2*cb, col_offset] = 1.0
        V[2*rb * p + (2*cb+1), col_offset] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col_offset] = 1.0  # C22
        col_offset += 1
        
        # M7: (b-d)*(g+h)
        U[2*ra * m + (2*ca+1), col_offset] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col_offset] = -1.0
        V[(2*rb+1) * p + 2*cb, col_offset] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col_offset] = 1.0
        W[2*cb * n + 2*ra, col_offset] = 1.0    # C11
        col_offset += 1
    
    # Helper: Add direct multiplication terms for 2x2 * 2x1
    def add_direct_2x2_2x1(ra, ca, rb, cb):
        """Add 4 direct terms for 2x2 * 2x1."""
        global col_offset
        # A is 2x2 at (ra, ca), B is 2x1 at (rb, cb)
        # Result is 2x1 at (ra, cb)
        # C[i] = A[i,0]*B[0] + A[i,1]*B[1] for i=0,1
        
        for i in range(2):
            for j in range(2):
                U[(2*ra + i) * m + (2*ca + j), col_offset] = 1.0
                V[(2*rb + j) * p + (2*cb), col_offset] = 1.0
                W[(2*cb) * n + (2*ra + i), col_offset] = 1.0
                col_offset += 1
    
    # Now add all the block multiplications
    # A = [[A00, A01], [A10, A11]] (2x2 blocks, each 2x2)
    # B = [[B00, B01, B02], [B10, B11, B12]] (2x3 blocks, first two cols are 2x2, third is 2x1)
    
    # C00 = A00*B00 + A01*B10 (2x2 * 2x2 + 2x2 * 2x2 = 2x2)
    add_strassen_2x2(0, 0, 0, 0)  # A00 * B00
    add_strassen_2x2(0, 1, 1, 0)  # A01 * B10
    
    # C01 = A00*B01 + A01*B11 (2x2)
    add_strassen_2x2(0, 0, 0, 1)  # A00 * B01
    add_strassen_2x2(0, 1, 1, 1)  # A01 * B11
    
    # C02 = A00*B02 + A01*B12 (2x1)
    add_direct_2x2_2x1(0, 0, 0, 1)  # A00 * B02 (B02 is at col 1 of block, which is actual col 2)
    add_direct_2x2_2x1(0, 1, 1, 1)  # A01 * B12
    
    # C10 = A10*B00 + A11*B10 (2x2)
    add_strassen_2x2(1, 0, 0, 0)  # A10 * B00
    add_strassen_2x2(1, 1, 1, 0)  # A11 * B10
    
    # C11 = A10*B01 + A11*B11 (2x2)
    add_strassen_2x2(1, 0, 0, 1)  # A10 * B01
    add_strassen_2x2(1, 1, 1, 1)  # A11 * B11
    
    # C12 = A10*B02 + A11*B12 (2x1)
    add_direct_2x2_2x1(1, 0, 0, 1)  # A10 * B02
    add_direct_2x2_2x1(1, 1, 1, 1)  # A11 * B12
    
    # Trim to actual used rank
    U = U[:, :col_offset]
    V = V[:, :col_offset]
    W = W[:, :col_offset]
    
    # Verify the decomposition
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition (rank n*m*p = 80)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
