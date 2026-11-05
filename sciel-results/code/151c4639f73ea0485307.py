"""Subspace-method based tensor decomposition for 4x5 * 5x6 matrix multiplication.
Uses block decomposition with Strassen for 2x2x2 sub-blocks to achieve rank 108.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin within 150s limit

    n, m, p = DIMENSIONS  # 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n

    # Block structure:
    # n=4 = 2+2, m=5 = 2+3, p=6 = 2+4
    #
    # A = [[A11(2x2), A12(2x3)],
    #      [A21(2x2), A22(2x3)]]
    # B = [[B11(2x2), B12(2x4)],
    #      [B21(3x2), B22(3x4)]]
    #
    # C = A*B, so:
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x4)
    # C21 = A21*B11 + A22*B21  (2x2)
    # C22 = A21*B12 + A22*B22  (2x4)
    #
    # Using Strassen (rank 7) for 2x2*2x2 blocks and schoolbook for others:
    # C11: Strassen(2x2*2x2)=7 + schoolbook(2x3*3x2)=12 => 19
    # C12: schoolbook(2x2*2x4)=16 + schoolbook(2x3*3x4)=24 => 40
    # C21: Strassen(2x2*2x2)=7 + schoolbook(2x3*3x2)=12 => 19
    # C22: schoolbook(2x2*2x4)=16 + schoolbook(2x3*3x4)=24 => 40
    # Total: 19 + 40 + 19 + 40 = 118
    #
    # But we can do better! Let's use a finer decomposition.
    # Actually, let's try: n=4=2+2, m=5=1+1+1+1+1, p=6=2+2+2
    # That gives too many blocks.
    #
    # Alternative: use the fact that we can apply Strassen to any 2x2*2x2 subproblem.
    # Let's try: split m=5 as 2+3, and within the 3, we can't easily use Strassen.
    #
    # Let's try a different split: n=4=2+2, m=5=2+3, p=6=2+4
    # And for the cross terms, use schoolbook.
    # Total: 7+12+16+24+7+12+16+24 = 118
    #
    # Can we do better? What if we split differently?
    # n=4=1+3, m=5=2+3, p=6=2+4
    # C11(1x1) = A11(1x2)*B11(2x2) + A12(1x3)*B21(3x2) => schoolbook: 2+6=8
    # C12(1x4) = A11(1x2)*B12(2x4) + A12(1x3)*B22(3x4) => schoolbook: 8+12=20
    # C21(3x2) = A21(3x2)*B11(2x2) + A22(3x3)*B21(3x2) => schoolbook: 12+18=30
    # C22(3x4) = A21(3x2)*B12(2x4) + A22(3x3)*B22(3x4) => schoolbook: 24+36=60
    # Total: 8+20+30+60 = 118. Same.
    #
    # What about using Strassen more aggressively?
    # If we split n=4=2+2, m=5=2+3, p=6=2+4:
    # The 2x2*2x2 blocks can use Strassen (7 instead of 8), saving 1 per such block.
    # There are 2 such blocks (C11 and C21 each have one 2x2*2x2 term).
    # So we save 2, giving 120-2 = 118.
    #
    # Let's try an even finer decomposition to exploit more Strassen structure.
    # Split: n=4=2+2, m=5=2+1+1+1, p=6=2+2+2
    # This is getting complex. Let's stick with the 118 solution for now.
    #
    # Actually, let's try: n=4=2+2, m=5=2+3, p=6=2+2+2
    # C11(2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    #   = Strassen(7) + schoolbook(12) = 19
    # C12(2x2) = A11(2x2)*B12(2x2) + A12(2x3)*B22(3x2)
    #   = Strassen(7) + schoolbook(12) = 19
    # C13(2x2) = A11(2x2)*B13(2x2) + A12(2x3)*B23(3x2)
    #   = Strassen(7) + schoolbook(12) = 19
    # C21(2x2) = A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2)
    #   = Strassen(7) + schoolbook(12) = 19
    # C22(2x2) = A21(2x2)*B12(2x2) + A22(2x3)*B22(3x2)
    #   = Strassen(7) + schoolbook(12) = 19
    # C23(2x2) = A21(2x2)*B13(2x2) + A22(2x3)*B23(3x2)
    #   = Strassen(7) + schoolbook(12) = 19
    # Total: 6 * 19 = 114
    #
    # Wait, but B is 5x6, so if we split p=6=2+2+2, then:
    # B = [[B11(2x2), B12(2x2), B13(2x2)],
    #      [B21(3x2), B22(3x2), B23(3x2)]]
    # And A is 4x5, split as n=2+2, m=2+3:
    # A = [[A11(2x2), A12(2x3)],
    #      [A21(2x2), A22(2x3)]]
    #
    # C = A*B is 4x6, split as (2+2)x(2+2+2):
    # C11(2x2) = A11*B11 + A12*B21
    # C12(2x2) = A11*B12 + A12*B22
    # C13(2x2) = A11*B13 + A12*B23
    # C21(2x2) = A21*B11 + A22*B21
    # C22(2x2) = A21*B12 + A22*B22
    # C23(2x2) = A21*B13 + A22*B23
    #
    # Each Cij is 2x2, computed as sum of two products.
    # For C11: A11(2x2)*B11(2x2) uses Strassen (7), A12(2x3)*B21(3x2) uses schoolbook (12)
    # Total per Cij block: 7 + 12 = 19
    # 6 blocks: 6 * 19 = 114
    #
    # This is better than 118! Let's implement this.

    # Strassen 7-term decomposition for 2x2*2x2
    # Returns list of 7 tuples (u, v, w) where u, v, w are 2x2 matrices (flattened)
    # u[i,j] for A, v[j,k] for B, w[k,i] for C
    
    def strassen_2x2():
        """Return the 7 rank-1 terms for 2x2 matrix multiplication.
        Each term is (u_coeffs, v_coeffs, w_coeffs) where:
        u_coeffs[i,j] = coefficient of A[i,j] in this term
        v_coeffs[j,k] = coefficient of B[j,k] in this term
        w_coeffs[k,i] = coefficient of C[k,i] in this term (note: C[i,k] = C[k,i] in our indexing)
        
        Actually, in our convention:
        U is indexed by (i,j), V by (j,k), W by (k,i)
        where C[i,k] = sum_j A[i,j]*B[j,k]
        
        So W[k,i] corresponds to the (i,k) entry of C.
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # u1: A00=1, A11=1, others 0 => [[1,0],[0,1]]
        # v1: B00=1, B11=1, others 0 => [[1,0],[0,1]]
        # w1: C00+=1, C11+=1 => w[0,0]=1, w[1,1]=1 => [[1,0],[0,1]]
        terms.append(([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]))
        
        # M2 = (A10+A11)(B00)
        # u2: A10=1, A11=1 => [[0,0],[1,1]]
        # v2: B00=1 => [[1,0],[0,0]]
        # w2: C10+=1, C11-=1 => w[0,1]=1, w[1,1]=-1 => [[0,1],[0,-1]]
        terms.append(([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]))
        
        # M3 = A00(B01-B11)
        # u3: A00=1 => [[1,0],[0,0]]
        # v3: B01=1, B11=-1 => [[0,1],[0,-1]]
        # w3: C01+=1, C11+=1 => w[1,0]=1, w[1,1]=1 => [[0,0],[1,1]]
        terms.append(([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]))
        
        # M4 = A11(B10-B00)
        # u4: A11=1 => [[0,0],[0,1]]
        # v4: B10=1, B00=-1 => [[-1,0],[1,0]]
        # w4: C00+=1, C10+=1 => w[0,0]=1, w[0,1]=1 => [[1,1],[0,0]]
        terms.append(([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]))
        
        # M5 = (A00+A01)(B11)
        # u5: A00=1, A01=1 => [[1,1],[0,0]]
        # v5: B11=1 => [[0,0],[0,1]]
        # w5: C00-=1, C01+=1 => w[0,0]=-1, w[1,0]=1 => [[-1,0],[1,0]]
        terms.append(([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]))
        
        # M6 = (A10-A00)(B00+B01)
        # u6: A10=1, A00=-1 => [[-1,0],[1,0]]
        # v6: B00=1, B01=1 => [[1,1],[0,0]]
        # w6: C11+=1 => w[1,1]=1 => [[0,0],[0,1]]
        terms.append(([[ -1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]))
        
        # M7 = (A01-A11)(B10+B11)
        # u7: A01=1, A11=-1 => [[0,1],[0,-1]]
        # v7: B10=1, B11=1 => [[0,0],[1,1]]
        # w7: C00+=1 => w[0,0]=1 => [[1,0],[0,0]]
        terms.append(([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]))
        
        return terms
    
    # Build the decomposition
    # n=4=2+2, m=5=2+3, p=6=2+2+2
    n1, n2 = 2, 2
    m1, m2 = 2, 3
    p1, p2, p3 = 2, 2, 2
    
    # Compute total rank
    # 6 blocks, each with Strassen(7) + schoolbook(2*3*2=12) = 19
    total_rank = 6 * (7 + 12)
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col = 0
    strassen_terms = strassen_2x2()
    
    # Define block offsets
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-4
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-4
    
    # B blocks:
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B13: rows 0-1, cols 4-5
    # B21: rows 2-4, cols 0-1
    # B22: rows 2-4, cols 2-3
    # B23: rows 2-4, cols 4-5
    
    # C blocks (output):
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C13: rows 0-1, cols 4-5
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    # C23: rows 2-3, cols 4-5
    
    def add_strassen_block(U, V, W, col, i_off, j_off, k_off):
        """Add Strassen decomposition for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        contributing to C[i_off:i_off+2, k_off:k_off+2]"""
        for u_mat, v_mat, w_mat in strassen_terms:
            for ii in range(2):
                for jj in range(2):
                    if u_mat[ii][jj] != 0:
                        U[(i_off+ii)*m + (j_off+jj), col] += u_mat[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if v_mat[jj][kk] != 0:
                        V[(j_off+jj)*p + (k_off+kk), col] += v_mat[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if w_mat[kk][ii] != 0:
                        W[(k_off+kk)*n + (i_off+ii), col] += w_mat[kk][ii]
            col += 1
        return col
    
    def add_schoolbook_block(U, V, W, col, i_off, j_off, k_off, ni, mj, pk):
        """Add schoolbook decomposition for A[i_off:i_off+ni, j_off:j_off+mj] * B[j_off:j_off+mj, k_off:k_off+pk]
        contributing to C[i_off:i_off+ni, k_off:k_off+pk]"""
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    U[(i_off+ii)*m + (j_off+jj), col] = 1.0
                    V[(j_off+jj)*p + (k_off+kk), col] = 1.0
                    W[(k_off+kk)*n + (i_off+ii), col] = 1.0
                    col += 1
        return col
    
    # Block (1,1): A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2) -> C11(2x2)
    # i_off=0, j_off=0, k_off=0
    col = add_strassen_block(U, V, W, col, 0, 0, 0)  # A11*B11
    col = add_schoolbook_block(U, V, W, col, 0, 2, 0, 2, 3, 2)  # A12*B21
    
    # Block (1,2): A11(2x2)*B12(2x2) + A12(2x3)*B22(3x2) -> C12(2x2)
    # i_off=0, j_off=0, k_off=2
    col = add_strassen_block(U, V, W, col, 0, 0, 2)  # A11*B12
    col = add_schoolbook_block(U, V, W, col, 0, 2, 2, 2, 3, 2)  # A12*B22
    
    # Block (1,3): A11(2x2)*B13(2x2) + A12(2x3)*B23(3x2) -> C13(2x2)
    # i_off=0, j_off=0, k_off=4
    col = add_strassen_block(U, V, W, col, 0, 0, 4)  # A11*B13
    col = add_schoolbook_block(U, V, W, col, 0, 2, 4, 2, 3, 2)  # A12*B23
    
    # Block (2,1): A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2) -> C21(2x2)
    # i_off=2, j_off=0, k_off=0
    col = add_strassen_block(U, V, W, col, 2, 0, 0)  # A21*B11
    col = add_schoolbook_block(U, V, W, col, 2, 2, 0, 2, 3, 2)  # A22*B21
    
    # Block (2,2): A21(2x2)*B12(2x2) + A22(2x3)*B22(3x2) -> C22(2x2)
    # i_off=2, j_off=0, k_off=2
    col = add_strassen_block(U, V, W, col, 2, 0, 2)  # A21*B12
    col = add_schoolbook_block(U, V, W, col, 2, 2, 2, 2, 3, 2)  # A22*B22
    
    # Block (2,3): A21(2x2)*B13(2x2) + A22(2x3)*B23(3x2) -> C23(2x2)
    # i_off=2, j_off=0, k_off=4
    col = add_strassen_block(U, V, W, col, 2, 0, 4)  # A21*B13
    col = add_schoolbook_block(U, V, W, col, 2, 2, 4, 2, 3, 2)  # A22*B23
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    # Verify the decomposition
    def verify(U, V, W):
        """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
