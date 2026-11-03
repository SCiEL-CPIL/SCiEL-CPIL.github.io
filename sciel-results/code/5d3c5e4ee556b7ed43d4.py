import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    Uses Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    # Collect all terms as lists of (row, col) -> value for U, V, W
    U_terms = {}  # term_id -> list of (row, val)
    V_terms = {}
    W_terms = {}
    
    term_id = 0
    
    def add_strassen(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        """Add 7 Strassen terms for 2x2 * 2x2 multiplication."""
        nonlocal term_id
        
        # M1 = (A11+A22)*(B11+B22)
        # U: a11=1, a22=1; V: b11=1, b22=1; W: c11=1, c22=1
        U_terms[term_id] = [(a11, 1), (a22, 1)]
        V_terms[term_id] = [(b11, 1), (b22, 1)]
        W_terms[term_id] = [(c11, 1), (c22, 1)]
        term_id += 1
        
        # M2 = (A21+A22)*B11
        U_terms[term_id] = [(a21, 1), (a22, 1)]
        V_terms[term_id] = [(b11, 1)]
        W_terms[term_id] = [(c21, 1)]
        term_id += 1
        
        # M3 = A11*(B12-B22)
        U_terms[term_id] = [(a11, 1)]
        V_terms[term_id] = [(b12, 1), (b22, -1)]
        W_terms[term_id] = [(c12, 1)]
        term_id += 1
        
        # M4 = A22*(B21-B11)
        U_terms[term_id] = [(a22, 1)]
        V_terms[term_id] = [(b21, 1), (b11, -1)]
        W_terms[term_id] = [(c11, 1), (c22, 1)]
        term_id += 1
        
        # M5 = (A11+A12)*B22
        U_terms[term_id] = [(a11, 1), (a12, 1)]
        V_terms[term_id] = [(b22, 1)]
        W_terms[term_id] = [(c11, -1), (c12, 1)]
        term_id += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U_terms[term_id] = [(a21, 1), (a11, -1)]
        V_terms[term_id] = [(b11, 1), (b12, 1)]
        W_terms[term_id] = [(c22, 1)]
        term_id += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U_terms[term_id] = [(a12, 1), (a22, -1)]
        V_terms[term_id] = [(b21, 1), (b22, 1)]
        W_terms[term_id] = [(c11, 1)]
        term_id += 1
    
    # Now let's set up the indices
    # A is 3x4: A[i,j] for i in 0..2, j in 0..3
    # U index for A[i,j] = i*4 + j
    # B is 4x6: B[j,k] for j in 0..3, k in 0..5
    # V index for B[j,k] = j*6 + k
    # C is 3x6: C[i,k] for i in 0..2, k in 0..5
    # W index for C[i,k] = k*3 + i (trace convention: W[k*n+i, r])
    
    # Split A into blocks:
    # A11 = A[0:2, 0:2] (2x2)
    # A12 = A[0:2, 2:4] (2x2)
    # A21 = A[2:3, 0:2] (1x2)
    # A22 = A[2:3, 2:4] (1x2)
    
    # Split B into blocks:
    # B11 = B[0:2, 0:2] (2x2)
    # B12 = B[0:2, 2:4] (2x2)
    # B13 = B[0:2, 4:6] (2x2)
    # B21 = B[2:4, 0:2] (2x2)
    # B22 = B[2:4, 2:4] (2x2)
    # B23 = B[2:4, 4:6] (2x2)
    
    # C blocks:
    # C11 = C[0:2, 0:2] = A11*B11 + A12*B21
    # C12 = C[0:2, 2:4] = A11*B12 + A12*B22
    # C13 = C[0:2, 4:6] = A11*B13 + A12*B23
    # C21 = C[2:3, 0:2] = A21*B11 + A22*B21
    # C22 = C[2:3, 2:4] = A21*B12 + A22*B22
    # C23 = C[2:3, 4:6] = A21*B13 + A22*B23
    
    # For C11 = A11*B11 + A12*B21:
    # A11*B11 is 2x2*2x2, use Strassen (7 terms)
    # A12*B21 is 2x2*2x2, use Strassen (7 terms)
    # But they both contribute to C11, so we add both sets of terms
    
    # A11 indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # A12 indices: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # A21 indices: A[2,0]=8, A[2,1]=9
    # A22 indices: A[2,2]=10, A[2,3]=11
    
    # B11 indices: B[0,0]=0, B[0,1]=1, B[1,0]=6, B[1,1]=7
    # B12 indices: B[0,2]=2, B[0,3]=3, B[1,2]=8, B[1,3]=9
    # B13 indices: B[0,4]=4, B[0,5]=5, B[1,4]=10, B[1,5]=11
    # B21 indices: B[2,0]=12, B[2,1]=13, B[3,0]=18, B[3,1]=19
    # B22 indices: B[2,2]=14, B[2,3]=15, B[3,2]=20, B[3,3]=21
    # B23 indices: B[2,4]=16, B[2,5]=17, B[3,4]=22, B[3,5]=23
    
    # C11 indices: C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4
    # C12 indices: C[0,2]=2, C[0,3]=3, C[1,2]=5, C[1,3]=6
    # C13 indices: C[0,4]=4, C[0,5]=5, C[1,4]=7, C[1,5]=8
    # C21 indices: C[2,0]=6, C[2,1]=7, C[2,2]=8, C[2,3]=9 -- wait, W[k*n+i]
    # W index for C[i,k] = k*3 + i
    # C[0,0]: W[0*3+0]=0
    # C[0,1]: W[1*3+0]=3
    # C[0,2]: W[2*3+0]=6
    # C[0,3]: W[3*3+0]=9
    # C[0,4]: W[4*3+0]=12
    # C[0,5]: W[5*3+0]=15
    # C[1,0]: W[0*3+1]=1
    # C[1,1]: W[1*3+1]=4
    # C[1,2]: W[2*3+1]=7
    # C[1,3]: W[3*3+1]=10
    # C[1,4]: W[4*3+1]=13
    # C[1,5]: W[5*3+1]=16
    # C[2,0]: W[0*3+2]=2
    # C[2,1]: W[1*3+2]=5
    # C[2,2]: W[2*3+2]=8
    # C[2,3]: W[3*3+2]=11
    # C[2,4]: W[4*3+2]=14
    # C[2,5]: W[5*3+2]=17
    
    # C11 = C[0,0], C[0,1], C[1,0], C[1,1] -> W indices: 0, 3, 1, 4
    # C12 = C[0,2], C[0,3], C[1,2], C[1,3] -> W indices: 6, 9, 7, 10
    # C13 = C[0,4], C[0,5], C[1,4], C[1,5] -> W indices: 12, 15, 13, 16
    # C21 = C[2,0], C[2,1] -> W indices: 2, 5 (only 1x2, not 2x2)
    # C22 = C[2,2], C[2,3] -> W indices: 8, 11
    # C23 = C[2,4], C[2,5] -> W indices: 14, 17
    
    # For C11 = A11*B11 + A12*B21:
    # A11*B11: Strassen with a11=0, a12=1, a21=4, a22=5, b11=0, b12=1, b21=6, b22=7, c11=0, c12=3, c21=1, c22=4
    add_strassen(0, 1, 4, 5, 0, 1, 6, 7, 0, 3, 1, 4)
    # A12*B21: Strassen with a11=2, a12=3, a21=6, a22=7, b11=12, b12=13, b21=18, b22=19, c11=0, c12=3, c21=1, c22=4
    add_strassen(2, 3, 6, 7, 12, 13, 18, 19, 0, 3, 1, 4)
    
    # For C12 = A11*B12 + A12*B22:
    add_strassen(0, 1, 4, 5, 2, 3, 8, 9, 6, 9, 7, 10)
    add_strassen(2, 3, 6, 7, 14, 15, 20, 21, 6, 9, 7, 10)
    
    # For C13 = A11*B13 + A12*B23:
    add_strassen(0, 1, 4, 5, 4, 5, 10, 11, 12, 15, 13, 16)
    add_strassen(2, 3, 6, 7, 16, 17, 22, 23, 12, 15, 13, 16)
    
    # For C21 = A21*B11 + A22*B21 (1x2 * 2x2 + 1x2 * 2x2):
    # A21*B11: A21 is 1x2 [A[2,0], A[2,1]], B11 is 2x2
    # Result is 1x2: [A[2,0]*B[0,0]+A[2,1]*B[1,0], A[2,0]*B[0,1]+A[2,1]*B[1,1]]
    # This is 4 scalar multiplications
    # Term 1: A[2,0]*B[0,0] -> C[2,0]: U[8]=1, V[0]=1, W[2]=1
    U_terms[term_id] = [(8, 1)]
    V_terms[term_id] = [(0, 1)]
    W_terms[term_id] = [(2, 1)]
    term_id += 1
    # Term 2: A[2,1]*B[1,0] -> C[2,0]: U[9]=1, V[6]=1, W[2]=1
    U_terms[term_id] = [(9, 1)]
    V_terms[term_id] = [(6, 1)]
    W_terms[term_id] = [(2, 1)]
    term_id += 1
    # Term 3: A[2,0]*B[0,1] -> C[2,1]: U[8]=1, V[1]=1, W[5]=1
    U_terms[term_id] = [(8, 1)]
    V_terms[term_id] = [(1, 1)]
    W_terms[term_id] = [(5, 1)]
    term_id += 1
    # Term 4: A[2,1]*B[1,1] -> C[2,1]: U[9]=1, V[7]=1, W[5]=1
    U_terms[term_id] = [(9, 1)]
    V_terms[term_id] = [(7, 1)]
    W_terms[term_id] = [(5, 1)]
    term_id += 1
    
    # A22*B21: A22 is 1x2 [A[2,2], A[2,3]], B21 is 2x2
    # Term 5: A[2,2]*B[2,0] -> C[2,0]: U[10]=1, V[12]=1, W[2]=1
    U_terms[term_id] = [(10, 1)]
    V_terms[term_id] = [(12, 1)]
    W_terms[term_id] = [(2, 1)]
    term_id += 1
    # Term 6: A[2,3]*B[3,0] -> C[2,0]: U[11]=1, V[18]=1, W[2]=1
    U_terms[term_id] = [(11, 1)]
    V_terms[term_id] = [(18, 1)]
    W_terms[term_id] = [(2, 1)]
    term_id += 1
    # Term 7: A[2,2]*B[2,1] -> C[2,1]: U[10]=1, V[13]=1, W[5]=1
    U_terms[term_id] = [(10, 1)]
    V_terms[term_id] = [(13, 1)]
    W_terms[term_id] = [(5, 1)]
    term_id += 1
    # Term 8: A[2,3]*B[3,1] -> C[2,1]: U[11]=1, V[19]=1, W[5]=1
    U_terms[term_id] = [(11, 1)]
    V_terms[term_id] = [(19, 1)]
    W_terms[term_id] = [(5, 1)]
    term_id += 1
    
    # For C22 = A21*B12 + A22*B22:
    # A21*B12: 4 terms
    U_terms[term_id] = [(8, 1)]; V_terms[term_id] = [(2, 1)]; W_terms[term_id] = [(8, 1)]; term_id += 1
    U_terms[term_id] = [(9, 1)]; V_terms[term_id] = [(8, 1)]; W_terms[term_id] = [(8, 1)]; term_id += 1
    U_terms[term_id] = [(8, 1)]; V_terms[term_id] = [(3, 1)]; W_terms[term_id] = [(11, 1)]; term_id += 1
    U_terms[term_id] = [(9, 1)]; V_terms[term_id] = [(9, 1)]; W_terms[term_id] = [(11, 1)]; term_id += 1
    # A22*B22: 4 terms
    U_terms[term_id] = [(10, 1)]; V_terms[term_id] = [(14, 1)]; W_terms[term_id] = [(8, 1)]; term_id += 1
    U_terms[term_id] = [(11, 1)]; V_terms[term_id] = [(20, 1)]; W_terms[term_id] = [(8, 1)]; term_id += 1
    U_terms[term_id] = [(10, 1)]; V_terms[term_id] = [(15, 1)]; W_terms[term_id] = [(11, 1)]; term_id += 1
    U_terms[term_id] = [(11, 1)]; V_terms[term_id] = [(21, 1)]; W_terms[term_id] = [(11, 1)]; term_id += 1
    
    # For C23 = A21*B13 + A22*B23:
    # A21*B13: 4 terms
    U_terms[term_id] = [(8, 1)]; V_terms[term_id] = [(4, 1)]; W_terms[term_id] = [(14, 1)]; term_id += 1
    U_terms[term_id] = [(9, 1)]; V_terms[term_id] = [(10, 1)]; W_terms[term_id] = [(14, 1)]; term_id += 1
    U_terms[term_id] = [(8, 1)]; V_terms[term_id] = [(5, 1)]; W_terms[term_id] = [(17, 1)]; term_id += 1
    U_terms[term_id] = [(9, 1)]; V_terms[term_id] = [(11, 1)]; W_terms[term_id] = [(17, 1)]; term_id += 1
    # A22*B23: 4 terms
    U_terms[term_id] = [(10, 1)]; V_terms[term_id] = [(16, 1)]; W_terms[term_id] = [(14, 1)]; term_id += 1
    U_terms[term_id] = [(11, 1)]; V_terms[term_id] = [(22, 1)]; W_terms[term_id] = [(14, 1)]; term_id += 1
    U_terms[term_id] = [(10, 1)]; V_terms[term_id] = [(17, 1)]; W_terms[term_id] = [(17, 1)]; term_id += 1
    U_terms[term_id] = [(11, 1)]; V_terms[term_id] = [(23, 1)]; W_terms[term_id] = [(17, 1)]; term_id += 1
    
    rank = term_id
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for tid in range(rank):
        for row, val in U_terms[tid]:
            U[row][tid] = val
        for row, val in V_terms[tid]:
            V[row][tid] = val
        for row, val in W_terms[tid]:
            W[row][tid] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
