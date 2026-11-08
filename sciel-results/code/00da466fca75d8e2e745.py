"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact for random integer inputs."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The decomposition should satisfy:
    # sum_k U[i,k] * V[j,k] * W[t,k] = 1 if t corresponds to the (i,j) product
    # where i = (row_a, col_a) of A, j = (row_b, col_b) of B
    # and the output C[row_a, col_b] = sum_col_a A[row_a,col_a] * B[col_b... wait
    # Actually for matrix multiplication C = A @ B where A is n x m and B is m x p:
    # C[r,c] = sum_k A[r,k] * B[k,c]
    # So the tensor T[r,k,c] = 1
    # In the decomposition: sum_l U[r,k,l] * V[k,c,l] * W[r,c,l] = T[r,k,c]
    # Wait, the trace convention is W[k*n+i, r] which I need to parse.
    # Actually let's just test with random matrices.
    
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The product is: for each l, U[:,l] is a vector in R^{n*m}, V[:,l] in R^{m*p}, W[:,l] in R^{p*n}
    # The outer product interpretation:
    # C = sum_l (reshape U[:,l] to n x m) * (reshape V[:,l] to m x p) with W[:,l] as coefficients?
    # No. The standard tensor decomposition of matrix multiplication:
    # T_{r,k,c} = sum_l U_{r,k,l} * V_{k,c,l} * W_{r,c,l}
    # where r is row of A (0..n-1), k is column of A / row of B (0..m-1), c is column of B (0..p-1)
    # U index: r*m + k, V index: k*p + c, W index: c*n + r (or r*n + c?)
    # The task says "W[k*n+i, r]" which is confusing. Let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Hmm, this might mean W is indexed as W[k*n+i] where the factor r is the coefficient.
    # Actually, looking at the seed code:
    # W is p*n x rank. The indices used are 0..11 for p*n=12.
    # In the seed, for C[0,0] they use W[0], for C[0,1] they use W[3], C[1,0] W[1], C[1,1] W[4], etc.
    # For n=3, p=4: C is 3x4. C[r,c] with r=0..2, c=0..3.
    # W[0] = C[0,0], W[1] = C[1,0], W[2] = C[2,0], W[3] = C[0,1], W[4] = C[1,1], W[5] = C[2,1], ...
    # So W index = r*p + c? No: W[0]=C[0,0], W[1]=C[1,0], W[2]=C[2,0] → column-major? 
    # Actually r=0,c=0→0; r=1,c=0→1; r=2,c=0→2; r=0,c=1→3. So W index = r + c*n? 
    # 0+0*3=0, 1+0*3=1, 2+0*3=2, 0+1*3=3, 1+1*3=4, 2+1*3=5. Yes! W index = r + c*n.
    # But the task says W[k*n+i, r]. That's confusing. Let me just verify by checking the tensor.
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n) where T[a,b,c] = 1 
    # if the (a-th) input of A times (b-th) input of B contributes to (c-th) output of C.
    # a = r*m + k (r=row of A, k=col of A), b = k*p + c (k=row of B, c=col of B)
    # Wait, k must match: col of A = row of B.
    # T[a,b,c] = 1 if a//m == b//p (the shared index) and c corresponds to (r,c) of C.
    # Actually a = r*m + k, b = k*p + c_b, and C[r, c_b] is at W index r + c_b*n.
    # So T[a, b, w_idx] = 1 if a//m == b//p AND w_idx = (a//m) + (b%p)*n... wait.
    # a = r*m + k → r = a//m, k = a%m
    # b = k*p + c_b → k = b//p, c_b = b%p
    # For T to be 1, we need a%m == b//p (the k matches), and w_idx = r + c_b*n = a//m + (b%p)*n
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for k in range(m):
            for c in range(p):
                a = r * m + k
                b = k * p + c
                w_idx = r + c * n
                T[a, b, w_idx] = 1
    
    # Now check: for all (a,b,c), sum_l U[a,l]*V[b,l]*W[c,l] should equal T[a,b,c]
    # This is expensive for large rank, but for rank 34 it's fine.
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                expected = T[a, b, c]
                actual = sum(U[a, l] * V[b, l] * W[c, l] for l in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    # Strassen Left (A11 * B1L)
    # A11 indices: 0,1,3,4
    # B1L indices: 0,1,4,5
    # C1L indices: 0,3,1,4
    
    # P1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # P2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # P3
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # P4
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # P5
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # P6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # P7
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # A12 * B2L
    # A12 indices: 2, 5
    # B2L indices: 8, 9
    # C1L contribution:
    # C[0,0]->0, C[0,1]->3, C[1,0]->1, C[1,1]->4
    
    # T1: A[0,2] B[2,0] C[0,0]
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,1] C[0,1]
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,0] C[1,0]
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,1] C[1,1]
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    # A21 * B1L
    # A21 indices: 6, 7
    # B1L indices: 0,1,4,5
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,0] B[0,0] C[2,0]
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,1] C[2,1]
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,0] C[2,0]
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,1] C[2,1]
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # A22 * B2L
    # A22 index: 8
    # B2L indices: 8, 9
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,2] B[2,0] C[2,0]
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,1] C[2,1]
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    # Strassen Right (A11 * B1R)
    # A11 indices: 0,1,3,4
    # B1R indices: 2,3,6,7
    # C1R indices: 6,9,7,10
    
    # P1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    # P2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    # P3
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    # P4
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # P5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    # P6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    # P7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    # A12 * B2R
    # A12 indices: 2, 5
    # B2R indices: 10, 11
    # C1R contribution:
    # C[0,2]->6, C[0,3]->9, C[1,2]->7, C[1,3]->10
    
    # T1: A[0,2] B[2,2] C[0,2]
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,3] C[0,3]
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,2] C[1,2]
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,3] C[1,3]
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    # A21 * B1R
    # A21 indices: 6, 7
    # B1R indices: 2,3,6,7
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,0] B[0,2] C[2,2]
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,3] C[2,3]
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,2] C[2,2]
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,3] C[2,3]
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    # A22 * B2R
    # A22 index: 8
    # B2R indices: 10, 11
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,2] B[2,2] C[2,2]
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,3] C[2,3]
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    # Verify the decomposition is valid
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
