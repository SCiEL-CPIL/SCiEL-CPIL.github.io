"""Rank-minimal exact tensor decomposition for 2x2 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=2, p=2, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = 1 if the product entry (i,k) 
    # receives contribution from A[i,j]*B[j,k]
    # The decomposition says: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # where U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
    # For matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for matmul)
    # Actually, the multiplication tensor has T[i,j,k] = 1 if the (i,k) entry of C
    # gets a contribution from A[i,j]*B[j,k]. For standard matmul, T[i,j,k]=1 for all i,j,k.
    
    # The decomposition: for each r, the contribution to T is U[(i*m+j),r] * V[(j*p+k),r] * W[(k*n+i),r]
    # We need sum_r U[(i*m+j),r] * V[(j*p+k),r] * W[(k*n+i),r] = T[i,j,k] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition():
    """Return the classical Strassen rank-7 decomposition for 2x2 matmul."""
    n, m, p = 2, 2, 2
    rank = 7
    
    # Strassen's 7 products for 2x2 matmul:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a-d)(e+h)
    # M2 = (a+b)e
    # M3 = (c+d)f
    # M4 = d(f-h)
    # M5 = (a+d)(h-g)
    # M6 = (b-c)(e+g)
    # M7 = (b-d)(f+h)
    # C = [[M5+M4-M7+M6, M1+M7],
    #      [M1+M3, M5+M2-M4+M3]]
    
    # U: shape (n*m, rank) = (4, 7), indexed by (i*m+j)
    # V: shape (m*p, rank) = (4, 7), indexed by (j*p+k)
    # W: shape (p*n, rank) = (4, 7), indexed by (k*n+i)
    
    # Let's define the factors.
    # For each of the 7 Strassen products, we have coefficients for A entries, B entries, and C entries.
    
    # A entries: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B entries: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C entries: C[0,0], C[0,1], C[1,0], C[1,1]
    # W is indexed by (k*n+i), so:
    #   k=0,i=0 -> C[0,0] -> index 0
    #   k=0,i=1 -> C[1,0] -> index 1  (wait, let me re-check)
    
    # W[k*n+i, r]: k is column of B / row of C? Let's be careful.
    # The output is C[i,k] = sum_j A[i,j] B[j,k]
    # W[k*n+i] means: first index is k (column of result), second is i (row of result)
    # So W index 0 = k=0,i=0 -> C[0,0]
    #    W index 1 = k=0,i=1 -> C[1,0]  (i=1, k=0)
    #    W index 2 = k=1,i=0 -> C[0,1]  (i=0, k=1)
    #    W index 3 = k=1,i=1 -> C[1,1]  (i=1, k=1)
    
    # U[i*m+j]: i is row of A, j is column of A
    #   U index 0 = i=0,j=0 -> a
    #   U index 1 = i=0,j=1 -> b
    #   U index 2 = i=1,j=0 -> c
    #   U index 3 = i=1,j=1 -> d
    
    # V[j*p+k]: j is row of B, k is column of B
    #   V index 0 = j=0,k=0 -> e
    #   V index 1 = j=0,k=1 -> f
    #   V index 2 = j=1,k=0 -> g
    #   V index 3 = j=1,k=1 -> h
    
    # Strassen products:
    # M1 = (a-d)(e+h): U: [1,0,0,-1], V: [1,0,0,1], W: C[0,1] gets +M1, C[1,0] gets +M1
    #   W index for C[0,1] is k=1,i=0 -> 1*2+0 = 2
    #   W index for C[1,0] is k=0,i=1 -> 0*2+1 = 1
    # M2 = (a+b)e: U: [1,1,0,0], V: [1,0,0,0], W: C[1,1] gets +M2
    #   W index for C[1,1] is k=1,i=1 -> 1*2+1 = 3
    # M3 = (c+d)f: U: [0,0,1,1], V: [0,1,0,0], W: C[0,0] gets +M3, C[1,1] gets +M3
    #   W index for C[0,0] is 0, C[1,1] is 3
    # M4 = d(f-h): U: [0,0,0,1], V: [0,1,0,-1], W: C[0,0] gets -M4, C[1,1] gets -M4
    #   Wait: C[0,0] = M5+M4-M7+M6, so M4 coefficient is +1 for C[0,0]
    #   C[1,1] = M5+M2-M4+M3, so M4 coefficient is -1 for C[1,1]
    # M5 = (a+d)(h-g): U: [1,0,0,1], V: [0,0,-1,1], W: C[0,0] gets +M5, C[1,1] gets +M5
    # M6 = (b-c)(e+g): U: [0,1,-1,0], V: [1,0,1,0], W: C[0,0] gets +M6
    # M7 = (b-d)(f+h): U: [0,1,0,-1], V: [0,1,0,1], W: C[0,1] gets +M7, C[0,0] gets -M7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a-d)(e+h)
    U[0, 0] = 1; U[3, 0] = -1
    V[0, 0] = 1; V[3, 0] = 1
    W[2, 0] = 1  # C[0,1]
    W[1, 0] = 1  # C[1,0]
    
    # M2: (a+b)e
    U[0, 1] = 1; U[1, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1  # C[1,1]
    
    # M3: (c+d)f
    U[2, 2] = 1; U[3, 2] = 1
    V[1, 2] = 1
    W[0, 2] = 1  # C[0,0]
    W[3, 2] = 1  # C[1,1]
    
    # M4: d(f-h)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1  # C[0,0]
    W[3, 3] = -1  # C[1,1]
    
    # M5: (a+d)(h-g)
    U[0, 4] = 1; U[3, 4] = 1
    V[2, 4] = -1; V[3, 4] = 1
    W[0, 4] = 1  # C[0,0]
    W[3, 4] = 1  # C[1,1]
    
    # M6: (b-c)(e+g)
    U[1, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1  # C[0,0]
    
    # M7: (b-d)(f+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1  # C[0,1]
    W[0, 6] = -1  # C[0,0]
    
    # Verify
    assert _verify_decomposition(U, V, W), "Strassen decomposition failed verification"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    """Return the rank-minimal exact tensor decomposition for 2x2 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try Strassen's rank-7 decomposition first (provably optimal)
    result = _strassen_decomposition()
    
    # Verify it's correct
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook rank-8 (should never happen, but safety net)
    n, m, p = 2, 2, 2
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
