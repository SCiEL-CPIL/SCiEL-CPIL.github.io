import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(i0, j0, k0, col_start):
        # i0, j0, k0 are the starting indices for the 2x2x2 block
        # A is 2x2 at (i0, j0)
        # B is 2x2 at (j0, k0)
        # C is 2x2 at (i0, k0)
        
        # Term 1: (A00 + A11) * (B00 + B11) -> C00, C11
        U[(i0+0)*m + (j0+0)][col_start] = 1
        U[(i0+1)*m + (j0+1)][col_start] = 1
        V[(j0+0)*p + (k0+0)][col_start] = 1
        V[(j0+1)*p + (k0+1)][col_start] = 1
        W[(k0+0)*n + (i0+0)][col_start] = 1
        W[(k0+1)*n + (i0+1)][col_start] = 1
        
        # Term 2: (A10 + A11) * B00 -> C10
        col_start += 1
        U[(i0+1)*m + (j0+0)][col_start] = 1
        U[(i0+1)*m + (j0+1)][col_start] = 1
        V[(j0+0)*p + (k0+0)][col_start] = 1
        W[(k0+0)*n + (i0+1)][col_start] = 1
        
        # Term 3: A00 * (B10 - B11) -> C01
        col_start += 1
        U[(i0+0)*m + (j0+0)][col_start] = 1
        V[(j0+1)*p + (k0+0)][col_start] = 1
        V[(j0+1)*p + (k0+1)][col_start] = -1
        W[(k0+1)*n + (i0+0)][col_start] = 1
        
        # Term 4: A11 * (B01 - B00) -> C00, C10
        col_start += 1
        U[(i0+1)*m + (j0+1)][col_start] = 1
        V[(j0+0)*p + (k0+1)][col_start] = 1
        V[(j0+0)*p + (k0+0)][col_start] = -1
        W[(k0+0)*n + (i0+0)][col_start] = 1
        W[(k0+1)*n + (i0+1)][col_start] = 1
        
        # Term 5: (A00 + A01) * B11 -> C00, C01
        col_start += 1
        U[(i0+0)*m + (j0+0)][col_start] = 1
        U[(i0+0)*m + (j0+1)][col_start] = 1
        V[(j0+1)*p + (k0+1)][col_start] = 1
        W[(k0+0)*n + (i0+0)][col_start] = -1
        W[(k0+1)*n + (i0+0)][col_start] = 1
        
        # Term 6: (A10 - A00) * (B00 + B01) -> C10
        col_start += 1
        U[(i0+1)*m + (j0+0)][col_start] = 1
        U[(i0+0)*m + (j0+0)][col_start] = -1
        V[(j0+0)*p + (k0+0)][col_start] = 1
        V[(j0+0)*p + (k0+1)][col_start] = 1
        W[(k0+0)*n + (i0+1)][col_start] = 1
        
        # Term 7: (A01 - A11) * (B10 + B11) -> C00, C11
        col_start += 1
        U[(i0+0)*m + (j0+1)][col_start] = 1
        U[(i0+1)*m + (j0+1)][col_start] = -1
        V[(j0+1)*p + (k0+0)][col_start] = 1
        V[(j0+1)*p + (k0+1)][col_start] = 1
        W[(k0+0)*n + (i0+0)][col_start] = -1
        W[(k0+1)*n + (i0+1)][col_start] = 1
        
        return col_start + 1

    # Block 1: i in {0,1}, j in {0,1}
    # k in {0,1}, {2,3}, {4,5}
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    
    # Block 2: i in {0,1}, j in {2,3}
    # k in {0,1}, {2,3}, {4,5}
    col = add_strassen_block(0, 2, 0, col)
    col = add_strassen_block(0, 2, 2, col)
    col = add_strassen_block(0, 2, 4, col)
    
    # Block 3: i in {0,1}, j in {4}
    # k in {0..5}
    # Schoolbook: 2 * 1 * 6 = 12 terms
    for i in range(2):
        for k in range(6):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Basic verification
    res = construct()
    n, m, p = DIMENSIONS
    import numpy as np
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Random test
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Reconstruct C
    # C[i, k] = sum_r U[i, j] V[j, k] W[k, i] ?
    # The decomposition is T = sum U^r o V^r o W^r
    # C = sum_r (U^r * A) * (B * V^r) * W^r ?
    # Actually, the standard way to verify is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # And the tensor decomposition ensures this identity holds.
    # To check the factors:
    # We can form the tensor T from U, V, W and compare with the identity tensor.
    
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(len(U[0])):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
    T_identity = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_identity[i*m + j, j*p + k, k*n + i] = 1
                
    error = np.linalg.norm(T_decomp - T_identity)
    print(f"Error: {error}")
    if error < 1e-9:
        print("Valid decomposition.")
    else:
        print("Invalid decomposition.")
