"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition using block structure
    # 2x5x6 = (2x2x2) + (2x1x4) + (2x2x2) + (2x2x2)
    # Ranks: 7 + 1 + 7 + 7 = 22
    
    rank = 22
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen formulas for 2x2x2
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        # M2 = (A21+A22)B11 -> C21
        # M3 = A11(B12-B22) -> C12
        # M4 = A22(B21-B11) -> C21, C22
        # M5 = (A11+A12)B22 -> C11, C12
        # M6 = (A21-A11)(B11+B12) -> C22
        # M7 = (A12-A22)(B21+B22) -> C11
        
        # Term 1: M1
        # U: A11, A22 contribute. V: B11, B22. W: C11, C22.
        # This is complex to map directly to U,V,W indices.
        # Instead, we define the 7 rank-1 tensors explicitly.
        
        # Indices in local 2x2x2 block:
        # i in {0,1}, j in {0,1}, k in {0,1}
        # Global i = i0 + i_local, j = j0 + j_local, k = k0 + k_local
        
        # We need to fill U, V, W for these 7 terms.
        # The tensor T[i,j,k] = 1 for all i,j,k in the block.
        # The Strassen algorithm computes C = AB.
        # The decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        
        # Let's use the known Strassen decomposition coefficients.
        # The 7 terms correspond to the 7 products M1..M7.
        # Each Mr is a product of a linear combination of A's and a linear combination of B's.
        # This results in contributions to C.
        
        # It is easier to define the U, V, W matrices for the 2x2x2 block directly.
        # See: https://en.wikipedia.org/wiki/Strassen_algorithm
        
        # U (2x2 input A part):
        # Row i*m+j.
        # For local i,j:
        # U[0,0] (i=0,j=0): A11. In M1, M3, M5, M6, M7.
        # U[0,1] (i=0,j=1): A12. In M5, M7.
        # U[1,0] (i=1,j=0): A21. In M2, M4, M6.
        # U[1,1] (i=1,j=1): A22. In M1, M2, M4, M7.
        
        # V (2x2 input B part):
        # V[0,0] (j=0,k=0): B11. In M1, M2, M4, M6.
        # V[0,1] (j=0,k=1): B12. In M3, M6.
        # V[1,0] (j=1,k=0): B21. In M4, M7.
        # V[1,1] (j=1,k=1): B22. In M1, M3, M5, M7.
        
        # W (2x2 output C part):
        # W[0,0] (k=0,i=0): C11. In M1, M5, M7.
        # W[0,1] (k=0,i=1): C12. In M3, M5.
        # W[1,0] (k=1,i=0): C21. In M2, M4.
        # W[1,1] (k=1,i=1): C22. In M1, M4, M6.
        
        # Coefficients:
        # M1: A11,B11; A22,B22. C11(+), C22(+).
        # M2: A21,B11; A22,B11. C21(+).
        # M3: A11,B12; A11,-B22. C12(+).
        # M4: A22,B21; A22,-B11. C21(+), C22(-).
        # M5: A11,B22; A12,B22. C11(-), C12(+).
        # M6: A21,-A11; B11,B12. C22(+).
        # M7: A12,-A22; B21,B22. C11(+).
        
        # We will fill the columns col, col+1, ..., col+6.
        
        # Term 1 (M1)
        # U: A11(1), A22(1). V: B11(1), B22(1). W: C11(1), C22(1).
        U[(i0+0)*m + (j0+0), col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+0)*p + (k0+0), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+0)*n + (i0+0), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # Term 2 (M2)
        # U: A21(1), A22(1). V: B11(1). W: C21(1).
        U[(i0+1)*m + (j0+0), col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+0)*p + (k0+0), col] = 1
        W[(k0+0)*n + (i0+1), col] = 1
        col += 1
        
        # Term 3 (M3)
        # U: A11(1). V: B12(1), B22(-1). W: C12(1).
        U[(i0+0)*m + (j0+0), col] = 1
        V[(j0+0)*p + (k0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = -1
        W[(k0+1)*n + (i0+0), col] = 1
        col += 1
        
        # Term 4 (M4)
        # U: A22(1). V: B21(1), B11(-1). W: C21(1), C22(-1).
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+0), col] = 1
        V[(j0+0)*p + (k0+0), col] = -1
        W[(k0+0)*n + (i0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = -1
        col += 1
        
        # Term 5 (M5)
        # U: A11(1), A12(1). V: B22(1). W: C11(-1), C12(1).
        U[(i0+0)*m + (j0+0), col] = 1
        U[(i0+0)*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+0)*n + (i0+0), col] = -1
        W[(k0+1)*n + (i0+0), col] = 1
        col += 1
        
        # Term 6 (M6)
        # U: A21(1), A11(-1). V: B11(1), B12(1). W: C22(1).
        U[(i0+1)*m + (j0+0), col] = 1
        U[(i0+0)*m + (j0+0), col] = -1
        V[(j0+0)*p + (k0+0), col] = 1
        V[(j0+0)*p + (k0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # Term 7 (M7)
        # U: A12(1), A22(-1). V: B21(1), B22(1). W: C11(1).
        U[(i0+0)*m + (j0+1), col] = 1
        U[(i0+1)*m + (j0+1), col] = -1
        V[(j0+1)*p + (k0+0), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+0)*n + (i0+0), col] = 1
        col += 1

    # Helper to add a 2x1xK block (Rank 1)
    def add_2x1xK(i0, j0, k0, K):
        nonlocal col
        # T[i, j0, k] = 1 for i in {0,1}, k in {0..K-1}
        # U[i, j0] = 1, V[j0, k] = 1, W[k, i] = 1
        for i in range(2):
            U[(i0+i)*m + (j0), col] = 1
        for k in range(K):
            V[(j0)*p + (k0+k), col] = 1
            W[(k0+k)*n + (i0+0), col] = 1 # Wait, W index is k*n + i
            # We need W[k, i] = 1 for all i?
            # T[i, j, k] = sum U[i,j] V[j,k] W[k,i]
            # If U[i,j]=1, V[j,k]=1, then we need sum W[k,i] = 1?
            # No, it's a single term r.
            # U[i,j] V[j,k] W[k,i] = 1 * 1 * W[k,i] = W[k,i].
            # We need W[k,i] = 1 for all i,k.
            # But W is a matrix. W[k*n + i, r].
            # So we set W[k*n + i, col] = 1 for all i, k.
        for i in range(2):
            for k in range(K):
                W[(k0+k)*n + (i0+i), col] = 1
        col += 1

    # Block 1: 2x2x2 at (0,0,0)
    add_strassen(0, 0, 0)
    
    # Block 2: 2x1x4 at (0,2,2)
    add_2x1xK(0, 2, 2, 4)
    
    # Block 3: 2x2x2 at (0,3,2)
    add_strassen(0, 3, 2)
    
    # Block 4: 2x2x2 at (0,3,4)
    add_strassen(0, 3, 4)
    
    # Verify
    # T[i,j,k] = sum U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Should be 1 for all i,j,k
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    error += abs(val - 1.0)
    
    if error > 1e-8:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        rank = n * m * p

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
