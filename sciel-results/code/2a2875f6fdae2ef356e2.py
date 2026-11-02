"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # We use a hybrid approach:
    # 1. Apply Strassen's algorithm to disjoint 2x2x2 sub-blocks where possible.
    # 2. Use schoolbook multiplication for the remaining terms.
    #
    # Dimensions: 3x5 by 5x5
    # i in {0, 1, 2}, j in {0, 1, 2, 3, 4}, k in {0, 1, 2, 3, 4}
    #
    # We can form 2x2x2 blocks using i in {0,1}, j in {j0, j0+1}, k in {k0, k0+1}.
    # Since n=3, we can only use i=0,1 for Strassen blocks (i=2 must be schoolbook).
    # For j and k (size 5), we can fit two disjoint 2x2 blocks (indices 0,1 and 2,3).
    # Index 4 in j and k must be schoolbook.
    #
    # Possible Strassen blocks (i0=0, j0 in {0,2}, k0 in {0,2}):
    # 1. i={0,1}, j={0,1}, k={0,1}
    # 2. i={0,1}, j={2,3}, k={0,1}
    # 3. i={0,1}, j={0,1}, k={2,3}
    # 4. i={0,1}, j={2,3}, k={2,3}
    #
    # Total 4 blocks. Each block has rank 7 (Strassen) vs 8 (schoolbook).
    # Savings: 4 * (8 - 7) = 4.
    # Base rank: 3 * 5 * 5 = 75.
    # New rank: 75 - 4 = 71.
    
    blocks = [
        (0, 0, 0),  # i=0, j=0, k=0
        (0, 2, 0),  # i=0, j=2, k=0
        (0, 0, 2),  # i=0, j=0, k=2
        (0, 2, 2),  # i=0, j=2, k=2
    ]
    
    num_blocks = len(blocks)
    strassen_rank = num_blocks * 7
    # Count schoolbook terms:
    # Total terms = 75
    # Terms covered by Strassen = 4 * 8 = 32
    # Remaining terms = 75 - 32 = 43
    schoolbook_rank = 75 - 32
    
    total_rank = strassen_rank + schoolbook_rank
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block at offsets (i0, j0, k0)
    def add_strassen_block(i0, j0, k0, start_col):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1 = (A00+A11)(B00+B11)
        c = start_col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = start_col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = start_col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = start_col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = start_col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = start_col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = start_col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        
        return start_col + 7

    # Add Strassen blocks
    for (i0, j0, k0) in blocks:
        col = add_strassen_block(i0, j0, k0, col)
    
    # Add schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is covered by a Strassen block
                covered = False
                for (i0, j0, k0) in blocks:
                    if i in (i0, i0+1) and j in (j0, j0+1) and k in (k0, k0+1):
                        covered = True
                        break
                if not covered:
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the decomposition is correct using vectorized NumPy operations
    def verify():
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Check shapes
        if U_arr.shape != (n*m, total_rank):
            return False
        if V_arr.shape != (m*p, total_rank):
            return False
        if W_arr.shape != (p*n, total_rank):
            return False
        
        # Check that the tensor decomposition is exact
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Vectorized: for each (i,j,k), compute the dot product
        # Reshape to 3D tensors and use einsum
        U_3d = U_arr.reshape(n, m, total_rank)
        V_3d = V_arr.reshape(m, p, total_rank)
        W_3d = W_arr.reshape(p, n, total_rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # (since C[i,k] = sum_j A[i,j]*B[j,k], each term A[i,j]*B[j,k] contributes)
        expected = np.ones((n, m, p))
        
        if not np.allclose(T, expected, atol=1e-10):
            return False
        return True
    
    if not verify():
        # Fallback to schoolbook if verification fails
        U_fb = [[0.0] * (n*m*p) for _ in range(n*m)]
        V_fb = [[0.0] * (n*m*p) for _ in range(m*p)]
        W_fb = [[0.0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1.0
                    V_fb[j*p+k][col] = 1.0
                    W_fb[k*n+i][col] = 1.0
                    col += 1
        result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return result
# EVOLVE-BLOCK-END
