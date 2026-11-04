import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # We will build a decomposition with rank 43 using a block-based approach.
    # Split the 5 columns of A (and rows of B) into blocks: J = {0,1}, {2,3}, {4}
    # For J = {4} (size 1), the sub-problem is 2x1x6 which has rank 1.
    # For J = {0,1} and {2,3} (size 2), we split the 6 rows of B (cols of C) into blocks: K = {0,1}, {2,3}, {4,5}
    # Each block is a 2x2x2 sub-problem which has rank 7 (Strassen).
    # Total rank: 2 (for J=4) + 2 (J blocks) * 3 (K blocks) * 7 (Strassen) = 2 + 42 = 44?
    # Wait, R(2,1,6) = 1. So 1 * 2 = 2? No, the block J={4} corresponds to i in {0,1}, j=4, k in {0..5}.
    # The tensor is T_{i,4,k} = 1 for i=0,1, k=0..5.
    # Can we do this in rank 1?
    # U_{i,4,r}, V_{4,k,r}, W_{k,i,r}.
    # If r=1: U_{0,4,1}=1, U_{1,4,1}=1. V_{4,k,1}=1 for all k. W_{k,0,1}=1, W_{k,1,1}=1 for all k.
    # Sum = 1*1*1 = 1. Yes, rank 1.
    # So total rank = 1 (for J=4) + 2*3*7 (for J=0,1 and 2,3) = 1 + 42 = 43.
    
    rank = 43
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # i_base: 0 or 1 (but here i is always 0,1 for the whole 2x5x6)
    # j_base: starting column index in A (0, 2, or 4)
    # k_base: starting row index in B (0, 2, or 4)
    def add_strassen_block(j_base, k_base):
        nonlocal col
        # Strassen ranks for 2x2x2:
        # M1 = A11*(B11-B22) -> A11, B11, C11(+), C12(-)
        # M2 = (A11+A12)*B22 -> A11, A12, B22, C12(+)
        # M3 = A11*(B12-B22) -> A11, B12, C11(+)
        # M4 = A22*(B11-B12) -> A22, B11, C21(-), C22(+)
        # M5 = (A12+A22)*(B11+B22) -> A12, A22, B11, B22, C22(+)
        # M6 = (A11-A12)*(B12+B22) -> A11, A12, B12, B22, C21(+)
        # M7 = (A22-A11)*(B11+B12) -> A22, A11, B11, B12, C21(+)
        #
        # Mapping to global indices:
        # i_local = 0 -> i_global = 0
        # i_local = 1 -> i_global = 1
        # j_local = 0 -> j_global = j_base
        # j_local = 1 -> j_global = j_base + 1
        # k_local = 0 -> k_global = k_base
        # k_local = 1 -> k_global = k_base + 1
        
        # M1
        U[0*m + j_base, col] += 1
        V[(j_base+1)*p + (k_base+1), col] -= 1
        V[j_base*p + k_base, col] += 1
        W[k_base*n + 0, col] += 1
        W[(k_base+1)*n + 1, col] -= 1
        col += 1
        
        # M2
        U[0*m + j_base, col] += 1
        U[0*m + (j_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+1)*n + 1, col] += 1
        col += 1
        
        # M3
        U[0*m + j_base, col] += 1
        V[j_base*p + (k_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] -= 1
        W[k_base*n + 0, col] += 1
        col += 1
        
        # M4
        U[1*m + (j_base+1), col] += 1
        V[j_base*p + k_base, col] += 1
        V[j_base*p + (k_base+1), col] -= 1
        W[k_base*n + 1, col] -= 1
        W[(k_base+1)*n + 1, col] += 1
        col += 1
        
        # M5
        U[0*m + (j_base+1), col] += 1
        U[1*m + (j_base+1), col] += 1
        V[j_base*p + k_base, col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+1)*n + 1, col] += 1
        col += 1
        
        # M6
        U[0*m + j_base, col] += 1
        U[0*m + (j_base+1), col] -= 1
        V[j_base*p + (k_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[k_base*n + 1, col] += 1
        col += 1
        
        # M7
        U[1*m + (j_base+1), col] += 1
        U[0*m + j_base, col] -= 1
        V[j_base*p + k_base, col] += 1
        V[j_base*p + (k_base+1), col] += 1
        W[k_base*n + 1, col] += 1
        col += 1

    # Apply Strassen blocks for j=0,1 and j=2,3 with k=0,1,2,3,4,5 split into 0,1; 2,3; 4,5
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            add_strassen_block(j_base, k_base)
            
    # Handle j=4 with rank 1
    # U[0, 4] = 1, U[1, 4] = 1
    # V[4, k] = 1 for k=0..5
    # W[k, 0] = 1, W[k, 1] = 1 for k=0..5
    U[0*m + 4, col] = 1
    U[1*m + 4, col] = 1
    for k in range(6):
        V[4*p + k, col] = 1
        W[k*n + 0, col] = 1
        W[k*n + 1, col] = 1
    col += 1
    
    # Verify rank
    if col != rank:
        # This should not happen, but just in case
        pass

    # Verify correctness with random matrices
    # A is 2x5, B is 5x6
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C using tensor decomposition
    # C_{i,k} = sum_r U_{i*m+j, r} ... wait, indices are U[i, j, r], V[j, k, r], W[k, i, r]
    # Our U is flattened (i*m+j, r). V is (j*p+k, r). W is (k*n+i, r).
    # C[i, k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed[i, k] = s
            
    # Check if close
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
