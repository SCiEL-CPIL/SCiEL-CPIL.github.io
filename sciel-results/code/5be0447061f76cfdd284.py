"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a matrix multiplication tensor decomposition.
    Uses block-wise Strassen+schoolbook decomposition for rank 55.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    deadline = time.time() + 115  # safety margin
    
    # Strategy:
    # Split j (middle dim, size 3) into j=0 and j∈{1,2}
    # Part 1: A[:,0] (2×1) × B[0,:] (1×10) → rank 20 (schoolbook)
    # Part 2: A[:,1:3] (2×2) × B[1:3,:] (2×10) → split k into 5 blocks of 2
    #   Each 2×2×2 block uses Strassen (rank 7) → 5×7 = 35
    # Total rank: 20 + 35 = 55
    
    rank = 55
    U = np.zeros((n * m, rank))  # (6, 55)
    V = np.zeros((m * p, rank))  # (30, 55)
    W = np.zeros((p * n, rank))  # (20, 55)
    
    col = 0
    
    # Part 1: j=0, schoolbook for 2×1×10
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    # col is now 20
    
    # Part 2: j∈{1,2}, k split into 5 blocks of 2, Strassen on each
    def add_strassen_block(b, col_start):
        """Add Strassen terms for k-block b (k = 2b, 2b+1)"""
        col = col_start
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # U indices for A[i,j], i∈{0,1}, j∈{1,2}
        u_A00 = 0 * m + 1
        u_A01 = 0 * m + 2
        u_A10 = 1 * m + 1
        u_A11 = 1 * m + 2
        
        # V indices for B[j,k], j∈{1,2}, k∈{k0,k1}
        v_B00 = 1 * p + k0
        v_B01 = 1 * p + k1
        v_B10 = 2 * p + k0
        v_B11 = 2 * p + k1
        
        # W indices for C[i,k], i∈{0,1}, k∈{k0,k1}
        w_C00 = k0 * n + 0
        w_C01 = k1 * n + 0
        w_C10 = k0 * n + 1
        w_C11 = k1 * n + 1
        
        # Term 1: (A00+A11)(B00+B11)(C00+C11)
        U[u_A00, col] += 1; U[u_A11, col] += 1
        V[v_B00, col] += 1; V[v_B11, col] += 1
        W[w_C00, col] += 1; W[w_C11, col] += 1
        col += 1
        
        # Term 2: (A10+A11)B00(C10)
        U[u_A10, col] += 1; U[u_A11, col] += 1
        V[v_B00, col] += 1
        W[w_C10, col] += 1
        col += 1
        
        # Term 3: A00(B10-B11)(C01+C11)
        U[u_A00, col] += 1
        V[v_B10, col] += 1; V[v_B11, col] -= 1
        W[w_C01, col] += 1; W[w_C11, col] += 1
        col += 1
        
        # Term 4: A11(B01-B10)(C00+C10)
        U[u_A11, col] += 1
        V[v_B01, col] += 1; V[v_B10, col] -= 1
        W[w_C00, col] += 1; W[w_C10, col] += 1
        col += 1
        
        # Term 5: (A00+A01)B11(C01-C00)
        U[u_A00, col] += 1; U[u_A01, col] += 1
        V[v_B11, col] += 1
        W[w_C01, col] += 1; W[w_C00, col] -= 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B10)(C11)
        U[u_A10, col] += 1; U[u_A00, col] -= 1
        V[v_B00, col] += 1; V[v_B10, col] += 1
        W[w_C11, col] += 1
        col += 1
        
        # Term 7: (A01-A11)(B01+B11)(C00)
        U[u_A01, col] += 1; U[u_A11, col] -= 1
        V[v_B01, col] += 1; V[v_B11, col] += 1
        W[w_C00, col] += 1
        col += 1
        
        return col
    
    # Add 5 Strassen blocks
    for b in range(5):
        if time.time() > deadline:
            break
        col = add_strassen_block(b, col)
    
    # Verify
    T_from_decomp = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_from_decomp[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_from_decomp, T_expected, atol=1e-10, rtol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
