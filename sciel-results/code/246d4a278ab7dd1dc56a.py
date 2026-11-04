"""
Program for 4x4x4 matrix multiplication tensor decomposition.
Uses a correct Kronecker product of Strassen rank-7 to achieve rank 49.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays with shape (4, 7), (4, 7), (4, 7).
    
    The 7 Strassen terms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    U[i*2+j, r] = coefficient of A[i,j] in term r
    V[j*2+k, r] = coefficient of B[j,k] in term r
    W[k*2+i, r] = coefficient of C[i,k] in term r (trace convention)
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C00
    W[1*2+1, 0] = 1  # C11
    
    # Term 2: M2 = (A10+A11)B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[0*2+1, 1] = 1  # C10
    W[1*2+1, 1] = -1 # C11
    
    # Term 3: M3 = A00(B01-B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01
    V[1*2+1, 2] = -1 # B11
    W[1*2+0, 2] = 1  # C01
    W[1*2+1, 2] = 1  # C11
    
    # Term 4: M4 = A11(B10-B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10
    V[0*2+0, 3] = -1 # B00
    W[0*2+0, 3] = 1  # C00
    W[0*2+1, 3] = 1  # C10
    
    # Term 5: M5 = (A00+A01)B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    W[0*2+0, 4] = -1 # C00
    W[1*2+0, 4] = 1  # C01
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    W[1*2+1, 5] = 1  # C11
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    W[0*2+0, 6] = 1  # C00
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    
    The idea: view 4x4 matrices as 2x2 block matrices of 2x2 submatrices.
    Apply Strassen at the block level (7 block multiplications), and each
    block multiplication is itself a 2x2*2x2 product decomposed via Strassen
    (rank 7). Total rank = 7 * 7 = 49.
    
    The Kronecker product of two tensor decompositions:
    If T1 = sum_r u1[r] v1[r] w1[r] and T2 = sum_s u2[s] v2[s] w2[s],
    then T1 ⊗ T2 = sum_{r,s} (u1[r]⊗u2[s]) (v1[r]⊗v2[s]) (w1[r]⊗w2[s])
    
    For 4x4 = (2x2)⊗(2x2):
    U[i1,i2,j1,j2, r1,r2] = U2[i1,j1,r1] * U2[i2,j2,r2]
    V[j1,j2,k1,k2, r1,r2] = V2[j1,k1,r1] * V2[j2,k2,r2]
    W[k1,k2,i1,i2, r1,r2] = W2[k1,i1,r1] * W2[k2,i2,r2]
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)  # (16, 49)
    V = np.zeros((m * p, rank), dtype=int)  # (16, 49)
    W = np.zeros((p * n, rank), dtype=int)  # (16, 49)
    
    # For 4x4 matrices:
    # i = 2*i1 + i2 (row of A, row of C)
    # j = 2*j1 + j2 (col of A, row of B)  
    # k = 2*k1 + k2 (col of B, col of C)
    # where i1, i2, j1, j2, k1, k2 in {0, 1}
    
    # The Kronecker product structure:
    # U[i, j, r1*7+r2] = U2[i1, j1, r1] * U2[i2, j2, r2]
    # V[j, k, r1*7+r2] = V2[j1, k1, r1] * V2[j2, k2, r2]
    # W[k, i, r1*7+r2] = W2[k1, i1, r1] * W2[k2, i2, r2]
    
    # Build using vectorized Kronecker product for efficiency
    # U2 has shape (4, 7), V2 has shape (4, 7), W2 has shape (4, 7)
    # U_full[i1*2+i2, j1*2+j2, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    
    # Reshape for Kronecker product:
    # U2 is (4, 7) where 4 = 2*2 (i1*2+i2, j1*2+j2) -- wait, no.
    # U2[i1*2+j1, r1] -- the index is i1*2+j1 which mixes row/col of the 2x2.
    # For Kronecker: U[i1,i2,j1,j2, r1,r2] = U2[i1,j1,r1] * U2[i2,j2,r2]
    # But U2 is indexed as U2[i*2+j, r] where i is row, j is col of 2x2.
    
    # Let's use explicit loops but optimize with numpy
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U column r: outer product of U2[:,r1] reshaped and U2[:,r2] reshaped
            # U2[:, r1] has 4 entries indexed by (i1*2+j1)
            # We need U[i1*2+i2, j1*2+j2, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            # Reshape U2[:, r1] to (2,2) with shape (i1, j1)
            u1_2d = U2[:, r1].reshape(2, 2)  # u1_2d[i1, j1]
            u2_2d = U2[:, r2].reshape(2, 2)  # u2_2d[i2, j2]
            # Kronecker: result[i1*2+i2, j1*2+j2] = u1_2d[i1,j1] * u2_2d[i2,j2]
            U_col = np.kron(u1_2d, u2_2d)  # shape (4,4), indexed [i1*2+i2, j1*2+j2]
            U[:, r] = U_col.flatten()
            
            v1_2d = V2[:, r1].reshape(2, 2)  # v1_2d[j1, k1]
            v2_2d = V2[:, r2].reshape(2, 2)  # v2_2d[j2, k2]
            V_col = np.kron(v1_2d, v2_2d)  # shape (4,4), indexed [j1*2+j2, k1*2+k2]
            V[:, r] = V_col.flatten()
            
            w1_2d = W2[:, r1].reshape(2, 2)  # w1_2d[k1, i1]
            w2_2d = W2[:, r2].reshape(2, 2)  # w2_2d[k2, i2]
            W_col = np.kron(w1_2d, w2_2d)  # shape (4,4), indexed [k1*2+k2, i1*2+i2]
            W[:, r] = W_col.flatten()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard: must return before 120s (evaluator kills at 150s)
    deadline = start_time + 110  # 10s safety margin
    
    # Try rank-49 (Kronecker product of Strassen) - primary approach
    try:
        result49 = _kronecker_decomposition(n, m, p)
        valid49, diff49 = _verify_decomp(n, m, p, result49["U"], result49["V"], result49["W"])
        
        if valid49:
            return {"U": result49["U"], "V": result49["V"], "W": result49["W"]}
        
        # If verification failed, fall through to schoolbook
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
