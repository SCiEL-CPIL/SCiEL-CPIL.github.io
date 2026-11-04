"""
Program for 4x4x4 matrix multiplication tensor decomposition.
Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2x2,
yielding a rank-49 decomposition. This is the best known rank for 4x4 matrix multiplication.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays with shape (4, 7), (4, 7), (4, 7).
    The indices correspond to:
    U: (i, j) where i in {0,1}, j in {0,1} -> i*2+j
    V: (j, k) where j in {0,1}, k in {0,1} -> j*2+k
    W: (k, i) where k in {0,1}, i in {0,1} -> k*2+i
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
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
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # Reshape 1D vectors of length 4 (2x2) into 2x2 matrices
            u1_2d = U2[:, r1].reshape(2, 2)
            u2_2d = U2[:, r2].reshape(2, 2)
            U_col = np.kron(u1_2d, u2_2d)
            U[:, r] = U_col.flatten()
            
            v1_2d = V2[:, r1].reshape(2, 2)
            v2_2d = V2[:, r2].reshape(2, 2)
            V_col = np.kron(v1_2d, v2_2d)
            V[:, r] = V_col.flatten()
            
            w1_2d = W2[:, r1].reshape(2, 2)
            w2_2d = W2[:, r2].reshape(2, 2)
            W_col = np.kron(w1_2d, w2_2d)
            W[:, r] = W_col.flatten()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}

def _schoolbook_decomposition(n, m, p):
    """Construct the schoolbook (rank n*m*p) decomposition."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

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
        
        # If verification failed, try with relaxed tolerance
        valid49b, diff49b = _verify_decomp(n, m, p, result49["U"], result49["V"], result49["W"], tol=1e-6)
        if valid49b:
            return {"U": result49["U"], "V": result49["V"], "W": result49["W"]}
            
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        # Fall back to schoolbook
        result = _schoolbook_decomposition(n, m, p)
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Final fallback: schoolbook
    result = _schoolbook_decomposition(n, m, p)
    return {"U": result["U"], "V": result["V"], "W": result["W"]}
# EVOLVE-BLOCK-END
