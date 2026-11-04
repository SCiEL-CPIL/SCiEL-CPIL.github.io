"""
Rewritten program for 4x4x4 matrix multiplication tensor decomposition.
Uses Kronecker product of Strassen rank-7 decomposition to achieve rank 49.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays with shape (4, 7), (4, 7), (4, 7).
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
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
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
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
    
    # Permutation mapping from natural 4x4 index to Kronecker row index
    perm = np.empty(16, dtype=int)
    for i in range(4):
        for j in range(4):
            S = i * 4 + j
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            a = i_hi * 2 + j_hi
            b = i_lo * 2 + j_lo
            R = a * 4 + b
            perm[S] = R
    
    # Build Kronecker products
    U_kron = np.kron(U2, U2)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)
    
    # Apply permutation
    U = U_kron[perm, :]
    V = V_kron[perm, :]
    W = W_kron[perm, :]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard: must return before 120s (evaluator kills at 150s)
    deadline = start_time + 110
    
    # Primary approach: rank-49 (Kronecker product of Strassen)
    try:
        result49 = _kronecker_decomposition(n, m, p)
        
        # Quick verification using einsum
        U_arr = result49["U"]
        V_arr = result49["V"]
        W_arr = result49["W"]
        
        # Spot check a few entries to ensure correctness
        T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Check that all tensor entries are correct
        T = _build_tensor(n, m, p)
        max_diff = np.max(np.abs(T_decomp - T))
        
        if max_diff < 1e-9:
            # Convert to lists for interface compliance
            U_list = U_arr.tolist()
            V_list = V_arr.tolist()
            W_list = W_arr.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Final fallback: schoolbook (rank 64)
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
