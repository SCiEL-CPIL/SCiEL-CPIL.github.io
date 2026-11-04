"""Combined program for matrix multiplication 2x3x8 tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast float-based check."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u is 4-vector (for 2x2 A), v is 4-vector (for 2x2 B), w is 4-vector (for 2x2 C).
    A is indexed [i*2+j], B is [j*2+k], C is [k*2+i].
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    """
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = []
    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]  # contributes to C00 and C11
    terms.append((u, v, [1, 0, 0, 1]))
    
    # M2 = (A10+A11)B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, 0]  # contributes to C10
    terms.append((u, v, [0, 0, 1, 0]))
    
    # M3 = A00(B10-B11)
    u = [1, 0, 0, 0]
    v = [0, 0, 1, -1]
    w = [0, 1, 0, 1]  # contributes to C01 and C11
    terms.append((u, v, [0, 1, 0, 1]))
    
    # M4 = A11(B01-B10)
    u = [0, 0, 0, 1]
    v = [0, 1, -1, 0]
    w = [1, 0, 1, 0]  # contributes to C00 and C10
    terms.append((u, v, [1, 0, 1, 0]))
    
    # M5 = (A00+A01)B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]  # contributes to C00 and C01
    terms.append((u, v, [-1, 1, 0, 0]))
    
    # M6 = (A10-A00)(B00+B10)
    u = [-1, 0, 1, 0]
    v = [1, 0, 1, 0]
    w = [0, 0, 0, 1]  # contributes to C11
    terms.append((u, v, [0, 0, 0, 1]))
    
    # M7 = (A01-A11)(B01+B11)
    u = [0, 1, 0, -1]
    v = [0, 1, 0, 1]
    w = [1, 0, 0, 0]  # contributes to C00
    terms.append((u, v, [1, 0, 0, 0]))
    
    return terms

def _build_strassen_decomposition():
    """
    Build a rank-44 decomposition using block-Strassen.
    
    For each pair of k-values (2k, 2k+1), k=0..3:
    - Use Strassen for the 2x2x2 subproblem with j=0,1: 7 terms
    - Add 4 terms for j=2: one for each (i, k_pair) combination
    
    Total: 4 * (7 + 4) = 44
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    strassen_terms = _strassen_2x2x2()
    
    for k_pair in range(4):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        
        # Add 7 Strassen terms for j=0,1
        for (u_4, v_4, w_4) in strassen_terms:
            # u_4 corresponds to A[0,0], A[0,1], A[1,0], A[1,1]
            # In our indexing: A[i,j] -> U[i*m+j] = U[i*3+j]
            # j=0: A[0,0]=U[0], A[1,0]=U[3]
            # j=1: A[0,1]=U[1], A[1,1]=U[4]
            
            # u_4 = [A00, A01, A10, A11]
            # U indices: A00->0, A01->1, A10->3, A11->4
            U[0][col] = u_4[0]   # A[0,0]
            U[1][col] = u_4[1]   # A[0,1]
            U[3][col] = u_4[2]   # A[1,0]
            U[4][col] = u_4[3]   # A[1,1]
            # U[2] (A[0,2]) and U[5] (A[1,2]) are 0 for this term
            
            # v_4 corresponds to B[0,0], B[0,1], B[1,0], B[1,1] for this k-pair
            # B[j,k] -> V[j*p+k]
            # For k-pair (k0, k1):
            # B[0,k0] -> V[0*8+k0] = V[k0]
            # B[0,k1] -> V[0*8+k1] = V[k1]
            # B[1,k0] -> V[1*8+k0] = V[8+k0]
            # B[1,k1] -> V[1*8+k1] = V[8+k1]
            
            # v_4 = [B00, B01, B10, B11] where 0/1 refer to j=0/1 and k=k0/k1
            V[k0][col] = v_4[0]       # B[0,k0]
            V[k1][col] = v_4[1]       # B[0,k1]
            V[8+k0][col] = v_4[2]     # B[1,k0]
            V[8+k1][col] = v_4[3]     # B[1,k1]
            # V for j=2 (rows 16-23) are 0 for this term
            
            # w_4 corresponds to C[0,0], C[0,1], C[1,0], C[1,1]
            # C[i,k] -> W[k*n+i] = W[k*2+i]
            # C[0,k0] -> W[k0*2+0] = W[2*k0]
            # C[0,k1] -> W[k1*2+0] = W[2*k1]
            # C[1,k0] -> W[k0*2+1] = W[2*k0+1]
            # C[1,k1] -> W[k1*2+1] = W[2*k1+1]
            
            W[2*k0][col] = w_4[0]     # C[0,k0]
            W[2*k1][col] = w_4[1]     # C[0,k1]
            W[2*k0+1][col] = w_4[2]   # C[1,k0]
            W[2*k1+1][col] = w_4[3]   # C[1,k1]
            
            col += 1
        
        # Add 4 terms for j=2
        # For each i in {0,1} and each k in {k0, k1}:
        for i in range(2):
            for k in [k0, k1]:
                U[i*3+2][col] = 1     # A[i,2]
                V[2*8+k][col] = 1     # B[2,k]
                W[k*2+i][col] = 1     # C[i,k]
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_and_clean(decomp, n, m, p):
    """Verify decomposition and clean up (remove zero columns if any)."""
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    rank = len(U[0])
    
    # Quick float check
    if not _check_fast(U, V, W, n, m, p):
        return None
    
    # Exact check
    if not _check_exact(U, V, W, n, m, p):
        return None
    
    # Remove zero columns
    new_U, new_V, new_W = [], [], []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(len(U))) or \
           any(V[i][c] != 0 for i in range(len(V))) or \
           any(W[i][c] != 0 for i in range(len(W))):
            new_U.append([U[i][c] for i in range(len(U))])
            new_V.append([V[i][c] for i in range(len(V))])
            new_W.append([W[i][c] for i in range(len(W))])
    
    # Transpose back to row-major
    new_rank = len(new_U)
    final_U = [[new_U[c][i] for c in range(new_rank)] for i in range(len(U))]
    final_V = [[new_V[c][i] for c in range(new_rank)] for i in range(len(V))]
    final_W = [[new_W[c][i] for c in range(new_rank)] for i in range(len(W))]
    
    return {"U": final_U, "V": final_V, "W": final_W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try the Strassen-based decomposition (rank 44)
    strassen_decomp = _build_strassen_decomposition()
    verified = _verify_and_clean(strassen_decomp, n, m, p)
    
    if verified is not None:
        return verified
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
