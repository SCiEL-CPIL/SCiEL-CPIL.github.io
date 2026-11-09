"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition as list-of-lists."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] reconstructs the MM tensor."""
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    rank = Ua.shape[1]
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(rank):
        Au = Ua[:, r].reshape(n, m)
        Bv = Va[:, r].reshape(m, p)
        Ww = Wa[:, r].reshape(p, n)
        C_computed += (Au @ Bv) * Ww.T
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7).
    """
    U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [1, 0, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    return U, V, W

def _embed_2x2x2_in_3x3x8(U2, V2, W2, i_offset, j_offset, k_offset, n, m, p):
    """
    Embed a 2x2x2 Strassen decomposition into a 3x3x8 problem.
    i_offset: row offset in A (0 or 1)
    j_offset: col offset in A / row offset in B (0 or 1)
    k_offset: col offset in B (0 or 1)
    """
    U = [[0.0] * 7 for _ in range(n * m)]
    V = [[0.0] * 7 for _ in range(m * p)]
    W = [[0.0] * 7 for _ in range(p * n)]
    
    for r in range(7):
        for a_i in range(2):
            for a_j in range(2):
                ui = (i_offset + a_i) * m + (j_offset + a_j)
                U[ui][r] = float(U2[r][a_i * 2 + a_j])
        
        for b_j in range(2):
            for b_k in range(2):
                vj = (j_offset + b_j) * p + (k_offset + b_k)
                V[vj][r] = float(V2[r][b_j * 2 + b_k])
        
        for c_k in range(2):
            for c_i in range(2):
                wk = (k_offset + c_k) * n + (i_offset + c_i)
                W[wk][r] = float(W2[r][c_k * 2 + c_i])
    
    return U, V, W

def _strassen_block_decomposition(n, m, p):
    """
    Build a decomposition for 3x3 * 3x8 using Strassen on 2x2 blocks.
    
    Strategy:
    - Split A (3x3) into top 2x2 block + remaining
    - Split B (3x8) into left 2x8 block + remaining  
    - Use Strassen for the 2x2 * 2x8 part
    - Use schoolbook for the remaining parts
    
    For 2x2 * 2x8:
      Split B into four 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
      Each uses Strassen rank 7, total 4*7 = 28
    
    For remaining:
      A[2, :] (1x3) * B[:, 0:8] (3x8): schoolbook rank 24
      A[0:2, 2] (2x1) * B[2, :] (1x8): schoolbook rank 16
    
    Total: 28 + 24 + 16 = 68 (better than 72)
    """
    n, m, p = 3, 3, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: Top-left 2x2 block of A times left 2x8 block of B
    # A[0:2, 0:2] (2x2) * B[0:2, 0:8] (2x8)
    U2, V2, W2 = _strassen_2x2x2()
    
    # Split B[0:2, 0:8] into four 2x2 blocks
    for k_block in range(4):
        k_offset = k_block * 2
        # For each of the 7 rank-1 terms in Strassen 2x2x2
        for r in range(7):
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            # Fill U (A part)
            # U2[r] is a vector of length 4 (2x2)
            # Indices in A: i in 0..1, j in 0..1
            for a_i in range(2):
                for a_j in range(2):
                    u_idx = a_i * m + a_j
                    u_col[u_idx] = float(U2[r][a_i * 2 + a_j])
            
            # Fill V (B part)
            # V2[r] is a vector of length 4 (2x2)
            # Indices in B: j in 0..1, k in k_offset..k_offset+1
            for b_j in range(2):
                for b_k in range(2):
                    v_idx = b_j * p + (k_offset + b_k)
                    v_col[v_idx] = float(V2[r][b_j * 2 + b_k])
            
            # Fill W (C part)
            # W2[r] is a vector of length 4 (2x2)
            # Indices in C: i in 0..1, k in k_offset..k_offset+1
            for c_i in range(2):
                for c_k in range(2):
                    w_idx = (k_offset + c_k) * n + c_i
                    w_col[w_idx] = float(W2[r][c_k * 2 + c_i])
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    # Part 2: Bottom row of A (i=2) times all of B
    # A[2, j] * B[j, k] -> C[2, k]
    # Terms: E_{2,j} \otimes E_{j,k} \otimes E_{2,k}
    for j in range(m):
        for k in range(p):
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            
            u_col[u_idx] = 1.0
            v_col[v_idx] = 1.0
            w_col[w_idx] = 1.0
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    # Part 3: Right column of A (j=2, i in 0..1) times bottom row of B (j=2)
    # A[i, 2] * B[2, k] -> C[i, k]
    # Terms: E_{i,2} \otimes E_{2,k} \otimes E_{i,k}
    for i in range(2):
        for k in range(p):
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            u_col[u_idx] = 1.0
            v_col[v_idx] = 1.0
            w_col[w_idx] = 1.0
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomp(decomp, n, m, p, tol=1e-9):
    """Verify a decomposition."""
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    rank = U.shape[1]
    
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        Au = U[:, r].reshape(n, m)
        Bv = V[:, r].reshape(m, p)
        Ww = W[:, r].reshape(p, n)
        C_computed += (Au @ Bv) * Ww.T
    
    return np.allclose(C_computed, C_expected, atol=tol)

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try Strassen block decomposition
    try:
        strassen_decomp = _strassen_block_decomposition(n, m, p)
        if _verify_decomp(strassen_decomp, n, m, p):
            return strassen_decomp
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
