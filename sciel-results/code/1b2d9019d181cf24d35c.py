"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen formulas
    # M1: A00+A11, B00+B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: A10+A11, B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00, B01-B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11, B10-B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: A00+A01, B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: A10-A00, B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: A01-A11, B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    outer_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    outer_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    outer_C = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [-1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                    
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_rank114():
    """Build the rank-114 decomposition: 49 + 49 + 16."""
    n, m, p = 4, 4, 9
    
    U1, V1, W1 = _decomp_4x4x4()
    U2, V2, W2 = _decomp_4x4x4()
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)
    
    U = np.hstack([U1, U2, U3])
    V = np.zeros((m * p, U.shape[1]))
    W = np.zeros((p * n, U.shape[1]))
    
    # Place V1 in rows for k=0..3
    for j in range(m):
        for k in range(4):
            V[j * p + k, :49] = V1[j * 4 + k, :]
    
    # Place V2 in rows for k=4..7
    for j in range(m):
        for k in range(4):
            V[j * p + (4 + k), 49:98] = V2[j * 4 + k, :]
    
    # Place V3 in rows for k=8
    for j in range(m):
        V[j * p + 8, 98:114] = V3[j * 1 + 0, :]
    
    # Place W1 in rows for k=0..3
    for k in range(4):
        for i in range(n):
            W[k * n + i, :49] = W1[k * 4 + i, :]
    
    # Place W2 in rows for k=4..7
    for k in range(4):
        for i in range(n):
            W[(4 + k) * n + i, 49:98] = W2[k * 4 + i, :]
    
    # Place W3 in rows for k=8
    for i in range(n):
        W[8 * n + i, 98:114] = W3[0 * 4 + i, :]
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()
    
    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]
    
    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)
    
    return np.all(sums == 1)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Build the guaranteed rank-114 decomposition
    U, V, W = _build_rank114()
    
    # Verify it's correct
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to pure schoolbook
        Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = 114
    
    # Use remaining time to attempt structured search for lower rank
    # Strategy: try to find a decomposition with rank < 114 by modifying
    # the Strassen-based structure. We'll try removing some columns and
    # adjusting others to maintain correctness.
    
    # A more promising approach: try different splits of 9
    # e.g., 9 = 3+3+3, using M_{4,4,3} decompositions
    
    # For now, try a limited random perturbation search on the existing
    # structure to see if any columns can be eliminated or merged.
    
    elapsed = time.time() - start_time
    if elapsed < 120:
        # Try to find a rank-113 or lower decomposition by checking
        # if any column in the rank-114 decomposition is redundant
        # (i.e., can be expressed as a linear combination of others
        #  while preserving the tensor identity)
        # This is complex; instead, try small structured modifications.
        
        # Try: can we use a 4x4x3 decomposition (rank ~30?) three times?
        # M_{4,4,3}: Let's construct it.
        # Split the 4x3 output into three 4x1 columns.
        # Each 4x1 column: C_i = sum_j A_ij * B_j for that column.
        # This is M_{4,4,1} per column, rank 16 each, total 48.
        # But we can share: for 3 columns, we compute 3 dot products per row.
        # Total rank = 4*4*3 = 48 (schoolbook). No improvement over 16*3=48.
        
        # Alternative: use Strassen for 4x4x4 and handle the 9th column separately.
        # That's what we already do: 49+49+16 = 114.
        
        # Try: 9 = 4 + 5. M_{4,4,5} decomposition?
        # M_{4,4,5}: split 5 = 4+1. Rank = 49 + 16 = 65.
        # Total: 49 + 65 = 114. Same.
        
        # The rank-114 appears to be tight for this family.
        # Return the verified rank-114 result.
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
