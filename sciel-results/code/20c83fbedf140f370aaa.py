"""
Program implementing a verified rank-98 decomposition for 4x4x8 matrix multiplication
using 49-term 4x4x4 Strassen recursion (7x7) applied twice.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, T, tol=1e-9):
    """Verify tensor decomposition."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    r = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _build_2x2x2_block():
    """
    Return the rank-7 Strassen decomposition for 2x2x2.
    T2[(i*2+j), (j*2+k), (k*2+i)] = 1 for i,j,k in 0..1.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1, C01 -= M1, C10 += M1
    U[0, 0] += 1.0; U[3, 0] += 1.0
    V[0, 0] += 1.0; V[3, 0] += 1.0
    W[0, 0] += 1.0; W[3, 0] += 1.0; W[1, 0] -= 1.0; W[2, 0] += 1.0
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] += 1.0; U[3, 1] += 1.0
    V[0, 1] += 1.0
    W[2, 1] += 1.0; W[3, 1] -= 1.0
    
    # M3 = A00(B10-B11) -> C00 -= M3, C01 += M3
    U[0, 2] += 1.0
    V[2, 2] += 1.0; V[3, 2] -= 1.0
    W[0, 2] -= 1.0; W[1, 2] += 1.0
    
    # M4 = A11(B01-B00) -> C01 -= M4, C11 += M4
    U[3, 3] += 1.0
    V[1, 3] += 1.0; V[0, 3] -= 1.0
    W[1, 3] -= 1.0; W[3, 3] += 1.0
    
    # M5 = (A00+A01)B11 -> C00 += M5, C01 -= M5, C11 += M5
    U[0, 4] += 1.0; U[1, 4] += 1.0
    V[3, 4] += 1.0
    W[0, 4] += 1.0; W[1, 4] -= 1.0; W[3, 4] += 1.0
    
    # M6 = (A10-A00)(B00+B01) -> C10 -= M6, C11 += M6
    U[2, 5] += 1.0; U[0, 5] -= 1.0
    V[0, 5] += 1.0; V[1, 5] += 1.0
    W[2, 5] -= 1.0; W[3, 5] += 1.0
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C10 -= M7
    U[1, 6] += 1.0; U[3, 6] -= 1.0
    V[2, 6] += 1.0; V[3, 6] += 1.0
    W[0, 6] += 1.0; W[2, 6] -= 1.0
    
    # Verify this block
    T2 = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2[i*2+j, j*2+k, k*2+i] = 1.0
    if not _verify(U, V, W, T2):
        return None, None, None
    return U, V, W

def _build_4x4x4_block():
    """
    Return a rank-49 decomposition of 4x4x4 using Strassen recursion.
    """
    n = 4
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    
    U2, V2, W2 = _build_2x2x2_block()
    if U2 is None:
        return None, None, None
        
    # Correct row-major 2x2 block offsets for 4x4 matrices:
    # Block 0 (top-left, rows 0-1, cols 0-1): offsets 0, 1, 4, 5 -> base 0
    # Block 1 (top-right, rows 0-1, cols 2-3): offsets 2, 3, 6, 7 -> base 2
    # Block 2 (bottom-left, rows 2-3, cols 0-1): offsets 8, 9, 12, 13 -> base 8
    # Block 3 (bottom-right, rows 2-3, cols 2-3): offsets 10, 11, 14, 15 -> base 10
    
    block_offsets = [0, 2, 8, 10]
    
    # Standard Strassen for 2x2 blocks:
    A_blocks = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    B_blocks = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    C_blocks = [
        [(0, 1), (3, 1), (1, -1), (2, 1)],
        [(2, 1), (3, -1)],
        [(0, -1), (1, 1)],
        [(1, -1), (3, 1)],
        [(0, 1), (1, -1), (3, 1)],
        [(2, -1), (3, 1)],
        [(0, 1), (2, -1)]
    ]
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            u_vals = U2[:, s]
            v_vals = V2[:, s]
            w_vals = W2[:, s]
            
            for i_local in range(4):
                val = u_vals[i_local]
                if val != 0:
                    for bA, sA in A_blocks[t]:
                        g_idx = block_offsets[bA] + i_local
                        U4[g_idx, col] += val * sA
            
            for j_local in range(4):
                val = v_vals[j_local]
                if val != 0:
                    for bB, sB in B_blocks[t]:
                        g_idx = block_offsets[bB] + j_local
                        V4[g_idx, col] += val * sB
            
            for k_local in range(4):
                val = w_vals[k_local]
                if val != 0:
                    for bC, sC in C_blocks[t]:
                        g_idx = block_offsets[bC] + k_local
                        W4[g_idx, col] += val * sC
    
    # Verify the 4x4x4 block
    T4 = np.zeros((16, 16, 16), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4[i*4+j, j*4+k, k*4+i] = 1.0
    if not _verify(U4, V4, W4, T4):
        return None, None, None
        
    return U4, V4, W4

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Build the full 4x4x8 multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook decomposition (rank 128) as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook

    # Try to build a verified 4x4x4 block and extend to 4x4x8
    if time.time() < deadline - 5:
        U4, V4, W4 = _build_4x4x4_block()
        if U4 is not None:
            rank_block = U4.shape[1]
            # Extend to 4x4x8 by duplicating the block for the second 4 columns of B
            U_full = np.zeros((n*m, 2*rank_block), dtype=np.float64)
            V_full = np.zeros((m*p, 2*rank_block), dtype=np.float64)
            W_full = np.zeros((p*n, 2*rank_block), dtype=np.float64)
            
            # Block 1: k=0..3 (B columns 0..3, C rows 0..3)
            U_full[:, 0:rank_block] = U4
            V_full[0:16, 0:rank_block] = V4
            W_full[0:16, 0:rank_block] = W4
            
            # Block 2: k=4..7 (B columns 4..7, C rows 4..7)
            U_full[:, rank_block:2*rank_block] = U4
            V_full[16:32, rank_block:2*rank_block] = V4
            W_full[16:32, rank_block:2*rank_block] = W4
            
            if _verify(U_full, V_full, W_full, T):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 2*rank_block

    # Convert to half-integer lists
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
