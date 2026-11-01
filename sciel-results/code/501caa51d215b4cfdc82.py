import time
import numpy as np

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0  # seconds (safety margin under 150 s wall-clock)

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition using vectorized operations."""
    rank = U.shape[1]
    # Reshape U, V, W to 3D tensors
    # U: (n*m, rank) -> (n, m, rank)
    U_3d = U.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_3d = V.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # We can do this by iterating over c or using einsum.
    # T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    # However, einsum might be slow for large ranks, but rank is small (28).
    T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    T_true = np.ones((n, m, p))
    return np.allclose(T_rec, T_true, atol=tol)

def _build_strassen_decomposition():
    """Build a rank-28 decomposition using 4 independent Strassen 2x2x2 blocks."""
    n, m, p = 2, 4, 4
    rank_total = 28
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Strassen templates (7 terms)
    # U_pattern[i, j], V_pattern[j, k], W_pattern[k, i]
    strassen_U = np.array([
        [[1,1],[1,1]],   # M1
        [[0,0],[1,1]],   # M2
        [[1,0],[0,0]],   # M3
        [[0,0],[0,1]],   # M4
        [[1,1],[0,1]],   # M5
        [[-1,0],[1,0]],  # M6
        [[0,1],[0,-1]]   # M7
    ])
    strassen_V = np.array([
        [[1,1],[1,1]],   # M1
        [[1,0],[0,0]],   # M2
        [[0,1],[0,-1]],  # M3
        [[-1,0],[1,0]],  # M4
        [[0,0],[0,1]],   # M5
        [[1,1],[0,0]],   # M6
        [[0,0],[1,1]]    # M7
    ])
    strassen_W = np.array([
        [[1,0],[0,1]],   # M1
        [[0,0],[1,0]],   # M2
        [[0,1],[0,0]],   # M3
        [[1,0],[0,1]],   # M4
        [[-1,0],[0,1]],  # M5
        [[0,0],[0,1]],   # M6
        [[1,0],[0,0]]    # M7
    ])
    
    blocks = [(0, 0), (2, 0), (0, 2), (2, 2)]
    
    for block_idx, (j_base, k_base) in enumerate(blocks):
        for t in range(7):
            col = block_idx * 7 + t
            
            # Fill U
            for i_local in range(2):
                for j_local in range(2):
                    val = strassen_U[t][i_local][j_local]
                    if val != 0:
                        U[i_local * m + (j_base + j_local), col] = val
            
            # Fill V
            for j_local in range(2):
                for k_local in range(2):
                    val = strassen_V[t][j_local][k_local]
                    if val != 0:
                        V[(j_base + j_local) * p + (k_base + k_local), col] = val
            
            # Fill W
            for k_local in range(2):
                for i_local in range(2):
                    val = strassen_W[t][k_local][i_local]
                    if val != 0:
                        W[(k_base + k_local) * n + i_local, col] = val
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build rank 28 decomposition
    U, V, W = _build_strassen_decomposition()
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fallback to schoolbook (rank 32) if verification fails
    rank_fallback = n * m * p
    U_fb = np.zeros((n * m, rank_fallback))
    V_fb = np.zeros((m * p, rank_fallback))
    W_fb = np.zeros((p * n, rank_fallback))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    return {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}

# EVOLVE-BLOCK-END
