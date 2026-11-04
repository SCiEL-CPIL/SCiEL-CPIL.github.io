import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication target tensor."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that the decomposition is correct."""
    X = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
    return np.allclose(X, T, atol=tol)

def _add_block_schoolbook(U, V, W, col_offset, n_start, m_start, p_start, n_size, m_size, p_size, n, m, p):
    """
    Add a schoolbook decomposition for a sub-block.
    n_start, m_start, p_start are the starting indices in the global n, m, p dimensions.
    n_size, m_size, p_size are the sizes of the sub-block.
    n, m, p are the global dimensions.
    """
    rank = n_size * m_size * p_size
    for i in range(n_size):
        for j in range(m_size):
            for k in range(p_size):
                # Global indices
                gi = n_start + i
                gj = m_start + j
                gk = p_start + k
                
                # U index: gi * m + gj
                u_idx = gi * m + gj
                # V index: gj * p + gk
                v_idx = gj * p + gk
                # W index: gk * n + gi
                w_idx = gk * n + gi
                
                col = col_offset + (i * m_size + j) * p_size + k
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                
    return rank

def _add_block_strassen(U, V, W, col_offset, n_start, m_start, p_start, n, m, p):
    """
    Add a Strassen decomposition for a 2x2x2 sub-block.
    Assumes n_size=2, m_size=2, p_size=2.
    """
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A indices (i, j) -> 0..3
    # B indices (j, k) -> 0..3
    # C indices (k, i) -> 0..3
    
    # Local indices
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # We need to map local (i,j) to global U index
    # Local i in 0..1, j in 0..1
    # Global i = n_start + i
    # Global j = m_start + j
    
    # Precompute global indices for U, V, W
    u_idx = [[(n_start + i) * m + (m_start + j) for j in range(2)] for i in range(2)]
    v_idx = [[(m_start + j) * p + (p_start + k) for k in range(2)] for j in range(2)]
    w_idx = [[(p_start + k) * n + (n_start + i) for i in range(2)] for k in range(2)]
    
    # Strassen steps
    # Each step r corresponds to one multiplication M_r
    # We need to define the coefficients for A, B, and C for each M_r
    
    # M1
    # A: (0,0)=1, (1,1)=1
    # B: (0,0)=1, (1,1)=1
    # C: (0,0)=1, (1,1)=1
    # M2
    # A: (1,0)=1, (1,1)=1
    # B: (0,0)=1
    # C: (1,0)=1
    # M3
    # A: (0,0)=1
    # B: (0,1)=1, (1,1)=-1
    # C: (0,1)=1
    # M4
    # A: (1,1)=1
    # B: (1,0)=1, (0,0)=-1
    # C: (0,0)=1, (1,0)=1
    # M5
    # A: (0,0)=1, (0,1)=1
    # B: (1,1)=1
    # C: (0,0)=-1, (0,1)=1
    # M6
    # A: (1,0)=1, (0,0)=-1
    # B: (0,0)=1, (0,1)=1
    # C: (1,1)=1
    # M7
    # A: (0,1)=1, (1,1)=-1
    # B: (1,0)=1, (1,1)=1
    # C: (0,0)=1
    
    # List of (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are dicts {(i,j): val}
    
    steps = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]), # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]), # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]), # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]), # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]), # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]) # M7
    ]
    
    rank = 0
    for A_c, B_c, C_c in steps:
        for (i, j, val) in A_c:
            U[u_idx[i][j], col_offset + rank] = val
        for (j, k, val) in B_c:
            V[v_idx[j][k], col_offset + rank] = val
        for (k, i, val) in C_c:
            W[w_idx[k][i], col_offset + rank] = val
        rank += 1
        
    return rank

def construct():
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Total rank estimate
    # 1x5x6 (30) + 2x2x6 (21) + 2x3x6 (36) = 87
    total_rank = 87
    
    U = np.zeros((size_u, total_rank))
    V = np.zeros((size_v, total_rank))
    W = np.zeros((size_w, total_rank))
    
    col = 0
    
    # Block 1: n=0, m=0..5, p=0..6 (1x5x6)
    r = _add_block_schoolbook(U, V, W, col, 0, 0, 0, 1, 5, 6, n, m, p)
    col += r
    
    # Block 2: n=1..3, m=0..2, p=0..6 (2x2x6)
    # Split p into 0..2, 2..4, 4..6
    for p_start in [0, 2, 4]:
        r = _add_block_strassen(U, V, W, col, 1, 0, p_start, n, m, p)
        col += r
        
    # Block 3: n=1..3, m=2..5, p=0..6 (2x3x6)
    # Split p into 0..3, 3..6
    for p_start in [0, 3]:
        r = _add_block_schoolbook(U, V, W, col, 1, 2, p_start, 2, 3, 3, n, m, p)
        col += r
        
    # Verify
    T = _build_target_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
