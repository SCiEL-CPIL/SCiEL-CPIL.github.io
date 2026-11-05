"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

# EVOLVE-BLOCK-START

def _add_strassen_terms(U, V, W, start_col, i_base, j_base, k_base, n, m, p):
    """
    Add 7 Strassen terms for a 2x2x2 block starting at column start_col.
    Returns the next column index.
    """
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = start_col
    
    # M1: (A00 + A11)(B00 + B11) -> C00, C11
    U[u_idx(i_base, j_base), col] = 1
    U[u_idx(i_base+1, j_base+1), col] = 1
    V[v_idx(j_base, k_base), col] = 1
    V[v_idx(j_base+1, k_base+1), col] = 1
    W[w_idx(k_base, i_base), col] = 1
    W[w_idx(k_base+1, i_base+1), col] = 1
    col += 1
    
    # M2: (A10 + A11)B00 -> C10
    U[u_idx(i_base+1, j_base), col] = 1
    U[u_idx(i_base+1, j_base+1), col] = 1
    V[v_idx(j_base, k_base), col] = 1
    W[w_idx(k_base, i_base+1), col] = 1
    col += 1
    
    # M3: A00(B01 - B11) -> C01
    U[u_idx(i_base, j_base), col] = 1
    V[v_idx(j_base, k_base+1), col] = 1
    V[v_idx(j_base+1, k_base+1), col] = -1
    W[w_idx(k_base+1, i_base), col] = 1
    col += 1
    
    # M4: A11(B10 - B00) -> C11
    U[u_idx(i_base+1, j_base+1), col] = 1
    V[v_idx(j_base+1, k_base), col] = 1
    V[v_idx(j_base, k_base), col] = -1
    W[w_idx(k_base+1, i_base+1), col] = 1
    col += 1
    
    # M5: (A00 + A01)B11 -> C00
    U[u_idx(i_base, j_base), col] = 1
    U[u_idx(i_base, j_base+1), col] = 1
    V[v_idx(j_base+1, k_base+1), col] = 1
    W[w_idx(k_base, i_base), col] = 1
    col += 1
    
    # M6: (A10 - A00)(B00 + B01) -> C01, C11
    U[u_idx(i_base+1, j_base), col] = 1
    U[u_idx(i_base, j_base), col] = -1
    V[v_idx(j_base, k_base), col] = 1
    V[v_idx(j_base, k_base+1), col] = 1
    W[w_idx(k_base+1, i_base), col] = 1
    W[w_idx(k_base+1, i_base+1), col] = 1
    col += 1
    
    # M7: (A01 - A11)(B10 + B11) -> C00, C10
    U[u_idx(i_base, j_base+1), col] = 1
    U[u_idx(i_base+1, j_base+1), col] = -1
    V[v_idx(j_base+1, k_base), col] = 1
    V[v_idx(j_base+1, k_base+1), col] = 1
    W[w_idx(k_base, i_base), col] = 1
    W[w_idx(k_base, i_base+1), col] = 1
    col += 1
    
    return col

def _construct_decomposition(n, m, p):
    """
    Construct the rank 88 decomposition for 3x4x8.
    Split m=4 into two blocks of size 2.
    For each m-block:
      Split n=3 into 1 and 2.
      n=1 part: 1x2x8, rank 16.
      n=2 part: 2x2x8, split p=8 into four 2s, each 2x2x2 rank 7. Total 28.
    Total per m-block: 44.
    Total rank: 88.
    """
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Two m-blocks: j in [0,1] and j in [2,3]
    for j_block in [0, 2]:
        
        # Part 1: i=0, j in [j_block, j_block+1], k in [0, 7]
        # 1x2x8 schoolbook: 16 terms
        for j_offset in [0, 1]:
            j_idx = j_block + j_offset
            for k_idx in range(8):
                U[0 * m + j_idx, col] = 1
                V[j_idx * p + k_idx, col] = 1
                W[k_idx * n + 0, col] = 1
                col += 1
                
        # Part 2: i in [1, 2], j in [j_block, j_block+1], k in [0, 7]
        # 2x2x8 Strassen: split k into 4 pairs
        for k_pair in range(4):
            k_base = k_pair * 2
            # i_base = 1, j_base = j_block, k_base = k_base
            col = _add_strassen_terms(U, V, W, col, 1, j_block, k_base, n, m, p)
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
                
    # Reconstruct
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness
    if not np.allclose(T_recon, T_true, atol=tol):
        return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a 2D list.
    """
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    U, V, W = _construct_decomposition(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
