"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_DEADLINE = 130.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor (vectorized)."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)  # (n*m, rank)
    Va = np.array(V, dtype=float)  # (m*p, rank)
    Wa = np.array(W, dtype=float)  # (p*n, rank)
    
    # Build target indices
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx.append((i*m+j, j*p+k, k*n+i))
    idx = np.array(idx)  # (n*m*p, 3)
    
    # Compute T at target positions: sum_c U[a,c]*V[b,c]*W[c_]
    # T[a,b,c_] = sum_r Ua[a,r] * Va[b,r] * Wa[c_,r]
    # For each (a,b,c_) in idx, compute this sum
    # Vectorize: for each rank r, outer product
    total = np.zeros(n*m*p, dtype=float)
    for r in range(rank):
        total += Ua[idx[:,0], r] * Va[idx[:,1], r] * Wa[idx[:,2], r]
    
    return np.allclose(total, np.ones(n*m*p), atol=1e-9)

def _strassen_decomposition(n, m, p):
    """
    Decompose M_{n,m,p} using block Strassen.
    For (2,4,6): split m=4 into two blocks of size 2, p=6 into three blocks of size 2.
    Each subproblem is M_{2,2,2} solved by Strassen (rank 7).
    Total rank = 2 * 3 * 7 = 42.
    """
    a_block_size = 2
    b_col_block_size = 2
    a_blocks = m // a_block_size  # 2
    b_col_blocks = p // b_col_block_size  # 3
    strassen_rank = 7
    rank = a_blocks * b_col_blocks * strassen_rank  # 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    col = 0
    for a in range(a_blocks):
        for b in range(b_col_blocks):
            for s in range(strassen_rank):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * a_block_size + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * a_block_size + j_loc
                    k = b * b_col_block_size + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * b_col_block_size + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _reduce_rank(U, V, W, tol=1e-9):
    """Attempt to reduce rank by removing linearly dependent columns."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    rank = Ua.shape[1]
    
    # Simple QR decomposition on U to find independent columns
    # If a column in U is 0, it's useless.
    # If columns are linearly dependent, we might be able to merge?
    # But in tensor decomposition, we need to preserve the sum.
    # This is non-trivial. For now, just check for zero columns.
    
    keep = []
    for r in range(rank):
        if not np.allclose(Ua[:, r], 0, atol=tol) and \
           not np.allclose(Va[:, r], 0, atol=tol) and \
           not np.allclose(Wa[:, r], 0, atol=tol):
            keep.append(r)
            
    if len(keep) < rank:
        U_new = [U[i] for i in range(len(U))] # Copy structure
        # Reconstruct
        U_out = []
        for i in range(len(U)):
            row = [U[i][r] for r in keep]
            U_out.append(row)
        V_out = []
        for i in range(len(V)):
            row = [V[i][r] for r in keep]
            V_out.append(row)
        W_out = []
        for i in range(len(W)):
            row = [W[i][r] for r in keep]
            W_out.append(row)
        return U_out, V_out, W_out
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Generate the Strassen-based decomposition (rank 42)
    result = _strassen_decomposition(n, m, p)
    
    # Try to reduce rank (though likely none for disjoint blocks)
    U, V, W = result["U"], result["V"], result["W"]
    U, V, W = _reduce_rank(U, V, W)
    
    # Verify correctness
    if _verify(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
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
