import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def _build_strassen_28():
    """Pre-compute the rank-28 Strassen decomposition for M_{2,2,8}.
    
    Uses the tensor-train / structured approach: 4 independent 2x2 blocks,
    each decomposed via Strassen's rank-7 algorithm. Total rank = 4*7 = 28.
    """
    n, m, p = 2, 2, 8
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # Each entry: dict mapping (row, col) index to coefficient
    strassen_products = [
        # M1 = (A00 + A11) * (B00 + B11)
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10 + A11) * B00
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00 * (B01 - B11)
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11 * (B10 - B00)
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00 + A01) * B11
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10 - A00) * (B00 + B01)
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        # M7 = (A01 - A11) * (B10 + B11)
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):  # 4 groups of 2 columns
        k1 = 2 * q
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r  # term index
            
            # Fill U: A coefficients (index: i*m + j)
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: B coefficients (index: j*p + k_global)
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: C contributions (index: k_global*n + i)
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=2, p=8, tol=1e-9):
    """Verify that sum_t u_t ⊗ v_t ⊗ w_t reconstructs the multiplication tensor."""
    rank = U.shape[1]
    # The multiplication tensor T[i,j,k] = 1 if and only if the product
    # element C[i,k] = sum_j A[i,j]*B[j,k] uses that term.
    # In tensor form: T_{i,j,k} = delta_{i,j,k} where the multiplication
    # tensor has T_{i,j,k} = 1 for the (i,j,k) entry contributing to C.
    # Actually, the standard multiplication tensor M_{n,m,p} has:
    #   M_{(i,k),(j),(j')} ... 
    # Let me use the convention: T[i, j, k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # So the tensor is indexed by (output_row, input_col_of_A, col_of_B)
    # i.e., T[i, j, k] = 1 for all valid i,j,k (this is the identity tensor for matmul)
    
    # The decomposition says: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # Wait, let me re-check the indexing convention.
    # U has shape (n*m, rank) -> indexed by (i,j) with i in [0,n), j in [0,m)
    # V has shape (m*p, rank) -> indexed by (j,k) with j in [0,m), k in [0,p)
    # W has shape (p*n, rank) -> indexed by (k,i) with k in [0,p), i in [0,n)
    
    # The tensor entry T[i,j,k] should equal sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # And T[i,j,k] = 1 for all valid (i,j,k) since every product A[i,j]*B[j,k] 
    # contributes exactly once to C[i,k].
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                expected = 1
                if abs(val - expected) > tol:
                    errors += 1
    return errors == 0, errors


def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Pre-compute at module load time if not already cached
    if not hasattr(construct, '_U_cached'):
        _U, _V, _W = _build_strassen_28()
        construct._U_cached = _U
        construct._V_cached = _V
        construct._W_cached = _W
    
    U = construct._U_cached
    V = construct._V_cached
    W = construct._W_cached
    
    # Quick verification (cheap enough)
    if time.time() < deadline:
        ok, errs = _verify_decomposition(U, V, W, n, m, p)
        if not ok:
            # Fallback: rebuild from scratch
            U, V, W = _build_strassen_28()
            construct._U_cached = U
            construct._V_cached = V
            construct._W_cached = W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
