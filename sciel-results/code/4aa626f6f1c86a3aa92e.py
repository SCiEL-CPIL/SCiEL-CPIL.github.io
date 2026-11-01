"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 135  # seconds, safety margin before 150s kill

def _validate_decomposition(U, V, W, n, m, p):
    """Check if the tensor decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reconstruct the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Reconstruct from U, V, W: T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    return np.allclose(T, T_recon, atol=1e-10)

def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen formulas for 2x2x2
    strassen_terms = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1), (1, -1), (2, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, -1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (0, -1), (2, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (1, 1) ]),
    ]
    
    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V[v_idx, col] += coeff
            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W[w_idx, col] += coeff
    
    schoolbook_terms = [
        ([ (0, 1) ], [ (8, 1) ], [ (16, 1) ]),
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]),
        ([ (2, 1) ], [ (8, 1) ], [ (17, 1) ]),
        ([ (3, 1) ], [ (17, 1) ], [ (17, 1) ]),
    ]
    
    col_start = 4 * 7
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank_31_via_perturbation():
    """
    Try to find a rank-31 decomposition by perturbing the rank-32 solution
    and checking if one term becomes linearly dependent.
    """
    n, m, p = 2, 2, 9
    U32, V32, W32 = _build_rank_32(n, m, p)
    U32 = np.array(U32, dtype=float)
    V32 = np.array(V32, dtype=float)
    W32 = np.array(W32, dtype=float)
    
    # For each pair of terms, try to see if we can merge them
    # by finding a linear combination that still satisfies the tensor equation.
    # This is a heuristic: try random perturbations and check validity.
    
    rng = np.random.RandomState(42)
    
    # Try removing each term and seeing if the remaining 31 terms
    # can be adjusted to still work.
    for removed in range(32):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        others = [i for i in range(32) if i != removed]
        
        U_rem = U32[:, others]
        V_rem = V32[:, others]
        W_rem = W32[:, others]
        
        # Check if the remaining terms already work (they won't, since we removed one)
        # Instead, try small perturbations
        for trial in range(50):
            if time.time() - _START_TIME > _TIMEOUT - 5:
                break
            
            # Perturb the remaining terms slightly
            noise_scale = 1e-6 * (trial + 1)
            U_pert = U_rem + rng.randn(*U_rem.shape) * noise_scale
            V_pert = V_rem + rng.randn(*V_rem.shape) * noise_scale
            W_pert = W_rem + rng.randn(*W_rem.shape) * noise_scale
            
            if _validate_decomposition(U_pert, V_pert, W_pert, n, m, p):
                return (U_pert.tolist(), V_pert.tolist(), W_pert.tolist())
    
    return None

def _search_rank_below_32():
    """
    Try to find a decomposition with rank < 32 using numerical methods.
    """
    n, m, p = 2, 2, 9
    U32, V32, W32 = _build_rank_32(n, m, p)
    U32 = np.array(U32, dtype=float)
    V32 = np.array(V32, dtype=float)
    W32 = np.array(W32, dtype=float)
    
    # Check if any column is redundant
    for target in range(32):
        others = [i for i in range(32) if i != target]
        
        Uo = U32[:, others]
        Vo = V32[:, others]
        Wo = W32[:, others]
        
        Ut = U32[:, target]
        Vt = V32[:, target]
        Wt = W32[:, target]
        
        # Solve for alpha from U
        alpha, _, _, _ = np.linalg.lstsq(Uo, Ut, rcond=None)
        
        V_pred = Vo @ alpha
        W_pred = Wo @ alpha
        
        if np.allclose(Vt, V_pred, atol=1e-10) and np.allclose(Wt, W_pred, atol=1e-10):
            U = U32[:, others]
            V = V32[:, others]
            W = W32[:, others]
            if _validate_decomposition(U, V, W, n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
    
    # Try perturbation-based search
    result = _try_rank_31_via_perturbation()
    if result is not None:
        return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-31 decomposition
    if time.time() - _START_TIME < _TIMEOUT - 10:
        result = _search_rank_below_32()
        if result is not None:
            U, V, W = result
            if _validate_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-32 Strassen-based decomposition
    U, V, W = _build_rank_32(n, m, p)
    
    if _validate_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook rank 36
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
