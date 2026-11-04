"""Tensor decomposition for 3x4 @ 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _get_strassen_terms(i0, j0, k0, n, m, p):
    """
    Return the 7 Strassen terms for a 2x2 block at position (i0, j0, k0).
    A is n x m, B is m x p, C is p x n.
    Block A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[k0:k0+2, i0:i0+2]
    """
    a00 = i0 * m + j0
    a01 = i0 * m + j0 + 1
    a10 = (i0 + 1) * m + j0
    a11 = (i0 + 1) * m + j0 + 1
    
    b00 = j0 * p + k0
    b01 = j0 * p + k0 + 1
    b10 = (j0 + 1) * p + k0
    b11 = (j0 + 1) * p + k0 + 1
    
    c00 = k0 * n + i0
    c01 = k0 * n + i0 + 1
    c10 = (k0 + 1) * n + i0
    c11 = (k0 + 1) * n + i0 + 1
    
    terms = []
    
    # P1 = (a00+a10)(b00+b10) -> c00 += 1, c11 += 1
    terms.append(({a00: 1, a10: 1}, {b00: 1, b10: 1}, {c00: 1, c11: 1}))
    
    # P2 = (a01+a11)b00 -> c10 += 1
    terms.append(({a01: 1, a11: 1}, {b00: 1}, {c10: 1}))
    
    # P3 = a00(b10-b00) -> c01 += 1, c11 -= 1
    terms.append(({a00: 1}, {b10: 1, b00: -1}, {c01: 1, c11: -1}))
    
    # P4 = a10(b11-b01) -> c00 -= 1, c10 += 1
    terms.append(({a10: 1}, {b11: 1, b01: -1}, {c00: -1, c10: 1}))
    
    # P5 = (a00+a01)b11 -> c10 += 1, c11 += 1
    terms.append(({a00: 1, a01: 1}, {b11: 1}, {c10: 1, c11: 1}))
    
    # P6 = (a11-a00)(b00+b01) -> c00 += 1, c11 -= 1
    terms.append(({a11: 1, a00: -1}, {b00: 1, b01: 1}, {c00: 1, c11: -1}))
    
    # P7 = (a01-a11)(b10+b11) -> c00 -= 1
    terms.append(({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: -1}))
    
    return terms

def _build_strassen_decomposition(n, m, p):
    """
    Build a decomposition using Strassen for 2x2 blocks where possible.
    For 3x4 @ 4x6:
    Top 2 rows of A (2x4) and all of B (4x6).
    Split A_top into A_TL (2x2) and A_TR (2x2).
    Split B into B_TL (2x2), B_TR (2x4), B_BL (2x2), B_BR (2x4).
    
    Blocks:
    1. A_TL (0:2, 0:2) * B_TL (0:2, 0:2) -> C[0:2, 0:2] (Strassen, rank 7)
    2. A_TL (0:2, 0:2) * B_TR (0:2, 2:6) -> C[0:2, 2:6]
       Split B_TR into B_TR1 (0:2, 2:4) and B_TR2 (0:2, 4:6).
       Two Strassen blocks (rank 7+7=14).
    3. A_TR (0:2, 2:4) * B_BL (2:4, 0:2) -> C[0:2, 0:2] (Strassen, rank 7)
    4. A_TR (0:2, 2:4) * B_BR (2:4, 2:6) -> C[0:2, 2:6]
       Split B_BR into B_BR1 (2:4, 2:4) and B_BR2 (2:4, 4:6).
       Two Strassen blocks (rank 7+7=14).
       
    Bottom row of A (1x4) * B (4x6) -> C[2, 0:6] (Schoolbook, rank 24)
    
    Total Rank: 7 + 14 + 7 + 14 + 24 = 66.
    """
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    col = 0
    
    def add_terms(terms):
        nonlocal col
        for U_dict, V_dict, W_dict in terms:
            for u_idx, coeff in U_dict.items():
                U[u_idx, col] += coeff
            for v_idx, coeff in V_dict.items():
                V[v_idx, col] += coeff
            for w_idx, coeff in W_dict.items():
                W[w_idx, col] += coeff
            col += 1
            
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_terms(_get_strassen_terms(0, 0, 0, n, m, p))
    
    # Block 2a: A[0:2, 0:2] * B[0:2, 2:4]
    add_terms(_get_strassen_terms(0, 0, 2, n, m, p))
    
    # Block 2b: A[0:2, 0:2] * B[0:2, 4:6]
    add_terms(_get_strassen_terms(0, 0, 4, n, m, p))
    
    # Block 3: A[0:2, 2:4] * B[2:4, 0:2]
    add_terms(_get_strassen_terms(0, 2, 0, n, m, p))
    
    # Block 4a: A[0:2, 2:4] * B[2:4, 2:4]
    add_terms(_get_strassen_terms(0, 2, 2, n, m, p))
    
    # Block 4b: A[0:2, 2:4] * B[2:4, 4:6]
    add_terms(_get_strassen_terms(0, 2, 4, n, m, p))
    
    # Bottom row: A[2, 0:4] * B[0:4, 0:6]
    # Schoolbook terms for i=2
    for j in range(m):
        for k in range(p):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
            
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return U, V, W, col

def _try_rank_reduction(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of rank target_rank by merging schoolbook terms
    through linear combinations. We start with the schoolbook decomposition and
    attempt to reduce the rank by finding linear dependencies.
    """
    if time.time() > deadline:
        return None
    
    # Build schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    rank_sb = n * m * p
    
    if target_rank >= rank_sb:
        return (U_sb, V_sb, W_sb)
    
    # For rank reduction, we look for linear combinations of columns
    # that can be merged. This is a hard problem, so we use a heuristic:
    # Try to find if there's a rank-r decomposition by solving a system.
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find U, V, W with r columns such that this holds.
    
    # This is a non-convex problem. For small rank reductions, we can try
    # to find linear combinations of schoolbook terms that sum to zero.
    
    # A simpler approach: use the fact that if we have a rank-r decomposition,
    # the tensor lies in a certain variety. We can try random initializations
    # and use gradient descent, but that's expensive.
    
    # For now, return None to indicate we couldn't find a better decomposition.
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140
    
    # Start with schoolbook as guaranteed fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    rank_school = n * m * p
    
    # Try Strassen-based decomposition
    try:
        U_str, V_str, W_str, rank_str = _build_strassen_decomposition(n, m, p)
        if _verify(U_str, V_str, W_str, n, m, p):
            if rank_str < rank_school:
                return {
                    "U": U_str.tolist(),
                    "V": V_str.tolist(),
                    "W": W_str.tolist()
                }
    except Exception:
        pass
    
    # Return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
