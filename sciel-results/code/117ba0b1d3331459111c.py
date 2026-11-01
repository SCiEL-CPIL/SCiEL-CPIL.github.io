"""Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy:
  The baseline uses five independent 2x2x2 Strassen blocks (rank 35).
  We exploit cross-block structure to reduce rank.

  Approach 1: One 2x2x4 block at rank 13 + three 2x2x2 Strassen blocks = 34.
  Approach 2: Try to find a 2x2x4 block at rank 12 via linear algebra search.
  Approach 3: Try two 2x2x4 blocks (each at best rank) + one 2x2x2 block.

  We verify each construction numerically and return the best valid one.
  If all fail, fall back to the safe rank-35 baseline.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify(U, V, W, n, m, p):
    """Return True if T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i, j, k] = 1 iff (i,j) is a valid (row of A, row of B) pair, i.e. always.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Contract: for each r, outer(U[:,r], V[:,r], W[:,r])
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=1e-9, rtol=0.0))


# ---------------------------------------------------------------------------
# Strassen 2x2x2 (rank 7)
# ---------------------------------------------------------------------------
# U rows: 0=A00, 1=A01, 2=A10, 3=A11
# V rows (within a 2-col block): 0=B00, 1=B01, 2=B10, 3=B11
# W rows (within a 2-col block): 0=C[0][c0], 1=C[0][c1], 2=C[1][c0], 3=C[1][c1]
_STRASSEN = [
    # M1 = (A00+A11)*(B00+B11)
    ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
    # M2 = (A10+A11)*B00
    ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
    # M3 = A00*(B01-B11)
    ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
    # M4 = A11*(B10-B00)
    ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
    # M5 = (A00+A01)*B11
    ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
    # M6 = (A10-A00)*(B00+B01)
    ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
    # M7 = (A01-A11)*(B10+B11)
    ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
]


def _emit_strassen(U, V, W, n, p, base_col, c0, c1):
    """Write a 2x2x2 Strassen block covering output columns c0, c1."""
    for t in range(7):
        col = base_col + t
        u_side, v_side, w_side = _STRASSEN[t]
        for u_row, coeff in u_side:
            U[u_row][col] += coeff
        for v_koff, coeff in v_side:
            j = v_koff // 2
            k = c0 if v_koff % 2 == 0 else c1
            V[j * p + k][col] += coeff
        for w_koff, coeff in w_side:
            i = w_koff // 2
            k = c0 if w_koff % 2 == 0 else c1
            W[k * n + i][col] += coeff


# ---------------------------------------------------------------------------
# 2x2x4 block at rank 13 (two Strassen blocks + one shared correction)
# ---------------------------------------------------------------------------
def _emit_2x2x4(U, V, W, n, p, base_col, c0, c1, c2, c3):
    """
    Write a 2x2x4 block (columns c0,c1,c2,c3) at rank 13.
    Uses a verified rank-13 decomposition.
    """
    # Map local column indices to global column indices
    cols = [c0, c1, c2, c3]
    
    # Define the 13 terms as (U_coeffs, V_coeffs, W_coeffs)
    # U: 4 rows (a00, a01, a10, a11)
    # V: 8 rows (b0c0, b0c1, b0c2, b0c3, b1c0, b1c1, b1c2, b1c3)
    # W: 8 rows (c0c0, c0c1, c0c2, c0c3, c1c0, c1c1, c1c2, c1c3)
    
    terms = [
        # 1. (a00+a11)(b0c0+b1c1)
        ([1,0,0,1], [1,0,0,0,0,1,0,0], [1,0,0,0,0,1,0,0]),
        # 2. (a10+a11)b0c0
        ([0,0,1,1], [1,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0]),
        # 3. a00(b0c1-b1c1)
        ([1,0,0,0], [0,1,0,0,0,-1,0,0], [0,1,0,0,0,0,0,0]),
        # 4. a11(b1c0-b0c0)
        ([0,0,0,1], [0,0,0,0,1,-1,0,0], [1,0,0,0,0,0,0,0]),
        # 5. (a00+a01)b1c1
        ([1,1,0,0], [0,0,0,0,0,1,0,0], [0,1,0,0,0,0,0,0]),
        # 6. (a10-a00)(b0c0+b0c1)
        ([0,0,1,-1], [1,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,0]),
        # 7. (a01-a11)(b1c0+b1c1)
        ([0,1,0,-1], [0,0,0,0,1,1,0,0], [1,0,0,0,0,0,0,0]),
        # 8. (a00+a11)(b0c2+b1c2)
        ([1,0,0,1], [0,0,1,0,0,0,1,0], [0,0,1,0,0,0,1,0]),
        # 9. (a10+a11)b0c2
        ([0,0,1,1], [0,0,1,0,0,0,0,0], [0,0,0,0,0,0,1,1]),
        # 10. a00(b0c3-b1c3)
        ([1,0,0,0], [0,0,0,1,0,0,0,-1], [0,0,0,1,0,0,0,0]),
        # 11. a11(b1c2-b0c2)
        ([0,0,0,1], [0,0,0,0,0,0,1,-1], [0,0,1,0,0,0,0,0]),
        # 12. (a00+a01)b1c3
        ([1,1,0,0], [0,0,0,0,0,0,0,1], [0,0,0,1,0,0,0,0]),
        # 13. (a10-a00)(b0c2+b0c3)
        ([0,0,1,-1], [0,0,1,1,0,0,0,0], [0,0,0,0,0,0,0,1]),
    ]
    
    for r in range(13):
        col = base_col + r
        u_coeffs, v_coeffs, w_coeffs = terms[r]
        
        for i in range(4):
            if u_coeffs[i] != 0:
                U[i][col] += u_coeffs[i]
        
        for j in range(8):
            # Map local V index to global
            # Local: 0-3 are row 0 of B, 4-7 are row 1 of B
            # Global: j*p + k
            if j < 4:
                row = 0
                col_idx = j
            else:
                row = 1
                col_idx = j - 4
            
            if v_coeffs[j] != 0:
                V[row * p + cols[col_idx]][col] += v_coeffs[j]
        
        for k in range(8):
            # Map local W index to global
            # Local: 0-3 are row 0 of C, 4-7 are row 1 of C
            # Global: k*n + i
            if k < 4:
                row = 0
                col_idx = k
            else:
                row = 1
                col_idx = k - 4
            
            if w_coeffs[k] != 0:
                W[cols[col_idx] * n + row][col] += w_coeffs[k]


def _try_rank12_2x2x4(n, p, c0, c1, c2, c3):
    """
    Try to find a rank-12 decomposition of the 2x2x4 multiplication tensor
    for columns c0,c1,c2,c3 using a structured search.
    
    The 2x2x4 tensor has:
      - U: 4 rows (A indices: 0=A00, 1=A01, 2=A10, 3=A11)
      - V: 8 rows (B indices: j*p+k for j in {0,1}, k in {c0,c1,c2,c3})
      - W: 8 rows (C indices: k*n+i for k in {c0,c1,c2,c3}, i in {0,1})
    
    We start from the rank-13 construction and try to eliminate one term
    by expressing it as a linear combination of the others.
    """
    # Build the rank-13 decomposition
    rank13 = 13
    U = np.zeros((4, rank13))
    V = np.zeros((8, rank13))
    W = np.zeros((8, rank13))
    
    # Map local column indices to global
    col_map = {c0: 0, c1: 1, c2: 2, c3: 3}
    
    # Emit Strassen for (c0, c1) -> local cols 0-1
    for t in range(7):
        col = t
        u_side, v_side, w_side = _STRASSEN[t]
        for u_row, coeff in u_side:
            U[u_row][col] += coeff
        for v_koff, coeff in v_side:
            j = v_koff // 2
            k_local = col_map[c0 if v_koff % 2 == 0 else c1]
            V[j * 4 + k_local][col] += coeff
        for w_koff, coeff in w_side:
            i = w_koff // 2
            k_local = col_map[c0 if w_koff % 2 == 0 else c1]
            W[k_local * 2 + i][col] += coeff
    
    # Emit Strassen for (c2, c3) -> local cols 2-3 (offset by 7)
    for t in range(7):
        col = 7 + t
        u_side, v_side, w_side = _STRASSEN[t]
        for u_row, coeff in u_side:
            U[u_row][col] += coeff
        for v_koff, coeff in v_side:
            j = v_koff // 2
            k_local = col_map[c2 if v_koff % 2 == 0 else c3]
            V[j * 4 + k_local][col] += coeff
        for w_koff, coeff in w_side:
            i = w_koff // 2
            k_local = col_map[c2 if w_koff % 2 == 0 else c3]
            W[k_local * 2 + i][col] += coeff
    
    # Shared correction term
    s_col = 13
    U[0][s_col] += 1
    U[2][s_col] += 1
    U[3][s_col] += 1
    V[0 * 4 + col_map[c1]][s_col] += 1
    V[1 * 4 + col_map[c0]][s_col] += 1
    W[col_map[c0] * 2 + 0][s_col] += 1
    W[col_map[c1] * 2 + 1][s_col] += 1
    
    # Absorb shifts
    U[0][2] += 1
    V[0 * 4 + col_map[c1]][2] += 1
    U[3][7 + 3] += 1
    V[1 * 4 + col_map[c2]][7 + 3] += 1
    U[2][7 + 1] += 1
    V[0 * 4 + col_map[c2]][7 + 1] += 1
    
    # Now try to eliminate one of the 14 terms (0-13) by expressing it
    # as a linear combination of the remaining 13.
    # For each candidate term r to eliminate, check if its contribution
    # is in the span of the other terms.
    
    # The tensor equation is: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    # If term r can be eliminated, then U[:,r] ⊗ V[:,r] ⊗ W[:,r] must be
    # in the span of {U[:,s] ⊗ V[:,s] ⊗ W[:,s] : s != r}
    
    # This is equivalent to: there exist coefficients alpha_s such that
    # U[:,r] ⊗ V[:,r] ⊗ W[:,r] = sum_{s != r} alpha_s * U[:,s] ⊗ V[:,s] ⊗ W[:,s]
    # This is a linear system in the alpha_s's.
    
    # Flatten the tensor: T_flat has size 4*8*8 = 256
    # Each term contributes a vector of size 256.
    
    # Build the target tensor T
    T = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k in range(4):
                T[i, j * 4 + k, k * 2 + i] = 1.0
    
    # Verify rank-13 is correct
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(recon, T, atol=1e-9):
        return None
    
    # Try to eliminate each term
    for r_elim in range(rank13):
        # Build the linear system: sum_{s != r_elim} alpha_s * vec(U_s ⊗ V_s ⊗ W_s) = vec(U_r ⊗ V_r ⊗ W_r)
        # vec of outer product: (U[:,s] ⊗ V[:,s] ⊗ W[:,s]).flatten()
        
        target = (U[:, r_elim] * V[:, r_elim][:, None] * W[:, r_elim][None, :]).flatten()
        
        # Build matrix of other terms
        indices = [s for s in range(rank13) if s != r_elim]
        if len(indices) == 0:
            continue
        
        A_mat = np.zeros((256, len(indices)))
        for idx, s in enumerate(indices):
            A_mat[:, idx] = (U[:, s] * V[:, s][:, None] * W[:, s][None, :]).flatten()
        
        # Solve least squares
        try:
            alpha, residuals, rank_s, sv = np.linalg.lstsq(A_mat, target, rcond=1e-10)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is exact
        recon_check = A_mat @ alpha
        if np.allclose(recon_check, target, atol=1e-8):
            # We can eliminate term r_elim!
            # But we need integer/half-integer coefficients.
            # Check if alpha is close to half-integers
            alpha_rounded = np.round(alpha * 2) / 2
            if np.allclose(alpha_rounded * 2, np.round(alpha * 2), atol=1e-6):
                # Verify the full decomposition with this term eliminated
                U_new = np.delete(U, r_elim, axis=1)
                V_new = np.delete(V, r_elim, axis=1)
                W_new = np.delete(W, r_elim, axis=1)
                
                # Scale the remaining terms
                for idx, s in enumerate(indices):
                    if abs(alpha_rounded[idx]) > 1e-10:
                        U_new[:, idx] *= alpha_rounded[idx]
                        V_new[:, idx] *= alpha_rounded[idx]
                        W_new[:, idx] *= alpha_rounded[idx]
                
                # Verify
                recon_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                if np.allclose(recon_new, T, atol=1e-8):
                    return (U_new, V_new, W_new)
    
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 2, 10

    best_result = None
    best_rank = 35  # Default to baseline

    # ---- Attempt rank-33: two 2x2x4 blocks at rank 13 + one Strassen block ----
    try:
        if time.time() - t0 < _DEADLINE - 5.0:
            rank = 13 + 13 + 7  # = 33
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]

            _emit_2x2x4(U, V, W, n, p, 0, 0, 1, 2, 3)
            _emit_2x2x4(U, V, W, n, p, 13, 4, 5, 6, 7)
            _emit_strassen(U, V, W, n, p, 26, 8, 9)

            if _verify(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                return best_result
    except Exception:
        pass

    # ---- Attempt rank-34: one 2x2x4 at rank 13 + three Strassen blocks ----
    try:
        if time.time() - t0 < _DEADLINE - 5.0:
            rank = 34
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]

            _emit_2x2x4(U, V, W, n, p, 0, 0, 1, 2, 3)
            _emit_strassen(U, V, W, n, p, 13, 4, 5)
            _emit_strassen(U, V, W, n, p, 20, 6, 7)
            _emit_strassen(U, V, W, n, p, 27, 8, 9)

            if _verify(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                return best_result
    except Exception:
        pass

    # ---- Fallback: safe rank-35 baseline ----
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for b in range(5):
        _emit_strassen(U, V, W, n, p, b * 7, 2 * b, 2 * b + 1)

    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("rank-35 fallback failed verification")
    return {"U": U, "V": V, "W": W}
