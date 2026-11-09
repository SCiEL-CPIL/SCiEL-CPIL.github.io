"""Matrix multiplication 2x3 x 3x4: rank-minimal exact tensor decomposition.

Dimensions: A is (n x m) = (2 x 3), B is (m x p) = (3 x 4), C = A B is (n x p) = (2 x 4).
Convention: U index = i*m+j (A entry), V index = j*p+k (B entry), W index = k*n+i (C entry).
Decomposition: C = sum_r U[:,r] * V[:,r] * W[:,r]  (outer product along r).
Score = rank (number of terms). Lower is better.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_BUDGET = 140.0  # evaluator kills at 150s; keep 10s margin


def _tensor_index(n, m, p):
    """Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == T exactly (within tol)."""
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(reconstructed, T, atol=tol, rtol=0.0)


def _build_rank22_decomposition(n, m, p):
    """Build the rank-22 Strassen-based decomposition for 2x3 x 3x4."""
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def _add_strassen_block(a_idx, b_idx, c_idx, col_start):
        col = col_start
        # M1 = (A00+A11)(B00+B11)
        U[a_idx['00']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M2 = (A10+A11)(B00)
        U[a_idx['10']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        W[c_idx['11']][col] += -1
        col += 1
        # M3 = A00(B01-B11)
        U[a_idx['00']][col] += 1
        V[b_idx['01']][col] += 1; V[b_idx['11']][col] += -1
        W[c_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[a_idx['11']][col] += 1
        V[b_idx['10']][col] += 1; V[b_idx['00']][col] += -1
        W[c_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[a_idx['00']][col] += 1; U[a_idx['01']][col] += 1
        V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += -1
        W[c_idx['01']][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[a_idx['10']][col] += 1; U[a_idx['00']][col] += -1
        V[b_idx['00']][col] += 1; V[b_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[a_idx['01']][col] += 1; U[a_idx['11']][col] += -1
        V[b_idx['10']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        return col

    # Block 1: A_L (2x2) * B_T1 (2x2) -> C[:, 0:2]
    a1 = {'00': 0, '01': 1, '10': 3, '11': 4}
    b1 = {'00': 0, '01': 1, '10': 4, '11': 5}
    c1 = {'00': 0, '01': 2, '10': 1, '11': 3}
    col = _add_strassen_block(a1, b1, c1, 0)

    # Block 2: A_L (2x2) * B_T2 (2x2) -> C[:, 2:4]
    b2 = {'00': 2, '01': 3, '10': 6, '11': 7}
    c2 = {'00': 4, '01': 6, '10': 5, '11': 7}
    col = _add_strassen_block(a1, b2, c2, col)

    # Block 3: A_R (2x1) * B_B (1x4) -> C (2x4), rank-8 schoolbook
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1.0
            V[2 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1

    return U.tolist(), V.tolist(), W.tolist()


def _try_reduce(U, V, W, T, max_attempts=200):
    """Attempt to find a lower-rank decomposition by identifying linear
    dependencies among the rank terms."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    n, m, p = T.shape

    # Step 1: Try merging proportional terms
    merged = True
    while merged and rank > 1:
        merged = False
        for r in range(rank):
            for s in range(r + 1, rank):
                if np.linalg.norm(U[:, r]) < 1e-12 or np.linalg.norm(U[:, s]) < 1e-12:
                    continue
                if np.linalg.norm(V[:, r]) < 1e-12 or np.linalg.norm(V[:, s]) < 1e-12:
                    continue
                ratio_u = np.dot(U[:, r], U[:, s]) / np.dot(U[:, s], U[:, s])
                if (np.allclose(U[:, r], ratio_u * U[:, s], atol=1e-12) and
                        np.allclose(V[:, r], ratio_u * V[:, s], atol=1e-12) and
                        np.allclose(W[:, r], ratio_u * W[:, s], atol=1e-12)):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    merged = True
                    break
            if merged:
                break

    # Step 2: Check for zero-contribution terms
    for r in range(rank):
        all_zero = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if U[a, r] != 0 and V[b, r] != 0 and W[c, r] != 0:
                        all_zero = False
                        break
                if not all_zero:
                    break
            if not all_zero:
                break
        if all_zero:
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            break

    # Step 3: Try removing each term and solving for the residual
    for r_remove in range(rank):
        if time.time() - START_TIME > TIME_BUDGET - 15:
            break
        U_rem = np.delete(U, r_remove, axis=1)
        V_rem = np.delete(V, r_remove, axis=1)
        W_rem = np.delete(W, r_remove, axis=1).copy()
        r_new = rank - 1

        valid = True
        for c in range(p * n):
            i_c = c % n
            k_c = c // n
            if k_c >= p:
                continue
            A_sys = np.zeros((m, r_new))
            b_sys = np.ones(m)
            for j in range(m):
                a_idx = i_c * m + j
                b_idx = j * p + k_c
                A_sys[j, :] = U_rem[a_idx, :] * V_rem[b_idx, :]

            sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            if residuals is not None and len(residuals) > 0:
                if residuals[0] > 1e-8:
                    valid = False
                    break
            else:
                if not np.allclose(A_sys @ sol, b_sys, atol=1e-8):
                    valid = False
                    break
            W_rem[c, :] = sol

        if valid:
            U_new = U_rem
            V_new = V_rem
            W_new = W_rem
            if _verify(U_new, V_new, W_new, T):
                U = U_new
                V = V_new
                W = W_new
                rank = r_new
                break

    return U, V, W


def _try_rank_reduction_via_gaussian(U, V, W, T):
    """Try to reduce rank by finding linear dependencies in the matricization.
    
    The tensor T can be matricized in different ways. If the rank of a matricization
    is less than the current rank, we can potentially reduce the decomposition rank.
    
    For a decomposition of rank r, the matricization T_(a|bc) has rank at most r.
    If we can find a rank-r' < r decomposition, we need the matricizations to have rank r'.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    n, m, p = T.shape
    
    # Check matricization ranks to find theoretical lower bound
    # T_(a|bc): matrix of shape (n*m) x (m*p * p*n)
    # This is too large, so use the fact that T is sparse
    
    # For the 2x3x4 case, the tensor has 24 non-zero entries.
    # The matricization T_(a|bc) has 6 rows and 12*8=96 columns.
    # But only 24 entries are non-zero, so the effective rank is limited.
    
    # Try: for each pair of terms (r, s), see if one can be eliminated
    # by adjusting the other terms.
    
    # A more systematic approach: try all subsets of size rank-1 and check
    # if the tensor is in their span.
    
    # For rank 22, this is 22 choose 21 = 22 subsets, which is feasible.
    # For each subset, we need to check if T can be represented.
    
    # The representation condition: there exist coefficients c_s such that
    # sum_s c_s * (U[:,s] ⊗ V[:,s] ⊗ W[:,s]) = T
    # But this is not linear in c_s because of the tensor product structure.
    
    # Alternative: fix all but one set of variables and solve.
    # If we fix U and V, then for each c, W[c,:] must satisfy a linear system.
    
    # Try removing one term at a time (already done in _try_reduce).
    # Now try removing two terms at a time.
    
    if rank <= 2:
        return U, V, W
    
    for r1 in range(rank):
        if time.time() - START_TIME > TIME_BUDGET - 10:
            break
        for r2 in range(r1 + 1, rank):
            if time.time() - START_TIME > TIME_BUDGET - 10:
                break
            U_rem = np.delete(U, [r1, r2], axis=1)
            V_rem = np.delete(V, [r1, r2], axis=1)
            W_rem = np.delete(W, [r1, r2], axis=1).copy()
            r_new = rank - 2
            
            valid = True
            for c in range(p * n):
                i_c = c % n
                k_c = c // n
                if k_c >= p:
                    continue
                A_sys = np.zeros((m, r_new))
                b_sys = np.ones(m)
                for j in range(m):
                    a_idx = i_c * m + j
                    b_idx = j * p + k_c
                    A_sys[j, :] = U_rem[a_idx, :] * V_rem[b_idx, :]
                
                sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        valid = False
                        break
                else:
                    if not np.allclose(A_sys @ sol, b_sys, atol=1e-8):
                        valid = False
                        break
                W_rem[c, :] = sol
            
            if valid:
                U_new = U_rem
                V_new = V_rem
                W_new = W_rem
                if _verify(U_new, V_new, W_new, T):
                    return U_new, V_new, W_new
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    T = _tensor_index(n, m, p)

    # Start with rank-22 Strassen decomposition
    U_list, V_list, W_list = _build_rank22_decomposition(n, m, p)
    U = np.array(U_list, dtype=np.float64)
    V = np.array(V_list, dtype=np.float64)
    W = np.array(W_list, dtype=np.float64)

    if not _verify(U, V, W, T):
        # Fallback: schoolbook rank n*m*p = 2*3*4 = 24
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Save the verified rank-22 as fallback
    U_fallback = U.copy()
    V_fallback = V.copy()
    W_fallback = W.copy()

    # Attempt rank reduction
    if time.time() - START_TIME < TIME_BUDGET - 20:
        U, V, W = _try_reduce(U, V, W, T)

    # Verify after reduction
    if not _verify(U, V, W, T):
        U = U_fallback
        V = V_fallback
        W = W_fallback

    # Try further reduction with two-term removal
    current_rank = U.shape[1]
    if current_rank > 2 and time.time() - START_TIME < TIME_BUDGET - 15:
        U2, V2, W2 = _try_rank_reduction_via_gaussian(U, V, W, T)
        if _verify(U2, V2, W2, T):
            U, V, W = U2, V2, W2

    # Convert to list format with clean values
    U = np.round(U, 10)
    V = np.round(V, 10)
    W = np.round(W, 10)

    # Final safety check
    if not _verify(U, V, W, T):
        U = np.round(U_fallback, 10)
        V = np.round(V_fallback, 10)
        W = np.round(W_fallback, 10)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Rank: {rank}")
    T = _tensor_index(2, 3, 4)
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Valid: {_verify(U, V, W, T)}")
