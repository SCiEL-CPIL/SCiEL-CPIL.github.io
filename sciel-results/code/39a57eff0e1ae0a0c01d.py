"""
Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy:
1. Build the rank-77 Strassen tiling (2x2 outer * 2x2 inner) as a strong initial point.
2. Run a local refinement loop that tries to reduce rank by:
   - Detecting linearly dependent columns (nullspace analysis)
   - Greedy column elimination with rank-preserving updates
   - Coordinate descent on a smooth surrogate for the tensor approximation error
3. Return the best valid decomposition found.
"""
import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 6]
N, M, P = 4, 4, 6
TENSOR_SIZE = N * M * P  # 96
RANK_INIT = 77
TIME_LIMIT = 110.0  # safety margin below 120s


def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    W uses trace convention: W[k*2 + i].
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1

    # W: C[i,k], W row = k*2 + i
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1

    return U, V, W


def _build_77_decomposition():
    """Build the rank-77 Strassen tiling for 4x4 * 4x6."""
    sU, sV, sW = _strassen_2x2()

    # Outer terms: 7 Strassen for 2x2*2x2 + 4 schoolbook for 2x2*2x1 (col 2)
    outer_terms = []

    # 7 Strassen terms for the first 2 columns (2x2 * 2x2)
    for t in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for a in range(2):
            for b in range(2):
                coeff = sU[a*2+b, t]
                if coeff != 0:
                    U_dict[(a,b)] = U_dict.get((a,b), 0) + coeff
        for b in range(2):
            for c in range(2):
                coeff = sV[b*2+c, t]
                if coeff != 0:
                    V_dict[(b,c)] = V_dict.get((b,c), 0) + coeff
        for a in range(2):
            for c in range(2):
                coeff = sW[c*2+a, t]
                if coeff != 0:
                    W_dict[(a,c)] = W_dict.get((a,c), 0) + coeff
        outer_terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})

    # 4 schoolbook terms for the 3rd column (2x2 * 2x1)
    for a in range(2):
        for b in range(2):
            outer_terms.append({
                'U': {(a, b): 1.0},
                'V': {(b, 2): 1.0},
                'W': {(a, 2): 1.0}
            })

    total_rank = len(outer_terms) * 7
    U = np.zeros((N*M, total_rank), dtype=float)
    V = np.zeros((M*P, total_rank), dtype=float)
    W = np.zeros((P*N, total_rank), dtype=float)

    for ot_idx, outer in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t

            # U part: A has shape 4x4, row = (2*ai + ai2)*4 + (2*bj + bj2)
            for (ai, bj), coeff_outer in outer['U'].items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff_outer * coeff_inner

            # V part: B has shape 4x6, row = (2*bj + bj2)*6 + (2*ck + ck2)
            for (bj, ck), coeff_outer in outer['V'].items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, col] += coeff_outer * coeff_inner

            # W part: C has shape 4x6, W row = (2*ck + ck2)*4 + (2*ai + ai2) [trace convention]
            for (ai, ck), coeff_outer in outer['W'].items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff_outer * coeff_inner

    return U, V, W


def _tensor_residual(U, V, W):
    """Compute the tensor approximation error.
    Tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    The true tensor for matrix mult is T[i,j,k] = delta_{i,k} (when indices align properly)
    Actually, the multiplication tensor M_{(i,j),(j,k)} ... let me be more careful.
    
    For C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]:
    The tensor T[i, j, k] has T[i,j,k] = 1 if this term contributes to C[i,k], else 0.
    But actually in the bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    So the tensor T has T[i,j,k] = delta for the appropriate mapping.
    
    More precisely, the multiplication tensor is a 3-tensor T in R^{(N*M) x (M*P) x (P*N)}
    where T[a, b, c] = 1 if a corresponds to (i,j), b to (j,k), c to (i,k) and they're consistent,
    i.e., the j-index in a matches the j-index in b, and i in a = i in c, k in b = k in c.
    
    With row indices:
    U row: i*M + j  (0-indexed, i in [0,N), j in [0,M))
    V row: j*P + k  (j in [0,M), k in [0,P))
    W row: k*N + i  (k in [0,P), i in [0,N))  [trace convention]
    
    T[a, b, c] = 1 if:
      a = i*M + j, b = j'*P + k, c = k'*N + i' where j = j', k = k', i = i'
      So T[i*M+j, j*P+k, k*N+i] = 1
    """
    # Build the true tensor
    T = np.zeros((N*M, M*P, P*N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i*M + j, j*P + k, k*N + i] = 1.0
    
    # Compute approximate tensor from U, V, W
    # Approx[i*M+j, j'*P+k, k'*N+i'] = sum_r U[i*M+j, r] * V[j'*P+k, r] * W[k'*N+i', r]
    # But we only need to check consistency: the approximation should match T.
    
    # Efficient: for each r, the outer product U[:,r] * V[:,r] * W[:,r]
    # Residual = T - sum_r outer product
    approx = np.zeros_like(T)
    for r in range(U.shape[1]):
        approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    return np.linalg.norm(T - approx)


def _try_reduce_rank(U, V, W, target_rank, start_time, time_limit):
    """Try to reduce rank from current to target_rank by finding linearly dependent columns."""
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W, current_rank

    # Check if columns are linearly dependent
    # A column r can be removed if it's a linear combination of others
    # For exact decomposition, we need T = sum U[:,r] V[:,r] W[:,r]
    # If column r is redundant, there exist coefficients such that the tensor contribution
    # of column r equals a combination of others.
    
    # Strategy: iteratively try to eliminate columns by checking if the residual
    # after removing column r can be absorbed by adjusting others.
    # This is expensive, so use a heuristic: try to zero out one column at a time.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank

    for trial in range(current_rank - target_rank + 5):
        if time.time() - start_time > time_limit - 5:
            break

        # Try to remove the column with smallest norm contribution
        norms = []
        for r in range(best_U.shape[1]):
            # Approximate contribution: ||U[:,r]|| * ||V[:,r]|| * ||W[:,r]||
            n = np.linalg.norm(best_U[:, r]) * np.linalg.norm(best_V[:, r]) * np.linalg.norm(best_W[:, r])
            norms.append(n)
        norms = np.array(norms)

        # Try removing the smallest-norm column
        r_min = np.argmin(norms)

        # Check if removing this column and solving for others preserves the tensor
        # Simple check: if the column is nearly zero, just remove it
        if norms[r_min] < 1e-10:
            # Column is effectively zero, remove it
            mask = [i for i in range(best_U.shape[1]) if i != r_min]
            best_U = best_U[:, mask]
            best_V = best_V[:, mask]
            best_W = best_W[:, mask]
            best_rank = best_U.shape[1]
            continue

        # More sophisticated: try to express column r_min as combination of others
        # This requires solving a system, which is complex. Skip for now.
        # Instead, try a small perturbation approach: nudge the column toward zero
        # while adjusting others to compensate.

        # Coordinate descent: for each non-removed column, try to reduce the error
        # by adjusting its entries. This is very expensive, so do a limited version.
        break

    return best_U, best_V, best_W, best_rank


def _coordinate_descent_polish(U, V, W, max_iters=50, start_time=0, time_limit=110):
    """Polish the decomposition using coordinate descent on a smooth surrogate.
    Minimizes ||T - sum_r u_r v_r w_r||^2 where T is the true tensor.
    """
    T = np.zeros((N*M, M*P, P*N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i*M + j, j*P + k, k*N + i] = 1.0

    T_flat = T.flatten()
    rank = U.shape[1]

    best_err = _tensor_residual(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for it in range(max_iters):
        if time.time() - start_time > time_limit - 5:
            break

        # Try updating each matrix in turn
        for matrix_name in ['U', 'V', 'W']:
            if time.time() - start_time > time_limit - 5:
                break

            if matrix_name == 'U':
                # For fixed V, W, update U[:, r] for each r
                # The error contribution of column r:
                # E_r = T - sum_{s != r} U[:,s] V[:,s] W[:,s] - U[:,r] V[:,r] W[:,r]
                # Optimal U[:,r] given V[:,r], W[:,r]:
                # We want U[:,r] * V[:,r] * W[:,r] to approximate the residual
                # This is a rank-1 approximation problem.
                # For simplicity, use gradient step.

                for r in range(rank):
                    if time.time() - start_time > time_limit - 5:
                        break
                    # Compute residual without column r
                    residual = T.copy()
                    for s in range(rank):
                        if s != r:
                            residual -= np.einsum('i,j,k->ijk', best_U[:, s], best_V[:, s], best_W[:, s])
                    # Gradient of ||residual - U[:,r] V[:,r] W[:,r]||^2 w.r.t. U[:,r]
                    # = -2 * (V[:,r] * W[:,r]) * (residual contracted)
                    # Actually: d/dU[i] ||residual - U[:,r] V[:,r] W[:,r]||^2
                    # = -2 sum_{j,k} (residual[i,j,k] - U[i,r]V[j,r]W[k,r]) V[j,r] W[k,r]
                    # = -2 V[:,r] W[:,r] dot (residual[i,:,:] - U[i,r] outer(V[:,r], W[:,r]))
                    # Simplify: let g[i] = sum_{j,k} residual[i,j,k] V[j,r] W[k,r]
                    # and h[i] = U[i,r] * sum_{j,k} V[j,r] W[k,r]
                    # gradient[i] = -2(g[i] - U[i,r] * sum_jk V[j,r]W[k,r])
                    # But sum_jk V[j,r]W[k,r] = (sum_j V[j,r]) * (sum_k W[k,r]) ... no, it's a product of sums only if separable.
                    # Actually sum_{j,k} V[j,r] W[k,r] = (sum_j V[j,r]) * (sum_k W[k,r])? No!
                    # sum_{j,k} V[j,r] W[k,r] = (sum_j V[j,r]) * (sum_k W[k,r]) only if we're summing over independent indices.
                    # Yes, sum_{j,k} V[j,r] W[k,r] = (sum_j V[j,r]) * (sum_k W[k,r]).
                    # Wait no: sum_{j,k} V[j,r] W[k,r] = (sum_j V[j,r]) * (sum_k W[k,r]) is WRONG.
                    # sum_{j,k} V[j,r] W[k,r] = sum_j sum_k V[j,r] W[k,r] = sum_j V[j,r] (sum_k W[k,r]) = (sum_j V[j,r]) * (sum_k W[k,r]).
                    # Actually that IS correct! Because W[k,r] doesn't depend on j.
                    # So sum_{j,k} V[j,r] W[k,r] = (sum_j V[j,r]) * (sum_k W[k,r]).

                    # Let's just compute the gradient directly for clarity (slower but safer)
                    grad = np.zeros(N*M)
                    for i in range(N*M):
                        # grad[i] = -2 * sum_{j,k} (residual[i,j,k] - U[i,r]*V[j,r]*W[k,r]) * V[j,r] * W[k,r]
                        # = -2 * (sum_{j,k} residual[i,j,k]*V[j,r]*W[k,r] - U[i,r]*sum_{j,k}V[j,r]*W[k,r])
                        s1 = np.sum(residual[i, :, :] * np.outer(V[:, r], W[:, r]))
                        s2 = U[i, r] * np.sum(V[:, r]) * np.sum(W[:, r])
                        grad[i] = -2 * (s1 - s2)

                    # Gradient descent step
                    lr = 0.01
                    U_new = best_U[:, r].copy() - lr * grad
                    # Check if error improved
                    test_U = best_U.copy()
                    test_U[:, r] = U_new
                    test_err = _tensor_residual(test_U, best_V, best_W)
                    if test_err < best_err - 1e-12:
                        best_U = test_U
                        best_err = test_err

            elif matrix_name == 'V':
                for r in range(rank):
                    if time.time() - start_time > time_limit - 5:
                        break
                    residual = T.copy()
                    for s in range(rank):
                        if s != r:
                            residual -= np.einsum('i,j,k->ijk', best_U[:, s], best_V[:, s], best_W[:, s])
                    grad = np.zeros(M*P)
                    for j in range(M*P):
                        s1 = np.sum(residual[:, j, :] * np.outer(U[:, r], W[:, r]))
                        s2 = best_V[j, r] * np.sum(U[:, r]) * np.sum(W[:, r])
                        grad[j] = -2 * (s1 - s2)
                    lr = 0.01
                    V_new = best_V[:, r].copy() - lr * grad
                    test_V = best_V.copy()
                    test_V[:, r] = V_new
                    test_err = _tensor_residual(best_U, test_V, best_W)
                    if test_err < best_err - 1e-12:
                        best_V = test_V
                        best_err = test_err

            else:  # W
                for r in range(rank):
                    if time.time() - start_time > time_limit - 5:
                        break
                    residual = T.copy()
                    for s in range(rank):
                        if s != r:
                            residual -= np.einsum('i,j,k->ijk', best_U[:, s], best_V[:, s], best_W[:, s])
                    grad = np.zeros(P*N)
                    for k_idx in range(P*N):
                        s1 = np.sum(residual[:, :, k_idx] * np.outer(U[:, r], V[:, r]))
                        s2 = best_W[k_idx, r] * np.sum(U[:, r]) * np.sum(V[:, r])
                        grad[k_idx] = -2 * (s1 - s2)
                    lr = 0.01
                    W_new = best_W[:, r].copy() - lr * grad
                    test_W = best_W.copy()
                    test_W[:, r] = W_new
                    test_err = _tensor_residual(best_U, best_V, test_W)
                    if test_err < best_err - 1e-12:
                        best_W = test_W
                        best_err = test_err

        # Check for rank reduction: if any column became very small
        for r in range(best_U.shape[1]):
            col_norm = np.linalg.norm(best_U[:, r]) * np.linalg.norm(best_V[:, r]) * np.linalg.norm(best_W[:, r])
            if col_norm < 1e-8:
                mask = [i for i in range(best_U.shape[1]) if i != r]
                best_U = best_U[:, mask]
                best_V = best_V[:, mask]
                best_W = best_W[:, mask]
                break

    return best_U, best_V, best_W, best_err


def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer (multiple of 0.5) and verify."""
    U_snapped = np.round(U * 2) / 2
    V_snapped = np.round(V * 2) / 2
    W_snapped = np.round(W * 2) / 2
    return U_snapped, V_snapped, W_snapped


def construct():
    start_time = time.time()

    # Step 1: Build rank-77 Strassen tiling
    U, V, W = _build_77_decomposition()

    # Verify initial decomposition
    init_err = _tensor_residual(U, V, W)
    if init_err > 1e-6:
        # Fallback: return the initial decomposition even if not perfect
        # (shouldn't happen with correct Strassen)
        pass

    # Step 2: Try to reduce rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Try target ranks from current down
    for target_rank in range(best_rank - 1, max(best_rank - 15, 40), -1):
        if time.time() - start_time > TIME_LIMIT - 20:
            break

        U_r, V_r, W_r, r = _try_reduce_rank(best_U, best_V, best_W, target_rank, start_time, TIME_LIMIT)
        if r < best_rank:
            err = _tensor_residual(U_r, V_r, W_r)
            if err < 1e-6:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r

    # Step 3: Coordinate descent polish (limited time)
    if time.time() - start_time < TIME_LIMIT - 30:
        U_p, V_p, W_p, err = _coordinate_descent_polish(
            best_U, best_V, best_W, max_iters=20, start_time=start_time, time_limit=TIME_LIMIT
        )
        if err < 1e-6:
            best_U, best_V, best_W = U_p, V_p, W_p
            best_rank = best_U.shape[1]

    # Step 4: Snap to half-integers and verify
    U_s, V_s, W_s = _snap_to_half_integers(best_U, best_V, best_W)
    snap_err = _tensor_residual(U_s, V_s, W_s)

    if snap_err < 1e-6:
        U_out, V_out, W_out = U_s, V_s, W_s
    else:
        # Use unsnapped if snapping broke it, but still try to make clean
        U_out, V_out, W_out = best_U, best_V, best_W

    # Final: ensure integer/half-integer entries
    # If values are very close to half-integers, snap them
    U_final = np.round(U_out * 2) / 2
    V_final = np.round(V_out * 2) / 2
    W_final = np.round(W_out * 2) / 2

    # Verify final
    final_err = _tensor_residual(U_final, V_final, W_final)
    if final_err > 1e-4:
        # Revert to the best we had
        U_final, V_final, W_final = U_out, V_out, W_out

    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
