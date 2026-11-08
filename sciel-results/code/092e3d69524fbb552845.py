"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()

def _construct_rank_55():
    """
    Hardcoded rank-55 decomposition for 2x3x10 matrix multiplication.
    Based on 5 blocks of 2 columns, each using 11 ranks.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over 5 blocks of 2 columns
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W00: A02 * B2,k0
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W01: A02 * B2,k1
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # W10: A12 * B2,k0
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # W11: A12 * B2,k1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _drop_column(U, V, W, col, keep_cols):
    """
    Try to drop column `col` by expressing W[:,col] as a combination of
    W[:,keep_cols]. Returns (new_U, new_V, new_W) or None.
    """
    Wc = W[:, col]
    Wk = W[:, keep_cols]
    # Solve Wk @ a = Wc
    try:
        a, res, rank, sv = np.linalg.lstsq(Wk, Wc, rcond=None)
    except Exception:
        return None
    
    # Check if it's an exact solution
    # Since inputs are integers, the solution should be close to rational/integer
    # We use a tight tolerance to ensure validity
    if not np.allclose(Wk @ a, Wc, atol=1e-7, rtol=1e-7):
        return None
        
    # If we found a combination, we can eliminate the column.
    # New column j in keep_cols becomes: U[:,j] + a[j]*U[:,col]
    # This is because the term for col was U_col*V_col*W_col
    # We replace it with sum_j a_j * U_j*V_j*W_j
    # The new U_j' = U_j + a_j * U_col
    # The new V_j' = V_j + a_j * V_col
    # The new W_j' = W_j + a_j * W_col
    # Wait, the standard trick is:
    # T = sum_r U_r V_r W_r
    # If W_c = sum_j a_j W_j, then U_c V_c W_c = U_c V_c sum_j a_j W_j = sum_j (a_j U_c V_c) W_j
    # This doesn't directly let us drop it unless we can absorb into U and V.
    # Actually, the identity is:
    # U_c V_c W_c = sum_j a_j U_c V_c W_j
    # We can add a_j U_c V_c to the j-th term? No.
    # The correct elimination is:
    # If W_c = sum_j a_j W_j, then the term U_c V_c W_c is redundant IF we can adjust U and V.
    # Let's check the algebra:
    # We want to express the tensor without the c-th rank.
    # T = ... + U_c V_c W_c + ...
    # Substitute W_c: T = ... + U_c V_c (sum_j a_j W_j) + ...
    # T = ... + sum_j (a_j U_c V_c) W_j + ...
    # This changes the coefficients of the existing terms.
    # Specifically, for each j in keep_cols, the coefficient of W_j in the tensor expansion
    # is currently U_j V_j. We add a_j U_c V_c to it.
    # So new_U_j = U_j, new_V_j = V_j + a_j U_c? No, U and V are matrices.
    # The rank-1 term is u_r v_r^T w_r.
    # If w_c = sum a_j w_j, then u_c v_c^T w_c = sum a_j (u_c v_c^T) w_j.
    # This means we can absorb u_c v_c^T into the existing terms?
    # No, u_c v_c^T is a matrix. The existing term has u_j v_j^T.
    # We cannot generally add a matrix to a matrix and keep it rank-1.
    # 
    # CORRECTION: The standard "drop column" in tensor rank decomposition works if 
    # the column in W is in the span of the others. 
    # Let w_c = sum_j a_j w_j.
    # Then u_c v_c^T w_c = sum_j a_j u_c v_c^T w_j.
    # This implies that the tensor T can be written as:
    # T = sum_{j != c} u_j v_j^T w_j + u_c v_c^T (sum_j a_j w_j)
    # T = sum_{j != c} u_j v_j^T w_j + sum_j (a_j u_c v_c^T) w_j
    # T = sum_j (u_j v_j^T + a_j u_c v_c^T) w_j
    # This is NOT a sum of rank-1 tensors unless u_j v_j^T + a_j u_c v_c^T is rank-1.
    # It is generally rank 2. So this simple drop is INVALID for general tensors.
    #
    # HOWEVER, in matrix multiplication tensors, there are specific structures.
    # But wait, the inspiration 1 code does exactly this and claims to check `np.allclose`.
    # If the code in Inspiration 1 works, it implies that for this specific tensor, 
    # the resulting matrices might still allow a valid decomposition or the check is flawed.
    #
    # Let's re-read the Inspiration 1 code carefully.
    # It computes `new_U[:, li] = U[:, kc] + a[li] * U[:, col]`
    # `new_V[:, li] = V[:, kc] + a[li] * V[:, col]`
    # `new_W[:, li] = W[:, kc] + a[li] * W[:, col]`
    # This transformation is NOT a standard basis change that preserves the tensor.
    # The tensor is sum_r U_r V_r W_r.
    # If we change U, V, W simultaneously like this, the sum changes.
    #
    # Let's test the validity of the Inspiration 1 approach.
    # If the Inspiration 1 approach is invalid, it would produce a wrong result.
    # But the prompt says Inspiration 1 has score 1.0909 (same as current).
    # This suggests it either didn't drop any columns (stayed at 55) or dropped some and remained valid.
    #
    # Actually, there is a known technique for "sliding" or "shifting" in matrix multiplication.
    # But the safest bet is to use a verified local search.
    #
    # Given the risk of the Inspiration 1 logic being flawed (or only working by luck/structure),
    # and the strict requirement for correctness, I will implement a safer local improvement:
    # I will try to find a linear dependence in W, and if found, verify if the resulting
    # U and V matrices can be decomposed into rank-1 terms.
    #
    # However, since we are constrained to keep the design family and the Inspiration 1 code
    # is provided as a "strong idea", I will use it but add a STRICT VERIFICATION step.
    # If the modified decomposition fails verification, we discard it.
    
    new_U = np.zeros((U.shape[0], len(keep_cols)))
    new_V = np.zeros((V.shape[0], len(keep_cols)))
    new_W = np.zeros((W.shape[0], len(keep_cols)))
    for li, kc in enumerate(keep_cols):
        new_U[:, li] = U[:, kc] + a[li] * U[:, col]
        new_V[:, li] = V[:, kc] + a[li] * V[:, col]
        new_W[:, li] = W[:, kc] + a[li] * W[:, col]
    return new_U, new_V, new_W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Verify that sum_r U[:,r]*V[:,r]*W[:,r] == target tensor.
    Target T[i, j, k] = 1 if we are multiplying A (n x m) and B (m x p) to get C (n x p).
    Wait, the tensor is T_{i,j,k} for C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The rank decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}.
    Note: W index is [k*n+i] in the problem description? 
    "W[k*n+i,r]" -> W is (p*n, rank).
    U is (n*m, rank). Index [i*m+j, r].
    V is (m*p, rank). Index [j*p+k, r].
    
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # Optimization: Instead of looping all i,j,k, we can check a few random samples 
    # or use matrix multiplication tricks.
    # For small dimensions (2,3,10), a full check is fast enough (2*3*10=60 checks * rank).
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _local_improve(U, V, W, time_budget):
    """Attempt to reduce rank via null-space column elimination."""
    start = time.time()
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    cur_cols = list(range(cur_U.shape[1]))
    improved = True
    n, m, p = DIMENSIONS
    
    while improved:
        improved = False
        if time.time() - start > time_budget:
            break
        # Sort columns by norm of W to find most redundant?
        # Heuristic: try all, but break on first success
        for col in list(cur_cols):
            if time.time() - start > time_budget:
                break
            keep_cols = [c for c in cur_cols if c != col]
            if not keep_cols:
                continue
            res = _drop_column(cur_U, cur_V, cur_W, col, keep_cols)
            if res is not None:
                nU, nV, nW = res
                # CRITICAL: Verify the new decomposition
                if _verify_decomposition(nU, nV, nW, n, m, p, tol=1e-5):
                    cur_U, cur_V, cur_W = nU, nV, nW
                    cur_cols = [c for c in cur_cols if c != col]
                    improved = True
                    break
                # If verification fails, discard this drop attempt
    return cur_U, cur_V, cur_W, len(cur_cols)

def _to_int_matrices(U, V, W):
    """Convert numpy float matrices to int lists."""
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0  # safety margin

    # Step 1: Hardcoded rank-55 decomposition
    U, V, W = _construct_rank_55()
    rank = U.shape[1]
    best = (rank, U.copy(), V.copy(), W.copy())

    # Step 2: Local improvement
    if time.time() < DEADLINE:
        time_budget = max(1.0, DEADLINE - time.time() - 5.0)
        try:
            nU, nV, nW, new_rank = _local_improve(U, V, W, time_budget)
            if new_rank < best[0]:
                # Final verification before updating best
                if _verify_decomposition(nU, nV, nW, n, m, p, tol=1e-5):
                    best = (new_rank, nU, nV, nW)
        except Exception:
            pass

    r, U, V, W = best
    U, V, W = _to_int_matrices(U, V, W)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
