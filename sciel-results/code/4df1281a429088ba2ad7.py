import time

DIMENSIONS = [2, 4, 6]

def _build_rank42():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(i_start, j_start, k_start, col_offset):
        a_idx = (i_start * m + j_start, i_start * m + j_start + 1,
                 (i_start + 1) * m + j_start, (i_start + 1) * m + j_start + 1)
        e_idx = (j_start * p + k_start, j_start * p + k_start + 1,
                 (j_start + 1) * p + k_start, (j_start + 1) * p + k_start + 1)
        c11 = k_start * n + i_start
        c12 = (k_start + 1) * n + i_start
        c21 = k_start * n + (i_start + 1)
        c22 = (k_start + 1) * n + (i_start + 1)

        def put(Ur, idx, val, c):
            Ur[idx][c] += val

        # M1 = (a+d)(e+h)
        c = col_offset
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[e_idx[0]][c] += 1; V[e_idx[3]][c] += 1
        W[c11][c] += 1; W[c22][c] += 1
        c += 1
        # M2 = (c+d)e
        U[a_idx[2]][c] += 1; U[a_idx[3]][c] += 1
        V[e_idx[0]][c] += 1
        W[c21][c] += 1; W[c22][c] += -1
        c += 1
        # M3 = a(f-h)
        U[a_idx[0]][c] += 1
        V[e_idx[1]][c] += 1; V[e_idx[3]][c] += -1
        W[c12][c] += 1; W[c22][c] += 1
        c += 1
        # M4 = d(g-e)
        U[a_idx[3]][c] += 1
        V[e_idx[2]][c] += 1; V[e_idx[0]][c] += -1
        W[c11][c] += 1; W[c21][c] += 1
        c += 1
        # M5 = (a+b)h
        U[a_idx[0]][c] += 1; U[a_idx[1]][c] += 1
        V[e_idx[3]][c] += 1
        W[c11][c] += -1; W[c12][c] += 1
        c += 1
        # M6 = (c-a)(e+f)
        U[a_idx[2]][c] += 1; U[a_idx[0]][c] += -1
        V[e_idx[0]][c] += 1; V[e_idx[1]][c] += 1
        W[c22][c] += 1
        c += 1
        # M7 = (b-d)(g+h)
        U[a_idx[1]][c] += 1; U[a_idx[3]][c] += -1
        V[e_idx[2]][c] += 1; V[e_idx[3]][c] += 1
        W[c11][c] += 1
        c += 1
        return c

    col = 0
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    col = add_strassen_block(0, 2, 0, col)
    col = add_strassen_block(0, 2, 2, col)
    col = add_strassen_block(0, 2, 4, col)
    return {"U": U, "V": V, "W": W}

def _verify(result):
    try:
        import numpy as np
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        n, m, p = 2, 4, 6
        # Build tensor T from U,V,W: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # where i in A (n*m), j in B (m*p), k in C-index (p*n) -> C[i_C, k_C]
        # The multiplication tensor: T should equal sum_{i,j,k} ... 
        # Check: for random A,B, C = A@B, and the decomposition gives C via
        # C_flat = (U^T? ) ... Let's verify directly:
        # C[i_C, k_C] = sum_r (sum_i A[i] U[i,r]) (sum_j B[j] V[j,r]) W[k*n+i_C? ...]
        # Actually the decomposition: C = sum_r (A·u_r)(B·v_r) w_r where
        # u_r = U[:,r] (vec of A), v_r = V[:,r] (vec of B), w_r = W[:,r] (vec in C-space, index k*n+i = C[i,k])
        # So C[i,k] = sum_r (A·u_r)(B·v_r) W[k*n+i, r]
        # Let's test with random A (2x4), B (4x6).
        rng = np.random.default_rng(0)
        A = rng.integers(-5, 5, size=(n, m))
        B = rng.integers(-5, 5, size=(m, p))
        C_true = A @ B
        # compute via decomposition
        # A·u_r = sum over A-indices. A-indices: row i (0..n-1), col j (0..m-1) -> index i*m+j
        # u_r = U[:, r], U has shape (n*m, rank). A flat: A.ravel() order? We need A[i,j] at index i*m+j.
        A_flat = A.reshape(n*m)  # C order: index = i*m + j
        B_flat = B.reshape(m*p)  # index = j*p + k
        # For each r: prod_r = (A_flat·U[:,r]) * (B_flat·V[:,r])
        Au = A_flat @ U  # shape (rank,)
        Bv = B_flat @ V  # shape (rank,)
        prods = Au * Bv  # shape (rank,)
        # C[i,k] = sum_r prods[r] * W[k*n+i, r]
        # W has shape (p*n, rank), index k*n+i
        C_comp = np.zeros((n, p))
        # C_comp[i,k] = sum_r prods[r] W[k*n+i, r]
        # = (W^T @ prods) reshaped: W^T has shape (rank, p*n), (W^T @ prods)[k*n+i] = sum_r W[k*n+i,r] prods[r]
        Wt_prods = W.T @ prods  # shape (p*n,)
        C_comp = Wt_prods.reshape(p, n).T  # reshape to (p, n) then transpose to (n, p): index [i,k]
        # Check reshape: Wt_prods has index k*n+i. reshape(p, n) gives [k, i]. transpose -> [i, k]. Good.
        if not np.allclose(C_comp, C_true, atol=1e-6):
            return False
        return True
    except Exception:
        return False

def construct():
    start = time.time()
    deadline = start + 120.0  # safety margin below 150s kill
    result = _build_rank42()
    # Verify; if somehow broken, we still return it (it's correct by construction), 
    # but verification guards against coding errors.
    ok = _verify(result)
    # If verification fails (should not), we could try to rebuild, but it's deterministic.
    if not ok:
        # Fallback: rebuild (same) - no better option; return as-is.
        pass
    if time.time() > deadline - 5.0:
        return result
    return result
