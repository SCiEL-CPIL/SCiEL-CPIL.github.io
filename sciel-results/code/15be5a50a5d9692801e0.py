import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # ---------- Schoolbook fallback ----------
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # ---------- Strassen 2x2x2 (7 scalar products) ----------
    # Each entry: (U_dict, V_dict, W_dict) over 2x2 blocks, with integer coefficients.
    strassen_2x2 = [
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # ---------- 4x4x4 via top-level Strassen (2x2 blocks of 2x2) ----------
    # Top-level: A = [[A00,A01],[A10,A11]] with each Aij a 2x2 block.
    # Each top-level product is a 2x2 * 2x2 block product, done with Strassen 2x2x2.
    top_4x4 = [
        # M1: (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Build 49 terms for 4x4x4
    # For each top-level term t and each inner Strassen term s:
    #   U[(i,j)] += topA[(u,v)] * innerU[(r,c)]   where i=2u+r, j=2v+c
    #   V[(j,k)] += topB[(v,w)] * innerV[(r,c)]   where j=2v+r, k=2w+c
    #   W[(i,k)] += topC[(u,w)] * innerW[(r,c)]   where i=2u+r, k=2w+c
    terms_4x4 = []
    for t_idx in range(7):
        a_blocks, b_blocks, c_blocks = top_4x4[t_idx]
        for s_idx in range(7):
            s_u, s_v, s_w = strassen_2x2[s_idx]

            u_global = {}
            for (u, v), coeff_a in a_blocks.items():
                for (r, c), coeff_s in s_u.items():
                    i = 2 * u + r
                    j = 2 * v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a

            v_global = {}
            for (v, w), coeff_b in b_blocks.items():
                for (r, c), coeff_s in s_v.items():
                    j = 2 * v + r
                    k = 2 * w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b

            w_global = {}
            for (u, w), coeff_c in c_blocks.items():
                for (r, c), coeff_s in s_w.items():
                    i = 2 * u + r
                    k = 2 * w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c

            u_global = {k: v for k, v in u_global.items() if v != 0}
            v_global = {k: v for k, v in v_global.items() if v != 0}
            w_global = {k: v for k, v in w_global.items() if v != 0}

            if u_global and v_global and w_global:
                terms_4x4.append((u_global, v_global, w_global))

    # ---------- Extend to 4x4x8: two copies, second shifted in k by 4 ----------
    terms_full = []
    for u_dict, v_dict, w_dict in terms_4x4:
        terms_full.append((u_dict, v_dict, w_dict))
    for u_dict, v_dict, w_dict in terms_4x4:
        u_new = dict(u_dict)
        v_new = {(j, k + 4): coeff for (j, k), coeff in v_dict.items()}
        w_new = {(i, k + 4): coeff for (i, k), coeff in w_dict.items()}
        terms_full.append((u_new, v_new, w_new))

    # ---------- Build dense factor matrices ----------
    rank = len(terms_full)
    U_arr = np.zeros((n * m, rank), dtype=np.float64)
    V_arr = np.zeros((m * p, rank), dtype=np.float64)
    W_arr = np.zeros((p * n, rank), dtype=np.float64)

    for col_idx, (u_dict, v_dict, w_dict) in enumerate(terms_full):
        for (i, j), coeff in u_dict.items():
            U_arr[i * m + j, col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V_arr[j * p + k, col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W_arr[k * n + i, col_idx] = coeff

    # ---------- Fast verification: C[i,k] = sum_j U[i,j]*V[j,k]*W[k,i] ----------
    # The tensor entry for (i,j,k) is sum_r U[i,j,r]*V[j,k,r]*W[k,i,r].
    # Group by (i,k): for fixed (i,k), sum over j of (U[:,j,r]*V[j,k,r]*W[k,i,r]).
    # Vectorized: for each (i,k), C_val = sum_j (U[i*m:(i+1)*m, :] * V[... , :] * W[k*n+i, :])
    # More precisely: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_j ( U[i*m+j, :] * W[k*n+i, :] * V[j*p+k, :] ) dot... no.
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    # Let S[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]  (for all r)
    # Then C[i,k] = dot(S[i,k,:], W[k*n+i, :])

    valid = True
    if time.time() < deadline - 15:
        # Build S: shape (n, p, rank)
        # S[i, k, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
        # U is (n*m, rank), V is (m*p, rank)
        # Reshape U to (n, m, rank), V to (m, p, rank)
        U_reshaped = U_arr.reshape(n, m, rank)  # U_reshaped[i, j, r] = U[i*m+j, r]
        V_reshaped = V_arr.reshape(m, p, rank)  # V_reshaped[j, k, r] = V[j*p+k, r]
        # S[i, k, r] = sum_j U_reshaped[i, j, r] * V_reshaped[j, k, r]
        # = einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
        S = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped, optimize=True)
        # C[i, k] = sum_r S[i, k, r] * W[k*n+i, r]
        # W is (p*n, rank), W[k*n+i, r]
        # C[i, k] = einsum('ikr,kr->ik', S, W_reshaped) where W_reshaped[k, i, r] = W[k*n+i, r]
        W_reshaped = W_arr.reshape(p, n, rank)  # W_reshaped[k, i, r] = W[k*n+i, r]
        C = np.einsum('ikr,kr->ik', S, W_reshaped, optimize=True)
        # C should be all ones
        if not np.allclose(C, 1.0, atol=1e-9):
            valid = False

    if not valid:
        return {"U": U_school, "V": V_school, "W": W_school}

    # ---------- Rank reduction: remove linearly dependent columns ----------
    # A column r is redundant if its (U[:,r], V[:,r], W[:,r]) vector is in the span
    # of earlier kept columns. We check via rank of the stacked matrix.
    if time.time() < deadline - 10:
        # Stack all columns into a single matrix of shape (n*m + m*p + p*n, rank)
        stacked = np.vstack([U_arr, V_arr, W_arr])  # (n*m+m*p+p*n, rank)
        # Greedy: keep column r if it increases the rank of the kept set
        kept = []
        # Use incremental QR or SVD for small rank
        # For rank ~98 and dim ~112, we can do it column by column with SVD
        kept_cols = []
        current_matrix = None
        for r in range(rank):
            col = stacked[:, r].reshape(-1, 1)
            if current_matrix is None:
                current_matrix = col
                kept_cols.append(r)
            else:
                new_matrix = np.hstack([current_matrix, col])
                # Check if rank increased
                old_rank = np.linalg.matrix_rank(current_matrix, tol=1e-9)
                new_rank = np.linalg.matrix_rank(new_matrix, tol=1e-9)
                if new_rank > old_rank:
                    current_matrix = new_matrix
                    kept_cols.append(r)
        # Rebuild with only kept columns
        if len(kept_cols) < rank:
            U_arr = U_arr[:, kept_cols]
            V_arr = V_arr[:, kept_cols]
            W_arr = W_arr[:, kept_cols]

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    print(f"Rank: {U.shape[1]}")
