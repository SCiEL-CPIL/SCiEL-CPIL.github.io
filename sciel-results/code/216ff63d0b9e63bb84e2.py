"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _start = time.time()
    _deadline = _start + 130  # safety margin for 150s limit
    n, m, p = DIMENSIONS  # 3, 4, 6

    def elapsed():
        return time.time() - _start

    def time_up():
        return elapsed() > _deadline

    # Build a block-structured decomposition:
    # - Split columns into 3 groups of 2
    # - For each group, use Strassen for the 2x2 sub-blocks in rows 1-2
    # - For row 0, use direct terms
    # This gives rank 66. Then attempt to reduce rank by finding
    # linear dependencies among the rank-1 terms.

    U66 = np.zeros((n * m, 66))
    V66 = np.zeros((m * p, 66))
    W66 = np.zeros((p * n, 66))

    col = [0]  # use list for nonlocal mutation

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        c = col[0]
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        U66[A00, c] += 1; U66[A11, c] += 1
        V66[B00, c] += 1; V66[B11, c] += 1
        W66[C00, c] += 1; W66[C11, c] += 1
        col[0] = c + 1

        # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
        c = col[0]
        U66[A10, c] += 1; U66[A11, c] += 1
        V66[B00, c] += 1
        W66[C10, c] += 1; W66[C11, c] -= 1
        col[0] = c + 1

        # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
        c = col[0]
        U66[A00, c] += 1
        V66[B01, c] += 1; V66[B11, c] -= 1
        W66[C01, c] += 1; W66[C11, c] += 1
        col[0] = c + 1

        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        c = col[0]
        U66[A11, c] += 1
        V66[B10, c] += 1; V66[B00, c] -= 1
        W66[C00, c] += 1; W66[C10, c] += 1
        col[0] = c + 1

        # M5 = (A00+A01)*B11 -> C01 += M5, C00 -= M5
        c = col[0]
        U66[A00, c] += 1; U66[A01, c] += 1
        V66[B11, c] += 1
        W66[C00, c] -= 1; W66[C01, c] += 1
        col[0] = c + 1

        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        c = col[0]
        U66[A10, c] += 1; U66[A00, c] -= 1
        V66[B00, c] += 1; V66[B01, c] += 1
        W66[C11, c] += 1
        col[0] = c + 1

        # M7 = (A01-A11)*(B10+B11) -> C00 += M7
        c = col[0]
        U66[A01, c] += 1; U66[A11, c] -= 1
        V66[B10, c] += 1; V66[B11, c] += 1
        W66[C00, c] += 1
        col[0] = c + 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        c = col[0]
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        # Direct: A0*B00 -> C0, A1*B10 -> C0
        U66[A0, c] += 1; V66[B00, c] += 1; W66[C0, c] += 1
        col[0] = c + 1
        c = col[0]
        U66[A1, c] += 1; V66[B10, c] += 1; W66[C0, c] += 1
        col[0] = c + 1
        # Direct: A0*B01 -> C1, A1*B11 -> C1
        c = col[0]
        U66[A0, c] += 1; V66[B01, c] += 1; W66[C1, c] += 1
        col[0] = c + 1
        c = col[0]
        U66[A1, c] += 1; V66[B11, c] += 1; W66[C1, c] += 1
        col[0] = c + 1

    # Build the decomposition
    # Row 0: direct terms for all 6 column positions (3 groups of 2)
    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    # Rows 1-2: Strassen for all 6 column positions (3 groups of 2)
    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    U = U66.copy()
    V = V66.copy()
    W = W66.copy()

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_tensor_matrix(U_mat, V_mat, W_mat, r):
        """Build the 72 x r tensor matrix efficiently using einsum."""
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        # T[l, i, j, k] = U[i,j,l] * V[j,k,l] * W[k,i,l]
        # We want M[i*m*p + j*p + k, l]
        T = np.einsum('ijl,jkl,kil->ijkl', u, v, w)
        return T.reshape(n * m * p, r)

    # Verify rank-66
    if not verify(U, V, W, 66):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_f = np.zeros((n * m, rank_school))
        V_f = np.zeros((m * p, rank_school))
        W_f = np.zeros((p * n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, c] = 1
                    V_f[j * p + k, c] = 1
                    W_f[k * n + i, c] = 1
                    c += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    best_rank = 66
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    nmp = n * m * p  # 72

    # Strategy: use SVD to find nullspace vectors, then try eliminating columns
    # in order of feasibility. Also try greedy elimination.
    
    def try_reduce(U_cur, V_cur, W_cur, r_cur, max_reductions=10):
        """Try to reduce rank by finding nullspace vectors and eliminating columns."""
        best_r = r_cur
        best_U = U_cur.copy()
        best_V = V_cur.copy()
        best_W = W_cur.copy()
        
        for reduction in range(max_reductions):
            if time_up():
                break
            if best_r <= 1:
                break
                
            M = build_tensor_matrix(best_U, best_V, best_W, best_r)
            
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-10
                rank_M = int(np.sum(s > tol))
                
                if rank_M >= best_r:
                    break  # No more dependencies
                
                null_space = Vt[rank_M:, :].T  # (best_r, best_r - rank_M)
                
                # Try each nullspace vector, eliminating the column with max |coeff|
                found = False
                # Sort nullspace vectors by potential: prefer those where max coeff dominates
                ns_list = []
                for nv in null_space:
                    abs_nv = np.abs(nv)
                    max_val = np.max(abs_nv)
                    if max_val < tol:
                        continue
                    # Score: ratio of max to sum of others (higher = easier to eliminate)
                    others_sum = np.sum(abs_nv) - max_val
                    ratio = max_val / (others_sum + 1e-15)
                    ns_list.append((ratio, nv))
                
                ns_list.sort(key=lambda x: -x[0])  # Try most dominant first
                
                for ratio, nv in ns_list:
                    if time_up():
                        break
                    max_idx = int(np.argmax(np.abs(nv)))
                    if abs(nv[max_idx]) < tol:
                        continue
                    
                    # Correct reduction:
                    # T = sum_{l != k} (1 - c_l/c_k) T_l
                    # So we scale the remaining terms by alpha_l = 1 - c_l/c_k
                    # and delete the k-th term.
                    # We can scale U, V, or W. Scaling U is sufficient.
                    alpha = 1.0 - nv / nv[max_idx]
                    alpha[max_idx] = 0.0 # Just to be safe, though we delete it
                    
                    # Create new matrices
                    U_new = best_U.copy()
                    V_new = best_V.copy()
                    W_new = best_W.copy()
                    
                    # Apply scaling to all columns except the one to be deleted
                    # Note: alpha is a vector of length best_r
                    # We want to scale column l by alpha[l] for l != max_idx
                    # U[:, l] *= alpha[l]
                    U_new[:, :max_idx] *= alpha[:max_idx]
                    U_new[:, max_idx+1:] *= alpha[max_idx+1:]
                    
                    # Delete the max_idx column
                    U_new = np.delete(U_new, max_idx, axis=1)
                    V_new = np.delete(V_new, max_idx, axis=1)
                    W_new = np.delete(W_new, max_idx, axis=1)
                    
                    if verify(U_new, V_new, W_new, best_r - 1):
                        best_r -= 1
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                        found = True
                        break
                
                if not found:
                    break
                    
            except Exception:
                break
        
        return best_r, best_U, best_V, best_W

    # First pass: try to reduce from 66
    best_rank, best_U, best_V, best_W = try_reduce(U, V, W, 66, max_reductions=3)

    # If we got to 65 or below, try more aggressive search
    # Try different elimination orders by permuting columns first
    if best_rank > 64 and not time_up():
        # Try a few random column permutations to find better dependencies
        import random
        rng = random.Random(42)
        
        for trial in range(5):
            if time_up():
                break
            
            # Start from the 66-term decomposition
            U_t = U66.copy()
            V_t = V66.copy()
            W_t = W66.copy()
            
            # Apply a random column permutation
            perm = list(range(66))
            rng.shuffle(perm)
            U_t = U_t[:, perm]
            V_t = V_t[:, perm]
            W_t = W_t[:, perm]
            
            r_t, U_t2, V_t2, W_t2 = try_reduce(U_t, V_t, W_t, 66, max_reductions=2)
            
            if r_t < best_rank:
                best_rank = r_t
                best_U = U_t2
                best_V = V_t2
                best_W = W_t2

    # Final: try to push one more reduction if we have time
    if best_rank > 63 and not time_up():
        r_final, U_f, V_f, W_f = try_reduce(best_U, best_V, best_W, best_rank, max_reductions=2)
        if r_final < best_rank:
            best_rank = r_final
            best_U = U_f
            best_V = V_f
            best_W = W_f

    # Convert to lists and return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
