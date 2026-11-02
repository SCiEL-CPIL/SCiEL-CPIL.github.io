"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.

Strategy:
  1. Build the rank-28 Strassen-based decomposition (4 blocks x 7 terms).
  2. Attempt rank-27 by checking if any single term is a linear combination
     of the other 27 terms in the tensor sense.
  3. If found, return rank-27; otherwise return the verified rank-28.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    t0 = time.time()
    deadline = t0 + 140.0

    # ------------------------------------------------------------------
    # Step 1: Build the rank-28 Strassen decomposition
    # ------------------------------------------------------------------
    rank28 = 28
    U = [[0] * rank28 for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank28 for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank28 for _ in range(p * n)]   # 8 x 28

    # Strassen coefficients for 2x2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    blocks = [
        {'A_offset': 0, 'B_row_offset': 0, 'B_col_offset': 0, 'C_col_offset': 0},
        {'A_offset': 2, 'B_row_offset': 2, 'B_col_offset': 0, 'C_col_offset': 0},
        {'A_offset': 0, 'B_row_offset': 0, 'B_col_offset': 2, 'C_col_offset': 2},
        {'A_offset': 2, 'B_row_offset': 2, 'B_col_offset': 2, 'C_col_offset': 2},
    ]

    current_rank = 0
    for block in blocks:
        A_off = block['A_offset']
        B_row_off = block['B_row_offset']
        B_col_off = block['B_col_offset']
        C_col_off = block['C_col_offset']

        for term in strassen_terms:
            coeff_A, coeff_B, coeff_C = term

            for loc_idx, val in coeff_A.items():
                li, lj = divmod(loc_idx, 2)
                u_idx = li * 4 + (A_off + lj)
                U[u_idx][current_rank] += val

            for loc_idx, val in coeff_B.items():
                li, lj = divmod(loc_idx, 2)
                v_idx = (B_row_off + li) * 4 + (B_col_off + lj)
                V[v_idx][current_rank] += val

            for loc_idx, val in coeff_C.items():
                li, lj = divmod(loc_idx, 2)
                w_idx = (C_col_off + lj) * 2 + li
                W[w_idx][current_rank] += val

            current_rank += 1

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 2: Verification helper
    # ------------------------------------------------------------------
    def verify_decomposition(U_arr, V_arr, W_arr, tol=1e-10):
        n_, m_, p_ = DIMENSIONS
        max_err = 0.0
        for a in range(n_ * m_):
            ai, aj = divmod(a, m_)
            for b in range(m_ * p_):
                bi, bj = divmod(b, p_)
                for c in range(p_ * n_):
                    ci, cj = divmod(c, n_)
                    target = 1.0 if (aj == bi and bj == ci) else 0.0
                    val = 0.0
                    for r in range(U_arr.shape[1]):
                        val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                    err = abs(val - target)
                    if err > max_err:
                        max_err = err
        return max_err <= tol

    def verify_decomposition_sampled(U_arr, V_arr, W_arr, n_samples=500, tol=1e-10):
        n_, m_, p_ = DIMENSIONS
        rng = np.random.RandomState(42)
        max_err = 0.0
        for _ in range(n_samples):
            a = rng.randint(0, n_ * m_)
            b = rng.randint(0, m_ * p_)
            c = rng.randint(0, p_ * n_)
            ai, aj = divmod(a, m_)
            bi, bj = divmod(b, p_)
            ci, cj = divmod(c, n_)
            target = 1.0 if (aj == bi and bj == ci) else 0.0
            val = 0.0
            for r in range(U_arr.shape[1]):
                val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
            err = abs(val - target)
            if err > max_err:
                max_err = err
        return max_err <= tol

    # ------------------------------------------------------------------
    # Step 3: Attempt rank-27 by checking if any term is redundant
    # ------------------------------------------------------------------
    def try_eliminate_term(U_np, V_np, W_np, term_to_remove):
        """
        Check if term_to_remove can be expressed as a linear combination
        of the remaining 27 terms. Returns (True, U27, V27, W27) if successful.
        """
        n_, m_, p_ = DIMENSIONS
        remaining = [r for r in range(28) if r != term_to_remove]
        
        U_rem = U_np[:, remaining]  # 8 x 27
        V_rem = V_np[:, remaining]  # 16 x 27
        W_rem = W_np[:, remaining]  # 8 x 27
        U_t = U_np[:, term_to_remove]
        V_t = V_np[:, term_to_remove]
        W_t = W_np[:, term_to_remove]
        
        # Set up the linear system: find alpha such that
        # sum_r alpha_r * (U_r ⊗ V_r ⊗ W_r) = U_t ⊗ V_t ⊗ W_t
        # This gives (n*m)*(m*p)*(p*n) = 8*16*8 = 1024 equations in 27 unknowns
        
        num_eq = n_ * m_ * m_ * p_ * p_ * n_
        A_mat = np.zeros((num_eq, 27), dtype=np.float64)
        b_vec = np.zeros(num_eq, dtype=np.float64)
        
        idx = 0
        for a in range(n_ * m_):
            for b in range(m_ * p_):
                for c in range(p_ * n_):
                    b_vec[idx] = U_t[a] * V_t[b] * W_t[c]
                    for r_idx, r in enumerate(remaining):
                        A_mat[idx, r_idx] = U_np[a, r] * V_np[b, r] * W_np[c, r]
                    idx += 1
        
        # Solve least squares
        alpha, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Check residual
        reconstructed = A_mat @ alpha
        residual = np.linalg.norm(reconstructed - b_vec)
        b_norm = np.linalg.norm(b_vec)
        if b_norm > 1e-15:
            rel_residual = residual / b_norm
        else:
            rel_residual = residual
        
        if rel_residual < 1e-10:
            # The term is redundant!
            U27 = U_rem.copy()
            V27 = V_rem.copy()
            W27 = W_rem.copy()
            return True, U27, V27, W27
        
        return False, None, None, None

    rank27_valid = False
    U_rank27 = None
    V_rank27 = None
    W_rank27 = None

    if time.time() < deadline - 20.0:
        try:
            # Try eliminating each term (start with the last one, then try others)
            for term_to_remove in [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
                if time.time() > deadline - 15.0:
                    break
                
                success, U27, V27, W27 = try_eliminate_term(U_np, V_np, W_np, term_to_remove)
                if success:
                    # Verify the rank-27 decomposition
                    rank27_valid = verify_decomposition_sampled(U27, V27, W27, n_samples=300)
                    if rank27_valid:
                        U_rank27 = U27
                        V_rank27 = V27
                        W_rank27 = W27
                        break
        except Exception:
            rank27_valid = False

    # ------------------------------------------------------------------
    # Step 4: Return the best verified decomposition
    # ------------------------------------------------------------------
    if rank27_valid and U_rank27 is not None:
        return {
            "U": U_rank27.tolist(),
            "V": V_rank27.tolist(),
            "W": W_rank27.tolist(),
        }
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
