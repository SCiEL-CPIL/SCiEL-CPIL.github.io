"""Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    import itertools
    import random

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    # Build initial rank-66 decomposition using Strassen blocks
    # Partition: A rows 1+2, cols 2+2; B rows 2+2, cols 2+2+2
    U66 = [[0.0] * 66 for _ in range(n * m)]
    V66 = [[0.0] * 66 for _ in range(m * p)]
    W66 = [[0.0] * 66 for _ in range(p * n)]

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
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

        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1

        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1

        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1

        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1

    # Build the decomposition
    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-8)

    if not verify(U, V, W, 66):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_tensor_matrix(U_cur, V_cur, W_cur, r_cur):
        """Build the 72 x r_cur matrix M where M[xyz, l] = U[i,j,l]*V[j,k,l]*W[k,i,l]."""
        nmp = n * m * p
        u3 = U_cur.reshape(n, m, r_cur)
        v3 = V_cur.reshape(m, p, r_cur)
        w3 = W_cur.reshape(p, n, r_cur)
        M = np.einsum('ijl,jkl,kil->ijk l', u3, v3, w3).reshape(nmp, r_cur)
        return M

    def try_remove_columns(U_cur, V_cur, W_cur, r_cur, cols_to_remove):
        """Try removing the specified columns simultaneously."""
        if len(cols_to_remove) == 0:
            return False, U_cur, V_cur, W_cur, r_cur
        
        U_new = np.delete(U_cur, list(cols_to_remove), axis=1)
        V_new = np.delete(V_cur, list(cols_to_remove), axis=1)
        W_new = np.delete(W_cur, list(cols_to_remove), axis=1)
        r_new = r_cur - len(cols_to_remove)
        
        if verify(U_new, V_new, W_new, r_new):
            return True, U_new, V_new, W_new, r_new
        return False, U_cur, V_cur, W_cur, r_cur

    def aggressive_rank_reduction(U_cur, V_cur, W_cur, r_cur, max_iterations=20):
        """Aggressively reduce rank by finding and removing linearly dependent columns."""
        best_rank = r_cur
        best_U = U_cur.copy()
        best_V = V_cur.copy()
        best_W = W_cur.copy()
        
        for iteration in range(max_iterations):
            if time.time() > _deadline - 15:
                break
                
            r = best_rank
            if r <= 1:
                break
                
            M = build_tensor_matrix(best_U, best_V, best_W, r)
            
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-9
                rank_M = int(np.sum(s > tol))
                
                if rank_M >= r:
                    break
                
                null_dim = r - rank_M
                null_space = Vt[rank_M:, :].T  # (r, null_dim)
                
                found_improvement = False
                
                # Try removing multiple columns at once
                if null_dim >= 2:
                    for nv1 in range(null_dim):
                        if time.time() > _deadline - 15:
                            break
                        for nv2 in range(nv1 + 1, null_dim):
                            if time.time() > _deadline - 15:
                                break
                            for w1 in [0.5, 1.0, 2.0]:
                                nv = w1 * null_space[:, nv1] + null_space[:, nv2]
                                abs_nv = np.abs(nv)
                                order = np.argsort(abs_nv)[::-1]
                                
                                top2 = [order[0], order[1]]
                                success, _, _, _, _ = try_remove_columns(
                                    best_U, best_V, best_W, r, top2)
                                if success:
                                    best_U, best_V, best_W, best_rank = try_remove_columns(
                                        best_U, best_V, best_W, r, top2)[1:]
                                    found_improvement = True
                                    break
                            
                            if found_improvement:
                                break
                        if found_improvement:
                            break
                
                if not found_improvement:
                    for nv_idx in range(null_dim):
                        if time.time() > _deadline - 15:
                            break
                        nv = null_space[:, nv_idx]
                        abs_nv = np.abs(nv)
                        order = np.argsort(abs_nv)[::-1]
                        
                        for max_idx in order:
                            if time.time() > _deadline - 15:
                                break
                            if abs_nv[max_idx] < tol:
                                continue
                            
                            success, _, _, _, _ = try_remove_columns(
                                best_U, best_V, best_W, r, [max_idx])
                            if success:
                                best_U, best_V, best_W, best_rank = try_remove_columns(
                                    best_U, best_V, best_W, r, [max_idx])[1:]
                                found_improvement = True
                                break
                        
                        if found_improvement:
                            break
                
                if not found_improvement:
                    break
                    
            except Exception:
                break
        
        return best_rank, best_U, best_V, best_W

    def gf13_peeling(U_cur, V_cur, W_cur, r_cur, max_rounds=50):
        """
        Use GF(13) arithmetic to detect and remove linearly dependent rank-1 components.
        This is a randomized iterative peeling algorithm similar to Wiedemann's method.
        """
        P = 13
        best_rank = r_cur
        best_U = U_cur.copy()
        best_V = V_cur.copy()
        best_W = W_cur.copy()
        
        for round_idx in range(max_rounds):
            if time.time() > _deadline - 20:
                break
            
            r = best_rank
            if r <= 2:
                break
            
            # Build the tensor matrix over GF(13)
            u3 = best_U.reshape(n, m, r)
            v3 = best_V.reshape(m, p, r)
            w3 = best_W.reshape(p, n, r)
            
            # M[i,j,k, l] = u[i,j,l] * v[j,k,l] * w[k,i,l] mod P
            M = np.einsum('ijl,jkl,kil->ijkl', u3, v3, w3)
            M = (M % P).astype(np.int64)
            M = M.reshape(n * m * p, r)
            
            # Find rank over GF(13)
            # Use Gaussian elimination
            M_work = M.copy()
            rank_gf = 0
            col_used = set()
            
            for col in range(r):
                if col in col_used:
                    continue
                # Find pivot
                pivot_row = -1
                for row in range(rank_gf, n * m * p):
                    if M_work[row, col] != 0:
                        pivot_row = row
                        break
                
                if pivot_row == -1:
                    continue
                
                # Swap rows
                M_work[[rank_gf, pivot_row]] = M_work[[pivot_row, rank_gf]]
                
                # Normalize pivot row
                pivot_val = M_work[rank_gf, col]
                inv_pivot = pow(pivot_val, P - 2, P)
                M_work[rank_gf, :] = (M_work[rank_gf, :] * inv_pivot) % P
                
                # Eliminate
                for row in range(n * m * p):
                    if row != rank_gf and M_work[row, col] != 0:
                        factor = M_work[row, col]
                        M_work[row, :] = (M_work[row, :] - factor * M_work[rank_gf, :]) % P
                
                rank_gf += 1
                col_used.add(col)
            
            if rank_gf >= r:
                break
            
            # Find nullspace vectors over GF(13)
            # The nullspace dimension is r - rank_gf
            # For each free column, find a null vector
            free_cols = [c for c in range(r) if c not in col_used]
            
            found_improvement = False
            
            for free_col in free_cols:
                if time.time() > _deadline - 20:
                    break
                
                # Construct a null vector: set free_col = 1, solve for pivot columns
                nv = np.zeros(r, dtype=np.int64)
                nv[free_col] = 1
                
                # Back-substitute
                pivot_cols = sorted(col_used, reverse=True)
                for pc in pivot_cols:
                    # Find which row this pivot is in
                    for row in range(rank_gf):
                        if M_work[row, pc] != 0:
                            # Check if this is the pivot row for pc
                            # In our elimination, row r has pivot at column col_used[r]
                            # Actually, let's just do direct substitution
                            break
                    
                    # Compute nv[pc] from the equation
                    # sum over all cols: M_work[row, col] * nv[col] = 0
                    # nv[pc] = -sum_{col != pc} M_work[row, col] * nv[col]
                    val = 0
                    for c in range(r):
                        if c != pc:
                            val = (val + M_work[row, c] * nv[c]) % P
                    nv[pc] = (-val) % P
                
                # Verify this is actually in the nullspace
                check = (M @ nv) % P
                if np.any(check != 0):
                    continue
                
                # Try to remove the column with largest absolute value in nv
                abs_nv = np.abs(nv.astype(float))
                order = np.argsort(abs_nv)[::-1]
                
                # Try removing top 1 or top 2 columns
                for num_remove in [1, 2]:
                    cols_to_remove = [order[i] for i in range(num_remove)]
                    success, _, _, _, _ = try_remove_columns(
                        best_U, best_V, best_W, r, cols_to_remove)
                    if success:
                        best_U, best_V, best_W, best_rank = try_remove_columns(
                            best_U, best_V, best_W, r, cols_to_remove)[1:]
                        found_improvement = True
                        break
                
                if found_improvement:
                    break
            
            if not found_improvement:
                break
        
        return best_rank, best_U, best_V, best_W

    def local_search(U_cur, V_cur, W_cur, r_cur, max_steps=100):
        """
        Local search: try small perturbations to U, V, W entries to find lower rank.
        We'll try to merge pairs of rank-1 terms by finding coefficients that cancel.
        """
        best_rank = r_cur
        best_U = U_cur.copy()
        best_V = V_cur.copy()
        best_W = W_cur.copy()
        
        # Try to find pairs of columns that can be merged
        # For columns a and b, we want to find alpha, beta such that
        # the combined term still contributes to the tensor
        # This is equivalent to checking if the span of columns a and b
        # can be represented by a single column
        
        for step in range(max_steps):
            if time.time() > _deadline - 25:
                break
            
            r = best_rank
            if r <= 2:
                break
            
            # Try all pairs of columns
            found = False
            for a in range(r):
                if time.time() > _deadline - 25:
                    break
                for b in range(a + 1, r):
                    if time.time() > _deadline - 25:
                        break
                    
                    # Check if removing either a or b works
                    for col_to_remove in [a, b]:
                        success, _, _, _, _ = try_remove_columns(
                            best_U, best_V, best_W, r, [col_to_remove])
                        if success:
                            best_U, best_V, best_W, best_rank = try_remove_columns(
                                best_U, best_V, best_W, r, [col_to_remove])[1:]
                            found = True
                            break
                    
                    if found:
                        break
                
                if found:
                    break
            
            if not found:
                break
        
        return best_rank, best_U, best_V, best_W

    # First pass: standard SVD-based rank reduction
    best_rank, best_U, best_V, best_W = aggressive_rank_reduction(U, V, W, 66, max_iterations=15)
    
    # Second pass: GF(13) peeling
    if time.time() < _deadline - 40 and best_rank > 50:
        best_rank, best_U, best_V, best_W = gf13_peeling(
            best_U, best_V, best_W, best_rank, max_rounds=30)
    
    # Third pass: local search
    if time.time() < _deadline - 30 and best_rank > 50:
        best_rank, best_U, best_V, best_W = local_search(
            best_U, best_V, best_W, best_rank, max_steps=50)
    
    # Fourth pass: more aggressive SVD reduction if we have time
    if time.time() < _deadline - 20 and best_rank > 50:
        best_rank, best_U, best_V, best_W = aggressive_rank_reduction(
            best_U, best_V, best_W, best_rank, max_iterations=10)
    
    if best_rank < 66:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
