"""Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    import itertools

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    # Build initial rank-66 decomposition using Strassen blocks
    # Partition: A rows 1+2, cols 2+2; B rows 2+2, cols 2+2+2
    # Total: 3*(4+4) + 3*(7+7) = 24 + 42 = 66
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

    def randomized_linear_algebraic_reduction(U_cur, V_cur, W_cur, r_cur, num_trials=200):
        """
        Apply randomized linear algebraic reduction:
        1. Construct random linear combinations of the 72 tensor flattening matrices
        2. Form a single matrix and compute its rank
        3. If rank < r_cur, use nullspace to identify removable columns
        """
        best_rank = r_cur
        best_U = U_cur.copy()
        best_V = V_cur.copy()
        best_W = W_cur.copy()
        
        nmp = n * m * p
        
        for trial in range(num_trials):
            if time.time() > _deadline - 10:
                break
            
            r = best_rank
            if r <= 1:
                break
            
            # Build the tensor matrix M (72 x r)
            M = build_tensor_matrix(best_U, best_V, best_W, r)
            
            # Create random linear combination of rows
            # This is equivalent to multiplying M by a random vector from the left
            # But we want to find dependencies among columns
            
            # Alternative approach: use SVD to find the null space
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=True)
                tol = 1e-10
                rank_M = int(np.sum(s > tol))
                
                if rank_M >= r:
                    # Try a different random perturbation
                    # Add small random noise to break symmetries
                    M_perturbed = M + np.random.randn(nmp, r) * 1e-6
                    U_svd, s, Vt = np.linalg.svd(M_perturbed, full_matrices=True)
                    rank_M = int(np.sum(s > tol))
                    
                    if rank_M >= r:
                        continue
                
                null_dim = r - rank_M
                if null_dim <= 0:
                    continue
                
                # Get null space vectors
                null_space = Vt[rank_M:, :].T  # (r, null_dim)
                
                found_improvement = False
                
                # Try single column removals based on null space
                for nv_idx in range(null_dim):
                    if time.time() > _deadline - 10:
                        break
                    
                    nv = null_space[:, nv_idx]
                    abs_nv = np.abs(nv)
                    order = np.argsort(abs_nv)[::-1]
                    
                    for col_idx in order:
                        if time.time() > _deadline - 10:
                            break
                        if abs_nv[col_idx] < tol:
                            continue
                        
                        success, U_new, V_new, W_new, r_new = try_remove_columns(
                            best_U, best_V, best_W, r, [col_idx])
                        if success:
                            best_U = U_new
                            best_V = V_new
                            best_W = W_new
                            best_rank = r_new
                            found_improvement = True
                            break
                    
                    if found_improvement:
                        break
                
                # Try pair removals
                if not found_improvement and null_dim >= 2:
                    for nv1 in range(null_dim):
                        if time.time() > _deadline - 10:
                            break
                        for nv2 in range(nv1 + 1, null_dim):
                            if time.time() > _deadline - 10:
                                break
                            
                            nv = null_space[:, nv1] + null_space[:, nv2]
                            abs_nv = np.abs(nv)
                            order = np.argsort(abs_nv)[::-1]
                            
                            top2 = [order[0], order[1]]
                            success, U_new, V_new, W_new, r_new = try_remove_columns(
                                best_U, best_V, best_W, r, top2)
                            if success:
                                best_U = U_new
                                best_V = V_new
                                best_W = W_new
                                best_rank = r_new
                                found_improvement = True
                                break
                        
                        if found_improvement:
                            break
                
                if not found_improvement:
                    # Try a different random combination
                    random_weights = np.random.randn(nmp)
                    M_weighted = M * random_weights[:, np.newaxis]
                    U_svd, s, Vt = np.linalg.svd(M_weighted, full_matrices=True)
                    rank_M = int(np.sum(s > tol))
                    
                    if rank_M < r:
                        null_dim = r - rank_M
                        null_space = Vt[rank_M:, :].T
                        
                        for nv_idx in range(null_dim):
                            nv = null_space[:, nv_idx]
                            abs_nv = np.abs(nv)
                            order = np.argsort(abs_nv)[::-1]
                            
                            for col_idx in order:
                                if abs_nv[col_idx] < tol:
                                    continue
                                
                                success, U_new, V_new, W_new, r_new = try_remove_columns(
                                    best_U, best_V, best_W, r, [col_idx])
                                if success:
                                    best_U = U_new
                                    best_V = V_new
                                    best_W = W_new
                                    best_rank = r_new
                                    found_improvement = True
                                    break
                                if found_improvement:
                                    break
                            if found_improvement:
                                break
                
                if not found_improvement:
                    break
                    
            except Exception:
                continue
        
        return best_rank, best_U, best_V, best_W

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
                # First, try pairs of columns from nullspace combinations
                if null_dim >= 2:
                    for nv1 in range(null_dim):
                        if time.time() > _deadline - 15:
                            break
                        for nv2 in range(nv1 + 1, null_dim):
                            if time.time() > _deadline - 15:
                                break
                            # Combine two nullspace vectors
                            for w1 in [0.5, 1.0, 2.0]:
                                nv = w1 * null_space[:, nv1] + null_space[:, nv2]
                                abs_nv = np.abs(nv)
                                order = np.argsort(abs_nv)[::-1]
                                
                                # Try removing top 2 columns
                                top2 = [order[0], order[1]]
                                success, _, _, _, _ = try_remove_columns(
                                    best_U, best_V, best_W, r, top2)
                                if success:
                                    best_rank = r - 2
                                    best_U, best_V, best_W, best_rank = try_remove_columns(
                                        best_U, best_V, best_W, r, top2)[1:]
                                    found_improvement = True
                                    break
                            
                            if found_improvement:
                                break
                        if found_improvement:
                            break
                
                # If no pair worked, try single columns
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

    # First pass: randomized linear algebraic reduction
    best_rank, best_U, best_V, best_W = randomized_linear_algebraic_reduction(U, V, W, 66, num_trials=100)
    
    # Second pass: aggressive rank reduction
    if time.time() < _deadline - 20 and best_rank > 50:
        best_rank, best_U, best_V, best_W = aggressive_rank_reduction(
            best_U, best_V, best_W, best_rank, max_iterations=10)
    
    # Third pass: try once more if we have time
    if time.time() < _deadline - 15 and best_rank > 50:
        best_rank, best_U, best_V, best_W = randomized_linear_algebraic_reduction(
            best_U, best_V, best_W, best_rank, num_trials=50)
    
    if best_rank < 66:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
