"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    import random

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_decomposition():
        """Build a block-structured rank-66 decomposition."""
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

            # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
            U66[A00][col] += 1; U66[A11][col] += 1
            V66[B00][col] += 1; V66[B11][col] += 1
            W66[C00][col] += 1; W66[C11][col] += 1
            col += 1

            # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
            U66[A10][col] += 1; U66[A11][col] += 1
            V66[B00][col] += 1
            W66[C10][col] += 1; W66[C11][col] -= 1
            col += 1

            # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
            U66[A00][col] += 1
            V66[B01][col] += 1; V66[B11][col] -= 1
            W66[C01][col] += 1; W66[C11][col] += 1
            col += 1

            # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
            U66[A11][col] += 1
            V66[B10][col] += 1; V66[B00][col] -= 1
            W66[C00][col] += 1; W66[C10][col] += 1
            col += 1

            # M5 = (A00+A01)*B11 -> C01 += M5, C00 -= M5
            U66[A00][col] += 1; U66[A01][col] += 1
            V66[B11][col] += 1
            W66[C00][col] -= 1; W66[C01][col] += 1
            col += 1

            # M6 = (A10-A00)*(B00+B01) -> C11 += M6
            U66[A10][col] += 1; U66[A00][col] -= 1
            V66[B00][col] += 1; V66[B01][col] += 1
            W66[C11][col] += 1
            col += 1

            # M7 = (A01-A11)*(B10+B11) -> C00 += M7
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

            # Direct: A0*B00 -> C0, A1*B10 -> C0
            U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
            col += 1
            U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
            col += 1
            # Direct: A0*B01 -> C1, A1*B11 -> C1
            U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
            col += 1
            U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
            col += 1

        # Build the decomposition
        # Row 0: direct terms for all 6 column positions (3 groups of 2)
        for t0 in [0, 2, 4]:
            add_1x2_2x2(0, 0, 0, t0, 0, t0)
            add_1x2_2x2(0, 2, 2, t0, 0, t0)

        # Rows 1-2: Strassen for all 6 column positions (3 groups of 2)
        for t0 in [0, 2, 4]:
            add_strassen(1, 0, 0, t0, 1, t0)
            add_strassen(1, 2, 2, t0, 1, t0)

        U = np.array(U66, dtype=float)
        V = np.array(V66, dtype=float)
        W = np.array(W66, dtype=float)
        return U, V, W

    def try_reduce_rank(U, V, W, r, max_reductions=5):
        """Try to reduce rank by finding linear dependencies via SVD."""
        best_rank = r
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for _ in range(max_reductions):
            if time.time() > _deadline - 20:
                break
                
            nmp = n * m * p
            M = np.zeros((nmp, best_rank))
            for l in range(best_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M[i * m * p + j * p + k, l] = (
                                best_U[i * m + j, l] * 
                                best_V[j * p + k, l] * 
                                best_W[k * n + i, l]
                            )
            
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-10
                rank_M = np.sum(s > tol)
                
                if rank_M < best_rank:
                    null_space = Vt[rank_M:, :].T
                    
                    found = False
                    for nv in null_space:
                        if time.time() > _deadline - 20:
                            break
                        max_idx = np.argmax(np.abs(nv))
                        if abs(nv[max_idx]) < tol:
                            continue
                        
                        U_new = np.delete(best_U, max_idx, axis=1)
                        V_new = np.delete(best_V, max_idx, axis=1)
                        W_new = np.delete(best_W, max_idx, axis=1)
                        
                        if verify(U_new, V_new, W_new, best_rank - 1):
                            best_U, best_V, best_W = U_new, V_new, W_new
                            best_rank -= 1
                            found = True
                            break
                    
                    if not found:
                        break
                else:
                    break
            except:
                break
        
        return best_rank, best_U, best_V, best_W

    def try_rank1_peeling(U, V, W, r, max_iterations=10):
        """Try to reduce rank by peeling off rank-1 components."""
        best_rank = r
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for _ in range(max_iterations):
            if time.time() > _deadline - 20:
                break
            
            # Compute the current tensor
            T_current = np.zeros((n, m, p))
            for l in range(best_rank):
                T_current += best_U[:, l].reshape(n, m) * best_V[:, l].reshape(m, p) * best_W[:, l].reshape(p, n).T
            
            # Try to find a rank-1 component that can be subtracted
            # We look for a direction where the tensor has a "corner" structure
            # that allows us to peel off a rank-1 term
            
            # For each possible rank-1 term, check if subtracting it still gives
            # a valid tensor (all entries non-negative in some sense, or exactly zero
            # in the right places)
            
            # A simpler approach: try to find a column that is "redundant" by checking
            # if it can be expressed as a combination of others
            # This is similar to the SVD approach but with a different perspective
            
            # Actually, let's try a different approach: look for pairs of columns
            # that can be merged
            merged = False
            for i in range(best_rank):
                if time.time() > _deadline - 20:
                    break
                for j in range(i + 1, best_rank):
                    # Try to merge columns i and j by finding a linear combination
                    # that preserves the tensor
                    # This is complex, so let's try a simpler heuristic:
                    # Check if column i is approximately a multiple of column j
                    # in all three matrices simultaneously
                    
                    # Compute the "angle" between the columns
                    u_i, u_j = best_U[:, i], best_U[:, j]
                    v_i, v_j = best_V[:, i], best_V[:, j]
                    w_i, w_j = best_W[:, i], best_W[:, j]
                    
                    # If the columns are linearly dependent in a way that allows
                    # merging, we might be able to reduce the rank
                    # For now, this is a placeholder for a more sophisticated
                    # merging strategy
                    
                    if time.time() > _deadline - 20:
                        break
                if time.time() > _deadline - 20:
                    break
            
            if not merged:
                break
        
        return best_rank, best_U, best_V, best_W

    # Build initial decomposition
    U, V, W = build_decomposition()
    
    # Verify rank-66
    if not verify(U, V, W, 66):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via SVD-based method
    best_rank, best_U, best_V, best_W = try_reduce_rank(U, V, W, 66, max_reductions=5)
    
    if time.time() > _deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to further reduce rank via rank-1 peeling
    best_rank2, best_U2, best_V2, best_W2 = try_rank1_peeling(best_U, best_V, best_W, best_rank, max_iterations=10)
    
    if best_rank2 < best_rank:
        best_rank, best_U, best_V, best_W = best_rank2, best_U2, best_V2, best_W2
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
