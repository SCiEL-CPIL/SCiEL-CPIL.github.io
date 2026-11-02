import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s for safety margin

    u_rows = n * m
    v_rows = m * p
    w_rows = p * n

    # Build the multiplication tensor T
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some i,j,k
    T = np.zeros((u_rows, v_rows, w_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def check_exact(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    def try_rank_als(r, seed=42, max_iter=200):
        """Alternating minimization to find rank-r decomposition."""
        if time.time() > deadline - 5:
            return None

        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(u_rows, r) * 0.1
        V = np.random.randn(v_rows, r) * 0.1
        W = np.random.randn(w_rows, r) * 0.1

        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None

            # Update U: fix V, W
            TW = V[:, None, :] * W[None, :, :]  # (v_rows, w_rows, r)
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (u_rows, r)
            G_U = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            try:
                U = np.linalg.solve(G_U, h_U.T).T
            except np.linalg.LinAlgError:
                pass

            # Update V: fix U, W
            UW = U[:, None, :] * W[None, :, :]  # (u_rows, w_rows, r)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (v_rows, r)
            G_V = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            try:
                V = np.linalg.solve(G_V, h_V.T).T
            except np.linalg.LinAlgError:
                pass

            # Update W: fix U, V
            UV = U[:, None, :] * V[None, :, :]  # (u_rows, v_rows, r)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (w_rows, r)
            G_W = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            try:
                W = np.linalg.solve(G_W, h_W.T).T
            except np.linalg.LinAlgError:
                pass

            # Early exit check
            if it % 10 == 0:
                T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
                if np.allclose(T_recon, T, atol=1e-8):
                    break

        return U, V, W

    def try_local_search(U, V, W, r):
        """
        Try to find an exact half-integer solution near the given U, V, W.
        We round to nearest half-integers and then check neighbors.
        """
        # Round to nearest half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if check_exact(U_r, V_r, W_r):
            return (r, U_r, V_r, W_r)
        
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_exact(U_i, V_i, W_i):
            return (r, U_i, V_i, W_i)
        
        # Local search: perturb the rounded values by +/- 0.5
        # This is a heuristic; for small r, we can afford to check some neighbors
        # We only do this if the error is small
        T_recon = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
        err = np.max(np.abs(T_recon - T))
        
        if err < 0.1:
            # Generate a few random perturbations of the rounded matrix
            # We pick a few random entries and flip them by 0.5
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                U_p = U_r.copy()
                V_p = V_r.copy()
                W_p = W_r.copy()
                
                # Perturb U
                idx_u = np.random.randint(0, U_p.size)
                row_u, col_u = np.unravel_index(idx_u, U_p.shape)
                U_p[row_u, col_u] += 0.5 * np.random.choice([-1, 1])
                
                # Perturb V
                idx_v = np.random.randint(0, V_p.size)
                row_v, col_v = np.unravel_index(idx_v, V_p.shape)
                V_p[row_v, col_v] += 0.5 * np.random.choice([-1, 1])
                
                # Perturb W
                idx_w = np.random.randint(0, W_p.size)
                row_w, col_w = np.unravel_index(idx_w, W_p.shape)
                W_p[row_w, col_w] += 0.5 * np.random.choice([-1, 1])
                
                if check_exact(U_p, V_p, W_p):
                    return (r, U_p, V_p, W_p)
                    
        return None

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(u_rows)]
        V = [[0] * rank for _ in range(v_rows)]
        W = [[0] * rank for _ in range(w_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Try ranks from a reasonable lower bound up to schoolbook - 1
    # Lower bound: max(n*p, n*m, m*p) = 15
    # Known results for 3x3x3 are 23, 3x3x4 is 28? Let's search from 15 to 44.
    ranks_to_try = list(range(15, best_rank))
    
    # Shuffle ranks to try diverse ones first? No, we want the minimal rank.
    # But maybe 15 is impossible. We iterate from low to high.
    
    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000, 2024, 3141, 2718, 111, 222, 333, 444, 555]

    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break

        for seed in seeds:
            if time.time() > deadline - 5:
                break
            
            # Run ALS
            U, V, W = try_rank_als(r, seed)
            
            if U is None:
                continue
                
            # Try to find exact solution near the ALS result
            result = try_local_search(U, V, W, r)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    best_result = {
                        "U": U_f.tolist(),
                        "V": V_f.tolist(),
                        "W": W_f.tolist()
                    }
                    break  # Found this rank, move to next lower rank (if any)

    return best_result
