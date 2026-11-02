import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    # Deadline: 140s to be safe (limit 150s)
    deadline = start_time + 140

    u_rows = n * m
    v_rows = m * p
    w_rows = p * n

    # Build the multiplication tensor T
    T = np.zeros((u_rows, v_rows, w_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def check_exact(U, V, W, tol=1e-9):
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    def discrete_local_search(U, V, W, max_iters=500):
        """
        Local search in the discrete space of half-integers.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Ensure we start with half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        current_err = np.max(np.abs(np.einsum('uc,vc,wc->uvw', U, V, W) - T))
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            if current_err < 1e-9:
                break
                
            improved = False
            
            # Helper to try updates on a matrix
            def try_updates(M, name):
                nonlocal current_err, improved
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        old_val = M[i, j]
                        # Try neighbors: old_val +/- 0.5
                        # Also try +/- 1.0 if stuck? No, 0.5 is the lattice step.
                        for delta in [-0.5, 0.5]:
                            new_val = old_val + delta
                            M[i, j] = new_val
                            err = np.max(np.abs(np.einsum('uc,vc,wc->uvw', U, V, W) - T))
                            if err < current_err - 1e-12: # Strict improvement
                                current_err = err
                                improved = True
                            else:
                                M[i, j] = old_val
                            
            try_updates(U, 'U')
            if not improved:
                try_updates(V, 'V')
            if not improved:
                try_updates(W, 'W')
                
            if not improved:
                break
                
        if current_err < 1e-9:
            return U, V, W
        return None

    def try_rank(r, seed=42, max_iter=100):
        """Alternating minimization to find rank-r decomposition."""
        if time.time() > deadline - 5:
            return None

        np.random.seed(seed)
        
        # Structured Initialization
        # Initialize with random values, but scale to be O(1)
        U = np.random.randn(u_rows, r)
        V = np.random.randn(v_rows, r)
        W = np.random.randn(w_rows, r)
        
        # Symmetry Breaking: Normalize columns so that max abs value is 1
        # This helps prevent the values from drifting to infinity or zero
        for c in range(r):
            norm_u = np.max(np.abs(U[:, c]))
            if norm_u > 0:
                U[:, c] /= norm_u
            norm_v = np.max(np.abs(V[:, c]))
            if norm_v > 0:
                V[:, c] /= norm_v
            norm_w = np.max(np.abs(W[:, c]))
            if norm_w > 0:
                W[:, c] /= norm_w

        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None

            # Update U: fix V, W
            # Minimize || T - sum_c U_c V_c W_c ||_F
            # Gradient w.r.t U_{u,c} is sum_{v,w} (T_{uvw} - recon_{uvw}) V_{v,c} W_{w,c}
            # Or use the normal equations:
            # sum_c U_{u,c} (sum_{v,w} V_{v,c} W_{w,c} V_{v,c'} W_{w,c'}) = sum_{v,w} T_{uvw} V_{v,c'} W_{w,c'}
            
            # Efficiently compute the Gram matrix and RHS
            # G_U[c, c'] = sum_{v,w} V_{v,c} W_{w,c} V_{v,c'} W_{w,c'}
            # h_U[u, c'] = sum_{v,w} T_{uvw} V_{v,c'} W_{w,c'}
            
            # We can compute this using einsum
            # TW = V[:, None, :] * W[None, :, :] # (v, w, r)
            # G_U = np.einsum('vca,wcb->ab', TW, TW) # (r, r)
            # h_U = np.einsum('uvw,wc,vc->uw', T, W, V) # Wait, indices
            # h_U[u, c] = sum_{v,w} T_{uvw} V_{v,c} W_{w,c}
            
            TW = V[:, None, :] * W[None, :, :]  # (v_rows, w_rows, r)
            G_U = np.einsum('vca,wcb->ab', TW, TW)  # (r, r)
            h_U = np.einsum('uvw,vc,wc->uc', T, V, W)  # (u_rows, r)
            
            try:
                U = np.linalg.solve(G_U, h_U.T).T
            except np.linalg.LinAlgError:
                pass

            # Update V
            UW = U[:, None, :] * W[None, :, :]  # (u_rows, w_rows, r)
            G_V = np.einsum('uca,wcb->ab', UW, UW)  # (r, r)
            h_V = np.einsum('uvw,uc,wc->vc', T, U, W)  # (v_rows, r)
            
            try:
                V = np.linalg.solve(G_V, h_V.T).T
            except np.linalg.LinAlgError:
                pass

            # Update W
            UV = U[:, None, :] * V[None, :, :]  # (u_rows, v_rows, r)
            G_W = np.einsum('uca,vcb->ab', UV, UV)  # (r, r)
            h_W = np.einsum('uvw,uc,vc->wc', T, U, V)  # (w_rows, r)
            
            try:
                W = np.linalg.solve(G_W, h_W.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Symmetry Breaking: Normalize columns
            for c in range(r):
                norm_u = np.max(np.abs(U[:, c]))
                if norm_u > 1e-9:
                    U[:, c] /= norm_u
                norm_v = np.max(np.abs(V[:, c]))
                if norm_v > 1e-9:
                    V[:, c] /= norm_v
                norm_w = np.max(np.abs(W[:, c]))
                if norm_w > 1e-9:
                    W[:, c] /= norm_w

            # Early exit check
            if it % 20 == 0:
                T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
                if np.allclose(T_recon, T, atol=1e-8):
                    break

        # Try rounding to half-integers and then discrete local search
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if already exact
        if check_exact(U_r, V_r, W_r):
            return (r, U_r, V_r, W_r)

        # Try discrete local search to fix the error
        result = discrete_local_search(U_r, V_r, W_r)
        if result is not None:
            return (r, result[0], result[1], result[2])

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

    # Ranks to try
    # Based on theory, R(3,3,5) is likely between 15 and 25.
    # We will try from 15 up to best_rank - 1.
    ranks_to_try = list(range(15, best_rank))
    
    # Prioritize lower ranks
    # We will try each rank with multiple seeds
    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000, 2024, 3141, 2718, 1, 2, 3, 4, 5]

    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            break

        for seed in seeds:
            if time.time() > deadline - 10:
                break
            result = try_rank(r, seed)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    best_result = {
                        "U": U_f.tolist(),
                        "V": V_f.tolist(),
                        "W": W_f.tolist()
                    }
                    break  # Found this rank, move to next lower rank

    return best_result
