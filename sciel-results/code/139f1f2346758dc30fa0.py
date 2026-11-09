import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (150s limit)
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # The tensor M_{3,3,5} is the direct sum of M_{3,3,2} and M_{3,3,3}
    # because the output columns are independent.
    # T_{335} = T_{332} (cols 0,1) + T_{333} (cols 2,3,4)
    
    # We will find decompositions for:
    # 1. M_{3,3,2}: Rank <= 18 (schoolbook), try to find lower.
    # 2. M_{3,3,3}: Rank <= 27 (schoolbook), try to find 21.
    
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i*m + j
                    b_idx = j*p + k
                    c_idx = k*n + i
                    T[a_idx, b_idx, c_idx] = 1.0
        return T
    
    def schoolbook_decomposition(n, m, p):
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, r
    
    def als_decompose(T, target_rank, max_iters=500, n_restarts=10, tol=1e-10):
        """
        Find CP decomposition of T with rank target_rank.
        Returns (U, V, W, success)
        """
        n_dim_a, n_dim_b, n_dim_c = T.shape
        T_norm = np.linalg.norm(T)
        
        def check_convergence(U, V, W, tol=1e-10):
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - T_recon) / T_norm
            return err < tol, err
        
        def polish(U, V, W, max_iters=50, tol=1e-12):
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Update U
                M = (V.T @ V) * (W.T @ W)
                rhs = np.einsum('abc,br,cr->ar', T, V, W)
                try:
                    U = np.linalg.solve(M.T, rhs.T).T
                except np.linalg.LinAlgError:
                    U = np.linalg.lstsq(M.T, rhs.T, rcond=None)[0].T
                
                # Update V
                M_v = (U.T @ U) * (W.T @ W)
                rhs_v = np.einsum('abc,ar,cr->br', T, U, W)
                try:
                    V = np.linalg.solve(M_v.T, rhs_v.T).T
                except np.linalg.LinAlgError:
                    V = np.linalg.lstsq(M_v.T, rhs_v.T, rcond=None)[0].T
                
                # Update W
                M_w = (U.T @ U) * (V.T @ V)
                rhs_w = np.einsum('abc,ar,br->cr', T, U, V)
                try:
                    W = np.linalg.solve(M_w.T, rhs_w.T).T
                except np.linalg.LinAlgError:
                    W = np.linalg.lstsq(M_w.T, rhs_w.T, rcond=None)[0].T
                
                conv, err = check_convergence(U, V, W, tol)
                if conv:
                    break
            return U, V, W

        for restart in range(n_restarts):
            if time.time() > deadline:
                return None, None, None, False
            
            # Initialize
            if restart == 0:
                U = np.random.randn(n_dim_a, target_rank)
                V = np.random.randn(n_dim_b, target_rank)
                W = np.random.randn(n_dim_c, target_rank)
            elif restart == 1:
                # Sparse init
                U = np.random.randn(n_dim_a, target_rank) * (np.random.rand(n_dim_a, target_rank) > 0.5)
                V = np.random.randn(n_dim_b, target_rank) * (np.random.rand(n_dim_b, target_rank) > 0.5)
                W = np.random.randn(n_dim_c, target_rank) * (np.random.rand(n_dim_c, target_rank) > 0.5)
            else:
                U = np.random.randn(n_dim_a, target_rank) * 0.1
                V = np.random.randn(n_dim_b, target_rank) * 0.1
                W = np.random.randn(n_dim_c, target_rank) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None, None, None, False
                
                # Update U
                M = (V.T @ V) * (W.T @ W)
                rhs = np.einsum('abc,br,cr->ar', T, V, W)
                try:
                    U_new = np.linalg.solve(M.T, rhs.T).T
                except np.linalg.LinAlgError:
                    U_new = np.linalg.lstsq(M.T, rhs.T, rcond=None)[0].T
                
                # Update V
                M_v = (U_new.T @ U_new) * (W.T @ W)
                rhs_v = np.einsum('abc,ar,cr->br', T, U_new, W)
                try:
                    V_new = np.linalg.solve(M_v.T, rhs_v.T).T
                except np.linalg.LinAlgError:
                    V_new = np.linalg.lstsq(M_v.T, rhs_v.T, rcond=None)[0].T
                
                # Update W
                M_w = (U_new.T @ U_new) * (V_new.T @ V_new)
                rhs_w = np.einsum('abc,ar,br->cr', T, U_new, V_new)
                try:
                    W_new = np.linalg.solve(M_w.T, rhs_w.T).T
                except np.linalg.LinAlgError:
                    W_new = np.linalg.lstsq(M_w.T, rhs_w.T, rcond=None)[0].T
                
                U, V, W = U_new, V_new, W_new
                
                if it % 10 == 0 or it == max_iters - 1:
                    conv, err = check_convergence(U, V, W, 1e-10)
                    if conv:
                        U, V, W = polish(U, V, W, max_iters=50, tol=1e-12)
                        conv_final, _ = check_convergence(U, V, W, 1e-10)
                        if conv_final:
                            return U, V, W, True
        
        return None, None, None, False

    # 1. Solve M_{3,3,3}
    T_333 = build_tensor(3, 3, 3)
    U_333, V_333, W_333, success_333 = als_decompose(T_333, 21, max_iters=300, n_restarts=15)
    if not success_333:
        # Fallback to schoolbook for 3x3x3
        U_333, V_333, W_333, r_333 = schoolbook_decomposition(3, 3, 3)
    else:
        r_333 = 21

    # 2. Solve M_{3,3,2}
    T_332 = build_tensor(3, 3, 2)
    # Try to find rank < 18? Unlikely, but try 18 first.
    U_332, V_332, W_332, success_332 = als_decompose(T_332, 18, max_iters=200, n_restarts=10)
    if not success_332:
        # Fallback to schoolbook for 3x3x2
        U_332, V_332, W_332, r_332 = schoolbook_decomposition(3, 3, 2)
    else:
        r_332 = 18

    # 3. Combine
    # The full tensor T_335 is T_332 (cols 0,1 of C) + T_333 (cols 2,3,4 of C)
    # U and V are the same for both parts? 
    # No, U and V depend on the specific tensor.
    # Wait. The multiplication tensor T_{n,m,p} is defined by T[i,j,k]=1 if j_A=j_B, i_A=i_C, k_B=k_C.
    # For M_{3,3,2}, k_B is in {0,1}, k_C is in {0,1}.
    # For M_{3,3,3}, k_B is in {2,3,4}, k_C is in {2,3,4}.
    # The indices i and j are the same.
    # So U and V for the combined tensor will have the same rows for i,j, but different columns for the rank components?
    # Yes.
    # U_combined[i, r] = U_332[i, r] for r < r_332, U_333[i, r-r_332] for r >= r_332.
    # V_combined[j, r] = V_332[j, r] for r < r_332, V_333[j, r-r_332] for r >= r_332.
    # W_combined[c, r] = W_332[c, r] for r < r_332, W_333[c, r-r_332] for r >= r_332.
    # BUT: W_332 has shape (2*3, r_332) = (6, r_332). It corresponds to output cols 0,1.
    # W_333 has shape (3*3, r_333) = (9, r_333). It corresponds to output cols 2,3,4.
    # The full W has shape (5*3, r_total) = (15, r_total).
    # We need to map the local W indices to global W indices.
    
    r_total = r_332 + r_333
    U_full = np.zeros((9, r_total))
    V_full = np.zeros((15, r_total))
    W_full = np.zeros((15, r_total))
    
    # Fill U and V (same rows for both parts)
    U_full[:, :r_332] = U_332
    V_full[:, :r_332] = V_332
    U_full[:, r_332:] = U_333
    V_full[:, r_332:] = V_333
    
    # Fill W
    # W_332 corresponds to k_C in {0,1}. Global index c = k*n + i.
    # For k_C=0, i in 0..2 -> c in 0..2.
    # For k_C=1, i in 0..2 -> c in 3..5.
    # W_333 corresponds to k_C in {2,3,4}.
    # For k_C=2, i in 0..2 -> c in 6..8.
    # For k_C=3, i in 0..2 -> c in 9..11.
    # For k_C=4, i in 0..2 -> c in 12..14.
    
    # W_332 has rows indexed by local k_C (0,1) and i (0,1,2).
    # Local index in W_332: l = k_local * 3 + i.
    # Global index in W_full: c = k_global * 3 + i.
    
    for k_local in range(2):
        for i in range(3):
            l_idx = k_local * 3 + i
            c_idx = k_local * 3 + i
            W_full[c_idx, :r_332] = W_332[l_idx, :]
            
    for k_local in range(3):
        for i in range(3):
            l_idx = k_local * 3 + i
            c_idx = (k_local + 2) * 3 + i  # Offset by 2
            W_full[c_idx, r_332:] = W_333[l_idx, :]

    # Verify
    T_full = build_tensor(3, 3, 5)
    T_recon = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    err = np.linalg.norm(T_full - T_recon) / np.linalg.norm(T_full)
    
    if err > 1e-9:
        # Fallback to full schoolbook
        U_full, V_full, W_full, r_total = schoolbook_decomposition(3, 3, 5)
        
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
