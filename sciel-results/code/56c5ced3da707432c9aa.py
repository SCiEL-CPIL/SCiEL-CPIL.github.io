import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, atol=1e-8):
        """Check if U, V, W form a valid decomposition of T."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_approx = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_approx, T, atol=atol)

    def to_list(arr):
        return [[float(x) for x in row] for row in arr]

    # Fallback: Rank 28 solution
    def get_rank_28_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
        r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1
        r += 1
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1
        W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1
        r += 1
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1
        W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
        r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1
        r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1
        r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1
        r += 1
        U[0 * m + 0][r] = 1
        V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1
        W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1
        r += 1
        U[1 * m + 1][r] = 1
        V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1
        W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1
        r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1
        r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        r += 1
        
        return {"U": U, "V": V, "W": W}

    def polish_solution(U, V, W, max_iters=100, tol=1e-12):
        """High-precision polish pass using tighter tolerance and more iterations."""
        if time.time() > deadline:
            return None
        
        T_flat_U = T.reshape(n * m, -1)
        T_flat_V = T.reshape(m * p, -1)
        T_flat_W = T.reshape(p * n, -1)
        target_rank = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U with higher precision
            B_mat_T = np.zeros((m * p * p * n, target_rank))
            for r in range(target_rank):
                B_mat_T[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            try:
                P_U = np.linalg.pinv(B_mat_T, rcond=1e-14)
                U_new = T_flat_U @ P_U
            except:
                break
            
            # Update V
            B_mat_T_V = np.zeros((n * m * p * n, target_rank))
            for r in range(target_rank):
                B_mat_T_V[:, r] = np.outer(U_new[:, r], W[:, r]).flatten()
            try:
                P_V = np.linalg.pinv(B_mat_T_V, rcond=1e-14)
                V_new = T_flat_V @ P_V
            except:
                break
            
            # Update W
            B_mat_T_W = np.zeros((n * m * m * p, target_rank))
            for r in range(target_rank):
                B_mat_T_W[:, r] = np.outer(U_new[:, r], V_new[:, r]).flatten()
            try:
                P_W = np.linalg.pinv(B_mat_T_W, rcond=1e-14)
                W_new = T_flat_W @ P_W
            except:
                break
            
            # Check convergence
            if (np.allclose(U, U_new, atol=tol) and 
                np.allclose(V, V_new, atol=tol) and 
                np.allclose(W, W_new, atol=tol)):
                U, V, W = U_new, V_new, W_new
                break
            
            U, V, W = U_new, V_new, W_new
        
        if verify(U, V, W):
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        return None
    
    def try_als_for_rank(target_rank, max_iters=80, n_restarts=15):
        if time.time() > deadline:
            return None
        
        T_flat_U = T.reshape(n * m, -1)
        T_flat_V = T.reshape(m * p, -1)
        T_flat_W = T.reshape(p * n, -1)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart)
            if restart % 3 == 0:
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
            elif restart % 3 == 1:
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                mask = np.random.random((n * m, target_rank)) > 0.7
                U[mask] = 0
                mask = np.random.random((m * p, target_rank)) > 0.7
                V[mask] = 0
                mask = np.random.random((p * n, target_rank)) > 0.7
                W[mask] = 0
            else:
                U = np.random.randint(0, 2, (n * m, target_rank)) - 0.5
                V = np.random.randint(0, 2, (m * p, target_rank)) - 0.5
                W = np.random.randint(0, 2, (p * n, target_rank)) - 0.5
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                B_mat_T = np.zeros((m * p * p * n, target_rank))
                for r in range(target_rank):
                    B_mat_T[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                try:
                    P_U = np.linalg.pinv(B_mat_T)
                    U = T_flat_U @ P_U
                except:
                    break
                
                B_mat_T_V = np.zeros((n * m * p * n, target_rank))
                for r in range(target_rank):
                    B_mat_T_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                try:
                    P_V = np.linalg.pinv(B_mat_T_V)
                    V = T_flat_V @ P_V
                except:
                    break
                
                B_mat_T_W = np.zeros((n * m * m * p, target_rank))
                for r in range(target_rank):
                    B_mat_T_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                try:
                    P_W = np.linalg.pinv(B_mat_T_W)
                    W = T_flat_W @ P_W
                except:
                    break
                
                if verify(U, V, W):
                    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                
                if it % 20 == 19:
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(T_approx - T))
                    if err < 1e-4:
                        polished = polish_solution(U, V, W)
                        if polished:
                            return polished
                    
        return None

    def greedy_matching_pursuit(target_rank, n_restarts=20):
        """
        Apply random linear basis changes to input spaces, then use greedy
        matching pursuit to find a sparse rank-1 decomposition.
        """
        if time.time() > deadline:
            return None
        
        T_flat_U = T.reshape(n * m, -1)
        T_flat_V = T.reshape(m * p, -1)
        T_flat_W = T.reshape(p * n, -1)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + restart * 7)
            
            # Random invertible basis changes
            A_basis = np.random.randn(n * m, n * m)
            B_basis = np.random.randn(m * p, m * p)
            C_basis = np.random.randn(p * n, p * n)
            
            # Ensure invertibility
            if abs(np.linalg.det(A_basis)) < 1e-6:
                continue
            if abs(np.linalg.det(B_basis)) < 1e-6:
                continue
            if abs(np.linalg.det(C_basis)) < 1e-6:
                continue
            
            A_inv = np.linalg.inv(A_basis)
            B_inv = np.linalg.inv(B_basis)
            C_inv = np.linalg.inv(C_basis)
            
            # Transform the tensor: T' = (A_inv ⊗ B_inv ⊗ C_inv) T
            # T'[a', b', c'] = sum_{a,b,c} A_inv[a',a] * B_inv[b',b] * C_inv[c',c] * T[a,b,c]
            T_prime = np.einsum('ai,bj,ck,ijk->abc', A_inv, B_inv, C_inv, T)
            
            T_prime_flat_U = T_prime.reshape(n * m, -1)
            T_prime_flat_V = T_prime.reshape(m * p, -1)
            T_prime_flat_W = T_prime.reshape(p * n, -1)
            
            # Greedy matching pursuit
            residual = T_prime.copy()
            U_terms = []
            V_terms = []
            W_terms = []
            
            for step in range(target_rank):
                if time.time() > deadline:
                    break
                
                # Find best rank-1 approximation of residual
                # Residual[a,b,c] ≈ u[a]*v[b]*w[c]
                # This is equivalent to finding the best rank-1 approximation
                # via SVD on the unfolded tensor
                
                # Unfold along first mode: R1[a, (b,c)]
                R1 = residual.reshape(n * m, m * p * p * n)
                U_svd, S_svd, Vt_svd = np.linalg.svd(R1, full_matrices=False)
                
                # Take the top singular vector
                if len(S_svd) == 0 or S_svd[0] == 0:
                    break
                    
                u_top = U_svd[:, 0]
                v_top_flat = Vt_svd[0, :]
                v_top = v_top_flat.reshape(m * p, p * n)
                
                # Now factor v_top into v[b]*w[c]
                U_svd2, S_svd2, Vt_svd2 = np.linalg.svd(v_top, full_matrices=False)
                if len(S_svd2) == 0 or S_svd2[0] == 0:
                    break
                    
                v_col = U_svd2[:, 0]
                w_row_flat = Vt_svd2[0, :]
                w_col = w_row_flat.reshape(p * n)
                
                # Normalize: u * v * w should match the residual best
                # Scale to minimize ||residual - u⊗v⊗w||
                # Optimal scaling: alpha = <residual, u⊗v⊗w> / ||u⊗v⊗w||^2
                term = np.einsum('a,b,c->abc', u_top, v_col, w_col)
                denom = np.sum(term * term)
                if denom < 1e-12:
                    break
                alpha = np.sum(residual * term) / denom
                
                U_terms.append(alpha * u_top)
                V_terms.append(v_col)
                W_terms.append(w_col)
                
                residual = residual - alpha * term
                
                if np.max(np.abs(residual)) < 1e-10:
                    break
            
            if len(U_terms) > target_rank:
                continue
                
            if len(U_terms) == 0:
                continue
                
            # Build U, V, W in transformed space
            rank_found = len(U_terms)
            U_prime = np.zeros((n * m, rank_found))
            V_prime = np.zeros((m * p, rank_found))
            W_prime = np.zeros((p * n, rank_found))
            
            for r in range(rank_found):
                U_prime[:, r] = U_terms[r]
                V_prime[:, r] = V_terms[r]
                W_prime[:, r] = W_terms[r]
            
            # Transform back: U = A_basis @ U_prime, V = B_basis @ V_prime, W = C_basis @ W_prime
            U_final = A_basis @ U_prime
            V_final = B_basis @ V_prime
            W_final = C_basis @ W_prime
            
            if verify(U_final, V_final, W_final, atol=1e-6):
                # Polish for higher precision
                polished = polish_solution(U_final, V_final, W_final)
                if polished:
                    return polished
                else:
                    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
            
            # Try ALS refinement from this starting point
            if rank_found <= target_rank and time.time() < deadline - 10:
                U = U_final
                V = V_final
                W = W_final
                for it in range(30):
                    if time.time() > deadline:
                        break
                    
                    B_mat_T = np.zeros((m * p * p * n, rank_found))
                    for r in range(rank_found):
                        B_mat_T[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                    try:
                        P_U = np.linalg.pinv(B_mat_T)
                        U = T_flat_U @ P_U
                    except:
                        break
                    
                    B_mat_T_V = np.zeros((n * m * p * n, rank_found))
                    for r in range(rank_found):
                        B_mat_T_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                    try:
                        P_V = np.linalg.pinv(B_mat_T_V)
                        V = T_flat_V @ P_V
                    except:
                        break
                    
                    B_mat_T_W = np.zeros((n * m * m * p, rank_found))
                    for r in range(rank_found):
                        B_mat_T_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                    try:
                        P_W = np.linalg.pinv(B_mat_T_W)
                        W = T_flat_W @ P_W
                    except:
                        break
                    
                    if verify(U, V, W):
                        polished = polish_solution(U, V, W)
                        if polished:
                            return polished
                        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                    
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(T_approx - T))
                    if err < 1e-4:
                        polished = polish_solution(U, V, W)
                        if polished:
                            return polished
        
        return None

    # Strategy 1: Try greedy matching pursuit for rank 27
    sol_27_mp = greedy_matching_pursuit(27, n_restarts=20)
    if sol_27_mp:
        return sol_27_mp
    
    # Strategy 2: Try ALS for rank 27
    sol_27_als = try_als_for_rank(27, max_iters=60, n_restarts=10)
    if sol_27_als:
        return sol_27_als
    
    # Strategy 3: If time remains, try rank 26
    if time.time() < deadline - 20:
        sol_26_mp = greedy_matching_pursuit(26, n_restarts=10)
        if sol_26_mp:
            return sol_26_mp
        
        sol_26_als = try_als_for_rank(26, max_iters=20, n_restarts=5)
        if sol_26_als:
            return sol_26_als
    
    # Return rank 28
    return get_rank_28_solution()

# EVOLVE-BLOCK-END
