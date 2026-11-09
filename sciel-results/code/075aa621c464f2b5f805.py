import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Target ranks to try, in increasing order
    target_ranks = list(range(18, 37))
    
    best_result = None
    best_rank = n * m * p  # 36
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = best_rank
    
    # Helper function to check if a decomposition is valid and exact with half-integers
    def check_decomposition(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check if entries are close to half-integers
        # And if the reconstruction is exact
        tol = 1e-6
        # Check half-integer property
        # 2 * entry should be close to integer
        if not np.allclose(2 * U, np.round(2 * U), atol=tol):
            return False
        if not np.allclose(2 * V, np.round(2 * V), atol=tol):
            return False
        if not np.allclose(2 * W, np.round(2 * W), atol=tol):
            return False
            
        # Round to half-integers
        U_r = np.round(2 * U) / 2
        V_r = np.round(2 * V) / 2
        W_r = np.round(2 * W) / 2
        
        # Reconstruct
        T = np.zeros((n, m, p))
        for c in range(rank):
            # U[:, c] is (n*m, 1)
            # V[:, c] is (m*p, 1)
            # W[:, c] is (p*n, 1)
            # We need to compute the outer product sum
            # This is slow in a loop, but rank is small
            # Let's vectorize if possible, or just use loops for clarity and small size
            pass
        
        # Vectorized reconstruction
        # Reshape U to (n, m, rank)
        U_reshaped = U_r.reshape(n, m, rank)
        V_reshaped = V_r.reshape(m, p, rank)
        W_reshaped = W_r.reshape(p, n, rank)
        
        # Sum over rank
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # This is a contraction.
        # We can do this by iterating over c
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            # u_c: (n, m), v_c: (m, p), w_c: (p, n)
            # We need to combine them into (n, m, p)
            # T_c[i, j, k] = u_c[i, j] * v_c[j, k] * w_c[k, i]
            # This is not a standard matrix product.
            # We can compute it by broadcasting
            # u_c[:, :, None] * v_c[:, None, :] * w_c.T[None, :, :] ?
            # u_c[i, j] -> (n, m, 1)
            # v_c[j, k] -> (1, m, p)
            # w_c[k, i] -> (n, 1, p) ? No, w_c is (p, n). w_c[k, i].
            # w_c.T is (n, p). w_c.T[i, k].
            # So w_c.T[None, :, :] is (1, n, p).
            # We need (n, m, p).
            # u_c[:, :, None] * v_c[:, None, :] * w_c.T[None, :, :]
            # Dimensions: (n, m, 1) * (1, m, p) * (1, n, p) -> (n, m, p).
            # Wait, v_c[:, None, :] is (m, 1, p).
            # u_c[:, :, None] is (n, m, 1).
            # Product is (n, m, p).
            # w_c.T[None, :, :] is (1, n, p).
            # Product is (n, m, p).
            # This works.
            term = U_reshaped[:, :, c][:, :, None] * V_reshaped[:, c, :][:, None, :] * W_reshaped[:, c, :].T[None, :, :]
            T_recon += term
            
        # Check if T_recon matches the target tensor
        # Target tensor T[i, j, k] = 1 for all i, j, k?
        # No, the target tensor for matrix multiplication is 1 for all i, j, k?
        # Yes, because C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor T has T[i, j, k] = 1.
        T_target = np.ones((n, m, p))
        
        if not np.allclose(T_recon, T_target, atol=tol):
            return False
            
        return True

    # ALS Solver
    def als_decomposition(rank, max_iter=100, tol=1e-8):
        # Initialize
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Target tensor
        T = np.ones((n, m, p))
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update U
            # Fix V, W
            # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            # For fixed i, j, the equation is:
            # T[i, j, :] = sum_c U[i, j, c] (V[:, c] * W[:, c].T)[:, i]?
            # Let's derive the linear system for U.
            # We want to minimize || T - sum_c U_c V_c W_c ||^2.
            # This is equivalent to solving for each i, j:
            # T[i, j, k] = sum_c U[i, j, c] M_c[i, j, k]
            # where M_c[i, j, k] = V[j, k, c] W[k, i, c].
            # So for each i, j, we have a system:
            # T[i, j, :] = U[i, j, :] @ M_ij
            # where M_ij is (rank, p).
            # M_ij[c, k] = V[j, k, c] W[k, i, c].
            
            # We can solve this for all i, j at once?
            # Let's just loop over i, j since n*m is small.
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Construct M_ij: (rank, p)
                    # M_ij[c, k] = V[j, k, c] * W[k, i, c]
                    # V is (m, p, rank)
                    # W is (p, n, rank)
                    # V[j, :, c] is (p,)
                    # W[:, i, c] is (p,)
                    # M_ij[c, k] = V[j, k, c] * W[k, i, c]
                    # We can compute this as:
                    # v_j = V[j, :, :] # (p, rank)
                    # w_i = W[:, i, :] # (p, rank)
                    # M_ij = (v_j * w_i).T # (rank, p)
                    v_j = V[j, :, :]  # (p, rank)
                    w_i = W[:, i, :]  # (p, rank)
                    M_ij = (v_j * w_i).T  # (rank, p)
                    
                    # Solve U[i, j, :] @ M_ij = T[i, j, :]
                    # U[i, j, :] is (rank,)
                    # M_ij is (rank, p)
                    # T[i, j, :] is (p,)
                    # This is an underdetermined system if rank > p.
                    # We want the minimum norm solution? Or just least squares?
                    # Since we want to match T exactly, and rank is likely >= p,
                    # we can use least squares.
                    # Actually, we want to find U such that U @ M_ij = T[i, j, :]
                    # If M_ij has full column rank, we can solve.
                    # Use np.linalg.lstsq
                    sol, residuals, rank_m, s = np.linalg.lstsq(M_ij.T, T[i, j, :], rcond=None)
                    # sol is (rank,)
                    U_new[i * m + j, :] = sol
                    
            U = U_new
            
            # Update V
            # Fix U, W
            # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            # For fixed j, k:
            # T[:, j, k] = V[j, k, :] @ N_jk
            # where N_jk[c, i] = U[i, j, c] W[k, i, c]
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # N_jk[c, i] = U[i, j, c] * W[k, i, c]
                    # U[:, j, c] is (n,)
                    # W[k, :, c] is (n,)
                    # N_jk = (U[:, j, :] * W[k, :, :]).T # (rank, n)
                    u_j = U[:, j, :]  # (n, rank)
                    w_k = W[k, :, :]  # (n, rank)
                    N_jk = (u_j * w_k).T  # (rank, n)
                    
                    # Solve V[j, k, :] @ N_jk = T[:, j, k]
                    # V[j, k, :] is (rank,)
                    # N_jk is (rank, n)
                    # T[:, j, k] is (n,)
                    sol, residuals, rank_n, s = np.linalg.lstsq(N_jk.T, T[:, j, k], rcond=None)
                    V_new[j * p + k, :] = sol
                    
            V = V_new
            
            # Update W
            # Fix U, V
            # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            # For fixed k, i:
            # T[i, :, k] = W[k, i, :] @ P_ki
            # where P_ki[c, j] = U[i, j, c] V[j, k, c]
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # P_ki[c, j] = U[i, j, c] * V[j, k, c]
                    # U[i, :, c] is (m,)
                    # V[:, k, c] is (m,)
                    # P_ki = (U[i, :, :] * V[:, k, :]).T # (rank, m)
                    u_i = U[i, :, :]  # (m, rank)
                    v_k = V[:, k, :]  # (m, rank)
                    P_ki = (u_i * v_k).T  # (rank, m)
                    
                    # Solve W[k, i, :] @ P_ki = T[i, :, k]
                    # W[k, i, :] is (rank,)
                    # P_ki is (rank, m)
                    # T[i, :, k] is (m,)
                    sol, residuals, rank_p, s = np.linalg.lstsq(P_ki.T, T[i, :, k], rcond=None)
                    W_new[k * n + i, :] = sol
                    
            W = W_new
            
            # Check convergence
            # Compute error
            T_recon = np.zeros((n, m, p))
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            for c in range(rank):
                term = U_reshaped[:, :, c][:, :, None] * V_reshaped[:, c, :][:, None, :] * W_reshaped[:, c, :].T[None, :, :]
                T_recon += term
            err = np.linalg.norm(T_recon - T)
            if err < tol:
                break
                
        return U, V, W

    # Try to find a decomposition for each rank
    for r in target_ranks:
        if time.time() > deadline:
            break
            
        if r >= best_rank:
            break
            
        # Try multiple random restarts
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            U, V, W = als_decomposition(r, max_iter=50)
            
            if check_decomposition(U, V, W, r):
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
                
    return best_result
