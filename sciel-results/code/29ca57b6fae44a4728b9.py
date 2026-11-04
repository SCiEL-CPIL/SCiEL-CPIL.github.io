"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build target tensor T[i,j,k] = 1 if (i,j,k) is valid else 0
    # T has shape (n*m, m*p, p*n)
    # Index: u = i*m+j, v = j*p+k, w = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    T_flat = T.flatten()
    NM, MP, PN = n*m, m*p, p*n
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-9)
    
    def als_decompose(rank, max_iters=100, tol=1e-10, seed=0):
        """Alternating Least Squares for 3rd order tensor decomposition"""
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n, m, rank) * 0.5
        V = np.random.randn(m, p, rank) * 0.5
        W = np.random.randn(p, n, rank) * 0.5
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current error
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if abs(prev_err - current_err) < tol:
                break
            prev_err = current_err
            
            # Solve for U: fix V, W
            # For each (i,j): T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # This is linear in U[i,j,:]
            for i in range(n):
                for j in range(m):
                    # A[c, k] = V[j,k,c] * W[k,i,c]
                    A = np.einsum('kc,kc->ck', V[j], W[:, i])  # shape (rank, p)
                    b = T[i, j, :]  # shape (p,)
                    # Solve A^T @ u = b
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                    # For fixed j,k: linear in V[j,k,:]
                    # A[c, i] = U[i,j,c] * W[k,i,c]
                    A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])  # shape (rank, n)
                    b = T[:, j, k]  # shape (n,)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W: fix U, V
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                    # For fixed k,i: linear in W[k,i,:]
                    # A[c, j] = U[i,j,c] * V[j,k,c]
                    A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])  # shape (rank, m)
                    b = T[i, :, k]  # shape (m,)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
        
        # Final check
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-8:
            # Try rounding to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if check_exact(U_rounded, V_rounded, W_rounded, rank):
                return U_rounded.flatten(axis=1).tolist(), V_rounded.flatten(axis=1).tolist(), W_rounded.flatten(axis=1).tolist()
            
            # Try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, rank):
                return U_int.flatten(axis=1).tolist(), V_int.flatten(axis=1).tolist(), W_int.flatten(axis=1).tolist()
        
        return None
    
    def try_rank(rank, num_restarts=5):
        """Try to find decomposition of given rank with multiple restarts"""
        for seed in range(num_restarts):
            if time.time() > deadline:
                return None
            result = als_decompose(rank, max_iters=50, tol=1e-10, seed=seed)
            if result is not None:
                return result
        return None
    
    # Search for lower rank, starting from rank_schoolbook-1 down
    best_result = None
    
    for rank in range(rank_schoolbook - 1, 40, -1):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(rank, num_restarts=3)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    # Fallback to schoolbook
    if best_result is None:
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
