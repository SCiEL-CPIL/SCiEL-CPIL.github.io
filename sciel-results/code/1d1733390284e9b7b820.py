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
        U_r = np.array(U).reshape(n, m, rank)
        V_r = np.array(V).reshape(m, p, rank)
        W_r = np.array(W).reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-9)
    
    def als_decompose(rank, max_iters=80, tol=1e-10, seed=0, init=None):
        """Alternating Least Squares for 3rd order tensor decomposition"""
        np.random.seed(seed)
        
        if init is not None:
            U, V, W = init
        else:
            # Initialize with small random values
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
            for i in range(n):
                for j in range(m):
                    # A[c, k] = V[j,k,c] * W[k,i,c]
                    A = np.einsum('kc,kc->ck', V[j], W[:, i])  # shape (rank, p)
                    b = T[i, j, :]  # shape (p,)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
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
        
        if final_err < 1e-6:
            # Try rounding to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if check_exact(U_rounded, V_rounded, W_rounded, rank):
                return (U_rounded.reshape(NM, rank).tolist(), 
                       V_rounded.reshape(MP, rank).tolist(), 
                       W_rounded.reshape(PN, rank).tolist())
            
            # Try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, rank):
                return (U_int.reshape(NM, rank).tolist(), 
                       V_int.reshape(MP, rank).tolist(), 
                       W_int.reshape(PN, rank).tolist())
            
            # Try rounding to nearest 0.25
            U_q = np.round(U * 4) / 4
            V_q = np.round(V * 4) / 4
            W_q = np.round(W * 4) / 4
            if check_exact(U_q, V_q, W_q, rank):
                return (U_q.reshape(NM, rank).tolist(), 
                       V_q.reshape(MP, rank).tolist(), 
                       W_q.reshape(PN, rank).tolist())
        
        return None
    
    def try_rank(rank, num_restarts=10):
        """Try to find decomposition of given rank with multiple restarts"""
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            result = als_decompose(rank, max_iters=60, tol=1e-10, seed=seed)
            if result is not None:
                return result
        
        # Also try with schoolbook-inspired initialization
        # Start from full rank and randomly zero out columns
        for seed in range(5):
            if time.time() > deadline - 5:
                return None
            np.random.seed(seed + 1000)
            # Create a random subset of schoolbook columns
            all_cols = list(range(rank_schoolbook))
            selected = np.random.choice(all_cols, size=min(rank, rank_schoolbook), replace=False)
            U_init = np.zeros((n, m, rank))
            V_init = np.zeros((m, p, rank))
            W_init = np.zeros((p, n, rank))
            for new_col, old_col in enumerate(selected):
                i = old_col // (m * p)
                j = (old_col // p) % m
                k = old_col % p
                U_init[i, j, new_col] = 1
                V_init[j, k, new_col] = 1
                W_init[k, i, new_col] = 1
            
            result = als_decompose(rank, max_iters=60, tol=1e-10, seed=seed, init=(U_init, V_init, W_init))
            if result is not None:
                return result
        
        return None
    
    # Search for lower rank, starting from reasonable lower bound
    # For 3x4x6, known results suggest rank around 50-60
    # Let's search from 55 down to 40
    best_result = None
    
    for rank in range(55, 39, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, num_restarts=8)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    # If no lower rank found, try a few more ranks with more restarts
    if best_result is None:
        for rank in range(60, 54, -1):
            if time.time() > deadline - 10:
                break
            result = try_rank(rank, num_restarts=12)
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
