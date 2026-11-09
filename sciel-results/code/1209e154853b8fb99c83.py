import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    # Schoolbook result
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    def verify_decomposition(U, V, W, T, atol=1e-9):
        """Check if U, V, W exactly decompose T."""
        R = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_recon, T, atol=atol), T_recon

    def als_decompose(T, R, n, m, p, U_init=None, V_init=None, W_init=None, max_iter=50, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        if U_init is not None:
            U = U_init.copy()
        else:
            U = np.random.randn(n * m, R)
            U /= np.linalg.norm(U, axis=0) + 1e-12
        
        if V_init is not None:
            V = V_init.copy()
        else:
            V = np.random.randn(m * p, R)
            V /= np.linalg.norm(V, axis=0) + 1e-12
        
        if W_init is not None:
            W = W_init.copy()
        else:
            W = np.random.randn(p * n, R)
            W /= np.linalg.norm(W, axis=0) + 1e-12
        
        prev_err = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for U
            for i in range(n):
                M_i = np.zeros((m * p, R))
                b_i = np.zeros(m * p)
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        M_i[idx, :] = V[j * p + k, :] * W[k * n + i, :]
                        b_i[idx] = T[i, j, k]
                U[i * m : (i + 1) * m, :], _ = np.linalg.lstsq(M_i, b_i, rcond=None)
                
            # Solve for V
            for j in range(m):
                M_j = np.zeros((n * p, R))
                b_j = np.zeros(n * p)
                for i in range(n):
                    for k in range(p):
                        idx = i * p + k
                        M_j[idx, :] = U[i * m + j, :] * W[k * n + i, :]
                        b_j[idx] = T[i, j, k]
                V[j * p : (j + 1) * p, :], _ = np.linalg.lstsq(M_j, b_j, rcond=None)
                
            # Solve for W
            for k in range(p):
                M_k = np.zeros((n * m, R))
                b_k = np.zeros(n * m)
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        M_k[idx, :] = U[i * m + j, :] * V[j * p + k, :]
                        b_k[idx] = T[i, j, k]
                W[k * n : (k + 1) * n, :], _ = np.linalg.lstsq(M_k, b_k, rcond=None)
            
            # Calculate error
            T_recon = np.zeros((n, m, p))
            for r in range(R):
                T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T - T_recon)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    def try_round_and_verify(U, V, W, T):
        """Try rounding to integers and half-integers, verify exactness."""
        R = U.shape[1]
        
        # Try rounding to nearest integer
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        valid, _ = verify_decomposition(U_int, V_int, W_int, T)
        if valid:
            return U_int, V_int, W_int, 'int'
        
        # Try rounding to half-integers
        U_half = np.round(2 * U) / 2
        V_half = np.round(2 * V) / 2
        W_half = np.round(2 * W) / 2
        valid, _ = verify_decomposition(U_half, V_half, W_half, T)
        if valid:
            return U_half, V_half, W_half, 'half'
        
        return None, None, None, None

    # Try ranks from 83 down to 60
    for R in range(rank_school - 1, max(50, rank_school - 35), -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random restarts
        for trial in range(8):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + R * 100 + trial)
            U, V, W, err = als_decompose(T, R, n, m, p, max_iter=30)
            
            if err < 1e-6:
                # Try to find integer/half-integer solution
                U_int, V_int, W_int, typ = try_round_and_verify(U, V, W, T)
                if U_int is not None:
                    best_rank = R
                    best_U, best_V, best_W = U_int, V_int, W_int
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
