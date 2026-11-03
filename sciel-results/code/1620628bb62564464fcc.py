import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Build target tensor T[a,b,c]
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 36)
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=int)
    V_fb = np.zeros((m*p, rank_fb), dtype=int)
    W_fb = np.zeros((p*n, rank_fb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_fb, V_fb, W_fb, rank_fb)
    best_rank = rank_fb
    
    # Helper function to check if a decomposition is exact
    def check_exact(U, V, W, T):
        # U, V, W are integer or half-integer arrays
        # Compute T_decomp
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        return np.allclose(T_decomp, T, atol=1e-10)
    
    # Helper function to snap to integers
    def snap_and_check(U, V, W, T):
        # Try integer snapping
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        if check_exact(U_int, V_int, W_int, T):
            return U_int, V_int, W_int
        
        # Try half-integer snapping (scale by 2)
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        # Check if they are actually half-integers (i.e., 2*U is integer)
        if np.allclose(U * 2, np.round(U * 2), atol=1e-6) and \
           np.allclose(V * 2, np.round(V * 2), atol=1e-6) and \
           np.allclose(W * 2, np.round(W * 2), atol=1e-6):
            if check_exact(U_half, V_half, W_half, T):
                return U_half, V_half, W_half
                
        return None

    # ALS function
    def run_als(U, V, W, T, iters=50):
        n_rows_U = U.shape[0]
        n_rows_V = V.shape[0]
        n_rows_W = W.shape[0]
        rank = U.shape[1]
        
        # Reshape T for ALS
        # T is (n*m, m*p, p*n)
        # For U: T_mat is (n*m, m*p * p*n)
        T_U = T.reshape(n_rows_U, n_rows_V * n_rows_W)
        # For V: T_mat is (m*p, p*n * n*m)
        T_V = T.transpose(1, 2, 0).reshape(n_rows_V, n_rows_W * n_rows_U)
        # For W: T_mat is (p*n, n*m * m*p)
        T_W = T.transpose(2, 0, 1).reshape(n_rows_W, n_rows_U * n_rows_V)
        
        for _ in range(iters):
            # Update U
            # A_U is (m*p * p*n, rank) where A_U[bc, t] = V[b,t] * W[c,t]
            A_U = np.einsum('bt,ct->bct', V, W).reshape(-1, rank)
            # Solve U @ A_U.T = T_U  => U = T_U @ inv(A_U.T @ A_U) @ A_U.T
            # Use least squares
            try:
                U = np.linalg.lstsq(A_U, T_U.T, rcond=None)[0].T
            except:
                pass
                
            # Update V
            # A_V is (p*n * n*m, rank) where A_V[ca, t] = W[c,t] * U[a,t]
            A_V = np.einsum('ct,at->cat', W, U).reshape(-1, rank)
            try:
                V = np.linalg.lstsq(A_V, T_V.T, rcond=None)[0].T
            except:
                pass
                
            # Update W
            # A_W is (n*m * m*p, rank) where A_W[ab, t] = U[a,t] * V[b,t]
            A_W = np.einsum('at,bt->abt', U, V).reshape(-1, rank)
            try:
                W = np.linalg.lstsq(A_W, T_W.T, rcond=None)[0].T
            except:
                pass
                
        # Calculate final error
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        error = np.linalg.norm(T_decomp - T)
        return U, V, W, error

    # Search for lower ranks
    # We prioritize ranks close to 36 first, as they are more likely to exist if any do.
    # Then go down.
    ranks_to_try = list(range(35, 19, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
            
        remaining = deadline - 10 - time.time()
        if remaining < 5:
            break
            
        # Number of restarts depends on remaining time
        # Each restart takes ~0.1-0.5s depending on rank and iterations
        attempts = int(remaining / 0.2)
        attempts = max(10, min(attempts, 200))
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
                
            # Initialize randomly
            # Scale matters. Start with small values.
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Run ALS
            U, V, W, error = run_als(U, V, W, T, iters=30)
            
            if error < 1e-4:
                # Try to snap
                result = snap_and_check(U, V, W, T)
                if result:
                    U_best, V_best, W_best = result
                    best_rank = rank
                    best_result = (U_best, V_best, W_best, rank)
                    break # Found a solution for this rank, go to next lower rank
    
    U_best, V_best, W_best, _ = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
