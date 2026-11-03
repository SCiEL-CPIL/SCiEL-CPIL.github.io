import time
import numpy as np
from scipy.optimize import lsq_linear

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n, m, p = 2, 5, 5
    
    # Schoolbook construction
    def build_schoolbook(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Fast validation using random matrices
    def is_valid(U, V, W, tol=1e-8, seed=42):
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        R = U.shape[1]
        A_flat = A.flatten()  # row-major: A[i*m+j]
        B_flat = B.flatten()  # row-major: B[j*p+k]
        
        L = U @ A_flat  # shape (R,)
        M = V @ B_flat  # shape (R,)
        S = L * M       # shape (R,)
        
        # W has shape (p*n, R), with W[k*n+i, r] being the coefficient
        # C_rec[i, k] = sum_r W[k*n+i, r] * S[r]
        # Reshape W.T (R, p*n) to (R, p, n) so W_reshaped[r, k, i] = W[k*n+i, r]
        W_reshaped = W.T.reshape(R, p, n)
        
        # C_rec[i, k] = sum_r W_reshaped[r, k, i] * S[r]
        C_rec = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                C_rec[i, k] = np.dot(S, W_reshaped[:, k, i])
        
        return np.allclose(C, C_rec, atol=tol)

    # ALS Step
    def als_step(U, V, W, R):
        U_new = np.zeros_like(U)
        V_new = np.zeros_like(V)
        W_new = np.zeros_like(W)
        
        # Precompute some structures for efficiency
        # W indexed as W[k*n+i, r]
        # V indexed as V[j*p+k, r]
        # U indexed as U[i*m+j, r]
        
        # Solve for U: for each (a, b), find u_r such that
        # for all k: sum_r W[k*n+a, r] * V[b*p+k, r] * u_r = 1
        for a in range(n):
            for b in range(m):
                A_eq = np.zeros((p, R))
                for k in range(p):
                    A_eq[k, :] = W[k*n+a, :] * V[b*p+k, :]
                b_vec = np.ones(p)
                u_sol, _, _, _ = np.linalg.lstsq(A_eq, b_vec, rcond=None)
                U_new[a*m+b, :] = u_sol
        
        # Solve for V: for each (c, d), find v_r such that
        # for all i: sum_r W[d*n+i, r] * U[i*m+c, r] * v_r = 1
        for c in range(m):
            for d in range(p):
                A_eq = np.zeros((n, R))
                for i in range(n):
                    A_eq[i, :] = W[d*n+i, :] * U[i*m+c, :]
                b_vec = np.ones(n)
                v_sol, _, _, _ = np.linalg.lstsq(A_eq, b_vec, rcond=None)
                V_new[c*p+d, :] = v_sol
        
        # Solve for W: for each (k, i), find w_r such that
        # for all b: sum_r U[i*m+b, r] * V[b*p+k, r] * w_r = 1
        for k in range(p):
            for i in range(n):
                A_eq = np.zeros((m, R))
                for b in range(m):
                    A_eq[b, :] = U[i*m+b, :] * V[b*p+k, :]
                b_vec = np.ones(m)
                w_sol, _, _, _ = np.linalg.lstsq(A_eq, b_vec, rcond=None)
                W_new[k*n+i, :] = w_sol
        
        return U_new, V_new, W_new

    # Try to find a solution of rank R
    def try_rank(R, max_iter=100, n_restarts=10):
        best_result = None
        best_error = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Try different initialization strategies
            if restart == 0:
                # Random init
                scale = 1.0 / np.sqrt(R)
                U = np.random.randn(n*m, R) * scale
                V = np.random.randn(m*p, R) * scale
                W = np.random.randn(p*n, R) * scale
            elif restart == 1:
                # Sparse random init
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
                # Zero out 50% of entries
                U[np.random.rand(*U.shape) < 0.5] = 0
                V[np.random.rand(*V.shape) < 0.5] = 0
                W[np.random.rand(*W.shape) < 0.5] = 0
            elif restart == 2:
                # Structured init based on schoolbook pattern
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                # Place some 1s in a pattern
                for r in range(R):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
            else:
                # More random with different scales
                scale = 0.5 + np.random.rand()
                U = np.random.randn(n*m, R) * scale
                V = np.random.randn(m*p, R) * scale
                W = np.random.randn(p*n, R) * scale
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                U, V, W = als_step(U, V, W, R)
                
                # Check validity with tighter tolerance
                if is_valid(U, V, W, tol=1e-8):
                    return U, V, W
                
                # Track best error for potential return
                # Compute a simple error metric
                np.random.seed(42)
                A = np.random.randn(n, m)
                B = np.random.randn(m, p)
                C = A @ B
                A_flat = A.flatten()
                B_flat = B.flatten()
                L = U @ A_flat
                M = V @ B_flat
                S = L * M
                W_reshaped = W.T.reshape(R, p, n)
                C_rec = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        C_rec[i, k] = np.dot(S, W_reshaped[:, k, i])
                err = np.linalg.norm(C - C_rec)
                if err < best_error:
                    best_error = err
                    best_result = (U.copy(), V.copy(), W.copy())
        
        # If we didn't find a valid solution, return the best we had
        if best_result is not None and best_error < 1e-4:
            return best_result
        return None

    # Start with schoolbook
    best_U, best_V, best_W = build_schoolbook(n*m*p)
    best_rank = n*m*p
    
    # Try lower ranks - prioritize finding any improvement first, then try to go lower
    # Theoretical lower bounds suggest rank >= 40 for 2x5x5
    # Try ranks from 49 down to 40
    for R in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 25:
            break
        
        result = try_rank(R)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = R
            # Continue trying lower ranks if time permits
            # But don't break - keep searching for lower rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
