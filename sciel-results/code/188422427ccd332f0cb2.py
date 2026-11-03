import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Schoolbook fallback
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
    
    def verify(U, V, W, tol=1e-8):
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T_target, atol=tol)
        
    def snap_to_half_int(x):
        return np.round(x * 2) / 2.0

    def als_step(U, V, W, T, n, m, p, iterations=50, tol=1e-10):
        r = U.shape[1]
        
        for it in range(iterations):
            if time.time() > deadline:
                return False, U, V, W
            
            # Update W
            W_new = np.zeros_like(W)
            for c in range(p * n):
                A = np.einsum('al,bl->abl', U, V).reshape(-1, r)
                b_vec = T[:, :, c].reshape(-1)
                W_c, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                W_new[c, :] = W_c
            W = W_new
            
            # Update V
            V_new = np.zeros_like(V)
            for b in range(m * p):
                A = np.einsum('al,cl->acl', U, W).reshape(-1, r)
                b_vec = T[:, b, :].reshape(-1)
                V_b, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                V_new[b, :] = V_b
            V = V_new
            
            # Update U
            U_new = np.zeros_like(U)
            for a in range(n * m):
                A = np.einsum('bl,cl->bcl', V, W).reshape(-1, r)
                a_vec = T[a, :, :].reshape(-1)
                U_a, residuals, rank, sv = np.linalg.lstsq(A, a_vec, rcond=None)
                U_new[a, :] = U_a
            U = U_new
            
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_comp - T)
            if error < tol:
                return True, U, V, W
                
        return False, U, V, W

    # Try ranks
    # Start from 50, go up to 79
    for r in range(50, rank_school):
        if time.time() > deadline:
            break
        
        found = False
        # Try 3 random restarts
        for trial in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + r * 10 + trial)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            success, U, V, W = als_step(U, V, W, T_target, n, m, p, iterations=100, tol=1e-12)
            
            if success:
                # Try to snap to half-integers
                U_snap = snap_to_half_int(U)
                V_snap = snap_to_half_int(V)
                W_snap = snap_to_half_int(W)
                
                if verify(U_snap, V_snap, W_snap, tol=1e-8):
                    best_rank = r
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    found = True
                    break
                else:
                    # If snapping fails, check if original floats are "close enough" to half-integers?
                    # Or just return floats if they are exact?
                    # The task says "integer/half-integer entries".
                    # If floats are not close to half-integers, this might be invalid.
                    # But if the exact solution is half-integer, ALS should converge to it.
                    pass
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
