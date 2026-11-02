import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second buffer before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Target Tensor Construction
    # A: n x m, B: m x p, C: p x n (Trace convention W[k*n+i,r])
    # T[a, b, c] = 1 if j_a == j_b and i_a == i_c and k_b == k_c
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1
        return T

    T = build_tensor(n, m, p)
    
    best_solution = None
    best_rank = 11
    
    # We strongly suspect rank 10 is achievable.
    target_rank = 10
    max_iters = 300
    tol = 1e-9
    
    # Increase restarts, decrease iterations per restart to allow more diverse starts
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Sometimes structured initialization helps
        if r % 10 == 0:
            # Sparse-ish random init
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
        else:
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
        
        # Normalize to start near unit norm
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        converged = False
        for it in range(max_iters):
            # Update U
            # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a, T[a, :] = sum_r U[a,r] vec(V[:,r] W[:,r])
            X_u = np.zeros((m*p*p*n, target_rank))
            for rr in range(target_rank):
                X_u[:,rr] = np.outer(V[:,rr], W[:,rr]).flatten()
            
            # Solve X_u^T U = T_flat^T ?
            # T.reshape(n*m, -1).T is (m*p*p*n, n*m)
            # We want U (n*m, target_rank)
            # So we solve for U^T (target_rank, n*m)
            try:
                U, _, _, _ = np.linalg.lstsq(X_u, T.reshape(n*m, -1).T, rcond=None)
                U = U.T
            except:
                break
                
            # Update V
            X_v = np.zeros((n*m*p*n, target_rank))
            for rr in range(target_rank):
                X_v[:,rr] = np.outer(U[:,rr], W[:,rr]).flatten()
            
            try:
                V, _, _, _ = np.linalg.lstsq(X_v, T.reshape(m*p, -1).T, rcond=None)
                V = V.T
            except:
                break
                
            # Update W
            X_w = np.zeros((n*m*m*p, target_rank))
            for rr in range(target_rank):
                X_w[:,rr] = np.outer(U[:,rr], V[:,rr]).flatten()
            
            try:
                W, _, _, _ = np.linalg.lstsq(X_w, T.reshape(p*n, -1).T, rcond=None)
                W = W.T
            except:
                break
                
            # Early exit if error is very small
            if it % 20 == 0:
                # Quick error check
                T_rec_approx = 0.0
                # Sample a few entries for speed? No, full check is safer for convergence detection
                # But full check is expensive. Let's do it every 50 iters.
                pass

        # Full Check
        T_rec = np.zeros((n*m, m*p, p*n))
        for rr in range(target_rank):
            T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
        
        err = np.linalg.norm(T - T_rec)
        if err < 1e-6:
            # Try to snap to rational numbers
            # Try denominators 2, 4, 6, 8, 12
            snapped = False
            for denom in [2, 4, 6, 8, 12, 1]:
                U_s = np.round(U * denom) / denom
                V_s = np.round(V * denom) / denom
                W_s = np.round(W * denom) / denom
                
                # Verify exactness with snapped values
                T_rec_exact = np.zeros((n*m, m*p, p*n))
                for rr in range(target_rank):
                    T_rec_exact += np.einsum('a,b,c->abc', U_s[:,rr], V_s[:,rr], W_s[:,rr])
                
                if np.allclose(T_rec_exact, T, atol=1e-9):
                    best_solution = (U_s, V_s, W_s)
                    best_rank = target_rank
                    snapped = True
                    break
            
            if snapped:
                break # Found rank 10

    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return known rank 11 solution
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        # 1. (A00+A11)(B00+B11) -> M1
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # 2. (A10+A11)B00 -> M2
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # 3. A00(B01-B11) -> M3
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # 4. A11(B10-B00) -> M4
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # 5. (A00+A01)B11 -> M5
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # 6. (A10-A00)(B00+B01) -> M6
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # 7. (A01-A11)(B10+B11) -> M7
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # 8. A00 B02 -> M8
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # 9. A01 B12 -> M9
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # 10. A10 B02 -> M10
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # 11. A11 B12 -> M11
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}
