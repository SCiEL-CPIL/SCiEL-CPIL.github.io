import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    
    # Schoolbook rank is 63
    # We want to find a rank < 63 decomposition
    # T[i, j, k] = 1 for all i,j,k
    # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    best_rank = n * m * p
    best_factors = None
    
    # Schoolbook fallback
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_fb, V_fb, W_fb, R_fb = schoolbook()
    
    # Helper: compute error
    def compute_error(U, V, W, R):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err

    def verify_exact(U, V, W, R):
        """Check if the decomposition is exact (within tolerance)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def try_snap_half_int(U, V, W, R):
        """Snap to half-integers and verify"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if verify_exact(U_snap, V_snap, W_snap, R):
            return U_snap.copy(), V_snap.copy(), W_snap.copy()
        return None

    def try_snap_int(U, V, W, R):
        """Snap to integers and verify"""
        U_snap = np.round(U)
        V_snap = np.round(V)
        W_snap = np.round(W)
        if verify_exact(U_snap, V_snap, W_snap, R):
            return U_snap.copy(), V_snap.copy(), W_snap.copy()
        return None

    # Strategy 1: Kronecker-product structured ansatz
    # Decompose 3x7 as (3x3) x (3x1) structure? No, 7 is prime.
    # Instead, let's try to construct a low-rank decomposition using a structured approach.
    
    # The tensor is all-ones. This is a very symmetric tensor.
    # Let's try to find a decomposition where the factors have a lot of structure.
    
    # Approach: Randomized ALS with multiple restarts, trying ranks from 50 to 62
    np.random.seed(42)
    
    found = False
    
    # Try ranks from 50 to 62
    for R in range(50, 63):
        if time.time() > deadline:
            break
            
        if found:
            break
            
        # Multiple restarts for each rank
        n_restarts = 10 if R <= 55 else 5
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            if restart % 3 == 0:
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
                W = np.random.randn(p * n, R)
            elif restart % 3 == 1:
                # Sparse initialization
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for r in range(R):
                    idx_u = np.random.randint(0, n*m, 2)
                    idx_v = np.random.randint(0, m*p, 2)
                    idx_w = np.random.randint(0, p*n, 2)
                    U[idx_u, r] = np.random.choice([-1, 1])
                    V[idx_v, r] = np.random.choice([-1, 1])
                    W[idx_w, r] = np.random.choice([-1, 1])
            else:
                # Block-structured initialization
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                # Divide R into blocks
                block_size = R // n
                for i in range(n):
                    start = i * block_size
                    end = start + block_size if i < n-1 else R
                    U[i*m:(i+1)*m, start:end] = np.random.randn(m, end-start)
                    V[:, start:end] = np.random.randn(m*p, end-start)
                    W[:, start:end] = np.random.randn(p*n, end-start)
            
            # ALS iterations
            max_iters = 50
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            MMt = M @ M.T
                            MMt += 1e-10 * np.eye(p)
                            x = np.linalg.solve(MMt, np.ones(p))
                            U[i * m + j, :] = M.T @ x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            MMt = M @ M.T
                            MMt += 1e-10 * np.eye(n)
                            x = np.linalg.solve(MMt, np.ones(n))
                            V[j * p + k, :] = M.T @ x
                        except:
                            pass
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            MMt = M @ M.T
                            MMt += 1e-10 * np.eye(m)
                            x = np.linalg.solve(MMt, np.ones(m))
                            W[k * n + i, :] = M.T @ x
                        except:
                            pass
                
                # Calculate error
                err = compute_error(U, V, W, R)
                
                if err < 1e-10:
                    break
                
                if err > prev_err * 1.1 and it > 10:
                    break
                prev_err = err
            
            # Check for snap-able solution
            result = try_snap_half_int(U, V, W, R)
            if result is None:
                result = try_snap_int(U, V, W, R)
                
            if result is not None:
                U_snap, V_snap, W_snap = result
                best_rank = R
                best_factors = (U_snap, V_snap, W_snap)
                found = True
                break
    
    if best_factors:
        U, V, W = best_factors
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
