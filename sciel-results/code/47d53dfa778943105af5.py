import numpy as np
import time

def construct():
    start = time.time()
    deadline = 130  # 20s safety margin
    
    n, m, p = 2, 2, 8
    num_U = n * m  # 4
    num_V = m * p  # 16
    num_W = p * n  # 16
    
    # Build target tensor T[a, b, c]
    # T[a, b, c] = 1 if (a//m == c%n) and (a%m == b//p) and (b%p == c//n)
    T = np.zeros((num_U, num_V, num_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((num_U, rank))
        V = np.zeros((num_V, rank))
        W = np.zeros((num_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = U.shape[1]
    best_result = (U, V, W)
    
    # Verify function
    def verify(U, V, W, tol=1e-8):
        rank = U.shape[1]
        # Compute the decomposed tensor
        # T_decomp[a,b,c] = sum_c U[a,c] * V[b,c] * W[c,c]
        # Do this efficiently
        for a in range(num_U):
            for b in range(num_V):
                for c in range(num_W):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - T[a, b, c]) > tol:
                        return False
        return True
    
    # Try to reduce rank via ALS
    def als_reduce(U, V, W, target_rank, max_iter=1000):
        U = U[:, :target_rank].copy()
        V = V[:, :target_rank].copy()
        W = W[:, :target_rank].copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Update U: for each a, U[a, :] = (V^T * W_reshaped) * something
            # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: sum_r U[a,r] * V[b,r] * W[c,r] = T[a, b, c]
            # This is: U[a, :] * (V * W elementwise) = T[a, :, :] reshaped
            # Let M_a[b, c] = T[a, b, c]
            # Let N_a[b, r] = V[b, r], P_a[c, r] = W[c, r]
            # We want: sum_r U[a,r] * N_a[b,r] * P_a[c,r] = M_a[b,c]
            # This is a linear system in U[a, :]
            # M_a = U[a, :]_reshaped * (N_a * P_a) ... 
            # Actually: M_a[b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let Q_a[b, c, r] = V[b,r] * W[c,r]
            # Then M_a[b,c] = sum_r U[a,r] * Q_a[b,c,r]
            # This is a linear system: flatten M_a, and the design matrix has entries Q_a[b,c,r]
            
            # For efficiency, let's do it differently
            # M_a[b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # = (U[a, :] * V^T) [b, r] * W[c, r] ... no
            # Let's define: for each r, the contribution is U[a,r] * outer(V[:,r], W[:,r])
            # So M_a = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # This is a linear combination of matrices outer(V[:,r], W[:,r])
            
            # Solve: M_a = V * diag(U[a,:]) * W^T
            # Where V is 16×rank, W is 16×rank
            # M_a[b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
            
            # Vectorize: vec(M_a) = (W ⊗ V) * U[a, :]
            # Where W ⊗ V is the Kronecker product (16*16) × rank
            
            # Actually: vec(M_a) = (W^T ⊗ V) * vec(diag(U[a,:])) ... no
            # M_a = V * diag(U[a,:]) * W^T
            # vec(M_a) = (W ⊗ V) * U[a, :]  (using column-major vec)
            # In numpy, let's just solve the linear system directly
            
            for a in range(num_U):
                # Build the linear system
                # M_a = T[a, :, :] (16×16)
                # M_a[b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
                # Let x = U[a, :] (rank,)
                # For each (b,c): sum_r V[b,r] * W[c,r] * x[r] = T[a,b,c]
                # Design matrix A_sys: (16*16, rank), A_sys[b*16+c, r] = V[b,r] * W[c,r]
                # RHS: T[a, :, :].flatten()
                
                A_sys = (V[:, :, None] * W[None, :, :]).reshape(-1, target_rank)
                rhs = T[a, :, :].flatten()
                
                # Solve least squares
                try:
                    x, _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
                    U[a, :] = x
                except:
                    pass
            
            # Update V similarly
            for b in range(num_V):
                A_sys = (U[:, :, None] * W[None, :, :]).reshape(-1, target_rank)
                rhs = T[:, b, :].flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
                    V[b, :] = x
                except:
                    pass
            
            # Update W similarly
            for c in range(num_W):
                A_sys = (U[:, :, None] * V[None, :, :]).reshape(-1, target_rank)
                rhs = T[:, :, c].flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
                    W[c, :] = x
                except:
                    pass
            
            # Check convergence
            err = 0
            for a in range(num_U):
                for b in range(num_V):
                    for c in range(num_W):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        err += (val - T[a, b, c])**2
            if err < 1e-12:
                break
        
        return U, V, W
    
    # Try reducing from 32 down
    for target_rank in range(31, 1, -1):
        if time.time() > deadline:
            break
        
        U_try, V_try, W_try = U[:, :target_rank].copy(), V[:, :target_rank].copy(), W[:, :target_rank].copy()
        U_try, V_try, W_try = als_reduce(U_try, V_try, W_try, target_rank, max_iter=500)
        
        if verify(U_try, V_try, W_try):
            best_rank = target_rank
            best_result = (U_try, V_try, W_try)
            break
    else:
        # If no reduction found, try random initialization
        for trial in range(10):
            if time.time() > deadline:
                break
            for target_rank in range(31, 25, -1):
                if time.time() > deadline:
                    break
                U_try = np.random.randn(num_U, target_rank)
                V_try = np.random.randn(num_V, target_rank)
                W_try = np.random.randn(num_W, target_rank)
                U_try, V_try, W_try = als_reduce(U_try, V_try, W_try, target_rank, max_iter=200)
                if verify(U_try, V_try, W_try):
                    best_rank = target_rank
                    best_result = (U_try, V_try, W_try)
                    break
            if best_rank < 32:
                break
    
    U_best, V_best, W_best = best_result
    
    # Round to integers or half-integers
    def round_half(x):
        return np.round(x * 2) / 2
    
    U_rounded = round_half(U_best)
    V_rounded = round_half(V_best)
    W_rounded = round_half(W_best)
    
    if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
        U_best, V_best, W_best = U_rounded, V_rounded, W_rounded
    else:
        # Try integer rounding
        U_int = np.round(U_best)
        V_int = np.round(V_best)
        W_int = np.round(W_best)
        if verify(U_int, V_int, W_int, tol=1e-6):
            U_best, V_best, W_best = U_int, V_int, W_int
    
    # Convert to lists
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
