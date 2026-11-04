import time
import numpy as np
import sys

def construct():
    """
    Attempt to find a rank-48 decomposition for 2x4 * 4x8 using random initialization
    and iterative refinement (approximating tensor rank minimization).
    Falls back to rank-56 Kronecker product (Strassen x Schoolbook) if time runs out
    or if the decomposition is invalid.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = 2, 4, 8
    target_rank = 48
    
    # Target tensor T[i, j, k] = delta_{j, k} in flattened indices?
    # T is a tensor of size (n*m) x (m*p) x (p*n)
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # T_{a, b, c} = sum_r U[a, r] V[b, r] W[c, r]
    #
    # Matrix multiplication tensor T_M:
    # A is n x m, B is m x p, C is n x p
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # Indices:
    # A index a = i * m + j
    # B index b = j * p + k
    # C index c = k * n + i  (Trace convention: W[k*n+i])
    
    # We want to find U, V, W such that for all i, j, k:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] == 1
    # And 0 otherwise.
    
    # Let's define the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-5):
        # Check if U, V, W decompose T
        # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can check this by computing the "slice" for a fixed c?
        # Or just check a few random indices?
        # Better: compute the full tensor and compare?
        # n*m = 8, m*p = 32, p*n = 16. Total size 8*32*16 = 4096.
        # R = 48.
        # Computation: 4096 * 48 ops. Very fast.
        R = U.shape[1]
        T_approx = np.zeros_like(T)
        # Use einsum or loops?
        # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_approx, T, atol=tol, rtol=tol)

    def fallback_rank_56():
        # Construct the known rank-56 decomposition
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Strassen factors for 2x2x2
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))

        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1

        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1

        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1

        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1

        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1

        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1

        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1

        # Schoolbook factors for 1x2x4
        U_b = np.zeros((2, 8))
        V_b = np.zeros((8, 8))
        W_b = np.zeros((4, 8))

        col = 0
        for j in range(2):
            for k in range(4):
                U_b[j, col] = 1
                V_b[j * 4 + k, col] = 1
                W_b[k, col] = 1
                col += 1

        for s in range(7):
            for b in range(8):
                col = s * 8 + b

                for i in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            val_s = U_s[i * 2 + j1, s]
                            val_b = U_b[j2, b]
                            if val_s != 0 and val_b != 0:
                                global_idx = i * 4 + (2 * j1 + j2)
                                U[global_idx, col] += val_s * val_b

                for r1 in range(2):
                    for r2 in range(2):
                        for c1 in range(2):
                            for c2 in range(4):
                                val_s = V_s[r1 * 2 + c1, s]
                                val_b = V_b[r2 * 4 + c2, b]
                                if val_s != 0 and val_b != 0:
                                    global_idx = (2 * r1 + r2) * 8 + (4 * c1 + c2)
                                    V[global_idx, col] += val_s * val_b

                for i in range(2):
                    for c1 in range(2):
                        for c2 in range(4):
                            val_s = W_s[c1 * 2 + i, s]
                            val_b = W_b[c2, b]
                            if val_s != 0 and val_b != 0:
                                global_k = 4 * c1 + c2
                                global_idx = global_k * 2 + i
                                W[global_idx, col] += val_s * val_b

        # Verify fallback
        if not verify(U, V, W, tol=1e-10):
             # Should not happen, but just in case, return schoolbook 64
             rank_fb = n * m * p
             U_fb = np.zeros((n * m, rank_fb))
             V_fb = np.zeros((m * p, rank_fb))
             W_fb = np.zeros((p * n, rank_fb))
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U_fb[i * m + j, col] = 1.0
                         V_fb[j * p + k, col] = 1.0
                         W_fb[k * n + i, col] = 1.0
                         col += 1
             return U_fb, V_fb, W_fb

        return U, V, W

    # Attempt Rank 48
    # Strategy: Random initialization + Gradient Descent on L2 error
    # Error E = sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
    # This is non-convex.
    
    # A better heuristic for tensor decomposition:
    # Initialize with small random values.
    # Iterate:
    #   Fix V, W -> Solve for U (linear least squares)
    #   Fix U, W -> Solve for V
    #   Fix U, V -> Solve for W
    # This is the "Tensor Power Iteration" or similar.
    
    # Let's try this.
    # Reshape T into a matrix for least squares.
    # T is (8, 32, 16).
    # If we fix V (32, 48) and W (16, 48).
    # Let M_r be the outer product of V[:, r] and W[:, r]?
    # T[a, b, c] = sum_r U[a, r] (V[b, r] W[c, r])
    # For a fixed a, T[a, :, :] = sum_r U[a, r] (V[:, r] W[:, r].T)
    # Let Z_r = V[:, r] W[:, r].T (size 32x16)
    # Then T[a] (flattened 32*16) = sum_r U[a, r] Z_r.flatten()
    # So we can solve for U[a, :] using least squares.
    
    rng = np.random.default_rng(42)
    
    U = rng.standard_normal((n*m, target_rank))
    V = rng.standard_normal((m*p, target_rank))
    W = rng.standard_normal((p*n, target_rank))
    
    # Scale down to avoid divergence
    U /= np.sqrt(target_rank)
    V /= np.sqrt(target_rank)
    W /= np.sqrt(target_rank)
    
    # Flatten T for least squares
    # T is (n*m, m*p, p*n)
    # T_mat is (n*m, m*p * p*n)
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # Precompute Z_r matrices?
    # Z_r = outer(V[:, r], W[:, r]) -> (m*p, p*n)
    # Flatten Z_r -> (m*p * p*n)
    # Z_mat is (target_rank, m*p * p*n)
    # Z_mat[r, :] = (V[:, r] * W[:, r, None]).flatten() ? No.
    # Z_r[b, c] = V[b, r] * W[c, r]
    
    max_iters = 1000
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # 1. Update U
        # T_mat[a] = sum_r U[a, r] * Z_r_flat
        # Z_mat[r, :] = (V[:, r] * W[:, r].T).flatten()
        # Note: V[:, r] is (m*p,), W[:, r] is (p*n,)
        # Outer product: V[:, r, None] * W[:, r, None].T -> (m*p, p*n)
        # Flatten: (m*p * p*n)
        
        # Compute Z_mat
        # V is (m*p, R), W is (p*n, R)
        # Z_flat[r, b, c] = V[b, r] * W[c, r]
        # Z_mat = (V * W.T).reshape(target_rank, -1) ?
        # V * W.T is (m*p, p*n, R) ? No.
        # V is (M, R), W is (P, R)
        # We want Z_r[b, c] = V[b, r] W[c, r]
        # Let's construct Z_mat (R, M*P)
        # Z_mat[r, b*M_p + c] = V[b, r] * W[c, r]
        
        # Efficiently:
        # V_expanded = V[:, :, None] * W[:, None, :] ? No.
        # V is (M, R), W is (P, R)
        # Z = V[:, :, None] * W[:, None, :] -> (M, P, R)
        # We want to transpose to (R, M, P) then flatten to (R, M*P)
        Z = V[:, :, None] * W[:, None, :] # (M, P, R)
        Z = np.transpose(Z, (2, 0, 1)) # (R, M, P)
        Z_mat = Z.reshape(target_rank, -1) # (R, M*P)
        
        # Solve T_mat = U @ Z_mat.T
        # T_mat (A, M*P) = U (A, R) @ Z_mat.T (R, M*P)
        # U = T_mat @ pinv(Z_mat.T)
        # Or solve linear system: Z_mat.T @ U.T = T_mat.T
        # U.T (R, A)
        
        # Use np.linalg.lstsq
        try:
            U_new, residuals, rank_u, s = np.linalg.lstsq(Z_mat.T, T_mat.T, rcond=None)
            U = U_new.T
        except np.linalg.LinAlgError:
            break
            
        # 2. Update V
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For fixed b: T[:, b, :] = sum_r V[b, r] (U[:, r] W[:, r].T)
        # Let Y_r = U[:, r] W[:, r].T (n*m, p*n)
        # T[:, b, :].flatten() = sum_r V[b, r] Y_r.flatten()
        # Y_mat (R, A*P)
        
        Y = U[:, :, None] * W[:, None, :] # (A, P, R)
        Y = np.transpose(Y, (2, 0, 1)) # (R, A, P)
        Y_mat = Y.reshape(target_rank, -1)
        
        # T[:, b, :] flattened is (A*P)
        # We need to form T_mat_V (B, A*P) where T_mat_V[b, :] = T[:, b, :].flatten()
        # T is (A, B, P)
        # T_mat_V = np.transpose(T, (1, 0, 2)).reshape(m*p, -1)
        
        T_mat_V = np.transpose(T, (1, 0, 2)).reshape(m*p, -1)
        
        try:
            V_new, residuals, rank_v, s = np.linalg.lstsq(Y_mat.T, T_mat_V.T, rcond=None)
            V = V_new.T
        except np.linalg.LinAlgError:
            break
            
        # 3. Update W
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For fixed c: T[:, :, c] = sum_r W[c, r] (U[:, r] V[:, r].T)
        # Let X_r = U[:, r] V[:, r].T (n*m, m*p)
        # X_mat (R, A*B)
        
        X = U[:, :, None] * V[:, None, :] # (A, B, R)
        X = np.transpose(X, (2, 0, 1)) # (R, A, B)
        X_mat = X.reshape(target_rank, -1)
        
        T_mat_W = np.transpose(T, (2, 0, 1)).reshape(p*n, -1)
        
        try:
            W_new, residuals, rank_w, s = np.linalg.lstsq(X_mat.T, T_mat_W.T, rcond=None)
            W = W_new.T
        except np.linalg.LinAlgError:
            break
        
        # Check convergence
        if it % 10 == 0:
            if verify(U, V, W, tol=1e-4):
                # Try to polish with higher precision
                # Run a few more iterations with tighter tolerance?
                # Or just accept?
                # Let's try to clean up small values
                U[np.abs(U) < 1e-6] = 0
                V[np.abs(V) < 1e-6] = 0
                W[np.abs(W) < 1e-6] = 0
                
                if verify(U, V, W, tol=1e-5):
                    # Success!
                    return U, V, W

    # Fallback
    U, V, W = fallback_rank_56()
    
    # Final check
    if not verify(U, V, W, tol=1e-10):
        # Extreme fallback: Schoolbook 64
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return U_fb, V_fb, W_fb

    return U, V, W

# Wrapper to match interface
def construct():
    U, V, W = _construct_impl()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# Rename to avoid conflict or just use the name directly
def _construct_impl():
    return construct_impl()

# Actually, let's just put the logic in construct() directly to avoid confusion.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 8
    target_rank = 48
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-5):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_approx, T, atol=tol, rtol=tol)

    def fallback_rank_56():
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))

        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1

        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1

        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1

        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1

        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1

        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1

        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1

        U_b = np.zeros((2, 8))
        V_b = np.zeros((8, 8))
        W_b = np.zeros((4, 8))

        col = 0
        for j in range(2):
            for k in range(4):
                U_b[j, col] = 1
                V_b[j * 4 + k, col] = 1
                W_b[k, col] = 1
                col += 1

        for s in range(7):
            for b in range(8):
                col = s * 8 + b

                for i in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            val_s = U_s[i * 2 + j1, s]
                            val_b = U_b[j2, b]
                            if val_s != 0 and val_b != 0:
                                global_idx = i * 4 + (2 * j1 + j2)
                                U[global_idx, col] += val_s * val_b

                for r1 in range(2):
                    for r2 in range(2):
                        for c1 in range(2):
                            for c2 in range(4):
                                val_s = V_s[r1 * 2 + c1, s]
                                val_b = V_b[r2 * 4 + c2, b]
                                if val_s != 0 and val_b != 0:
                                    global_idx = (2 * r1 + r2) * 8 + (4 * c1 + c2)
                                    V[global_idx, col] += val_s * val_b

                for i in range(2):
                    for c1 in range(2):
                        for c2 in range(4):
                            val_s = W_s[c1 * 2 + i, s]
                            val_b = W_b[c2, b]
                            if val_s != 0 and val_b != 0:
                                global_k = 4 * c1 + c2
                                global_idx = global_k * 2 + i
                                W[global_idx, col] += val_s * val_b

        if not verify(U, V, W, tol=1e-10):
             rank_fb = n * m * p
             U_fb = np.zeros((n * m, rank_fb))
             V_fb = np.zeros((m * p, rank_fb))
             W_fb = np.zeros((p * n, rank_fb))
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U_fb[i * m + j, col] = 1.0
                         V_fb[j * p + k, col] = 1.0
                         W_fb[k * n + i, col] = 1.0
                         col += 1
             return U_fb, V_fb, W_fb

        return U, V, W

    rng = np.random.default_rng(42)
    
    U = rng.standard_normal((n*m, target_rank))
    V = rng.standard_normal((m*p, target_rank))
    W = rng.standard_normal((p*n, target_rank))
    
    U /= np.sqrt(target_rank)
    V /= np.sqrt(target_rank)
    W /= np.sqrt(target_rank)
    
    T_mat = T.reshape(n*m, m*p * p*n)
    
    max_iters = 2000
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        Z = V[:, :, None] * W[:, None, :]
        Z = np.transpose(Z, (2, 0, 1))
        Z_mat = Z.reshape(target_rank, -1)
        
        try:
            U_new, residuals, rank_u, s = np.linalg.lstsq(Z_mat.T, T_mat.T, rcond=None)
            U = U_new.T
        except np.linalg.LinAlgError:
            break
            
        Y = U[:, :, None] * W[:, None, :]
        Y = np.transpose(Y, (2, 0, 1))
        Y_mat = Y.reshape(target_rank, -1)
        
        T_mat_V = np.transpose(T, (1, 0, 2)).reshape(m*p, -1)
        
        try:
            V_new, residuals, rank_v, s = np.linalg.lstsq(Y_mat.T, T_mat_V.T, rcond=None)
            V = V_new.T
        except np.linalg.LinAlgError:
            break
            
        X = U[:, :, None] * V[:, None, :]
        X = np.transpose(X, (2, 0, 1))
        X_mat = X.reshape(target_rank, -1)
        
        T_mat_W = np.transpose(T, (2, 0, 1)).reshape(p*n, -1)
        
        try:
            W_new, residuals, rank_w, s = np.linalg.lstsq(X_mat.T, T_mat_W.T, rcond=None)
            W = W_new.T
        except np.linalg.LinAlgError:
            break
        
        if it % 20 == 0:
            if verify(U, V, W, tol=1e-4):
                U[np.abs(U) < 1e-6] = 0
                V[np.abs(V) < 1e-6] = 0
                W[np.abs(W) < 1e-6] = 0
                
                if verify(U, V, W, tol=1e-5):
                    return U, V, W

    U, V, W = fallback_rank_56()
    
    if not verify(U, V, W, tol=1e-10):
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return U_fb, V_fb, W_fb

    return U, V, W
