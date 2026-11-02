import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Build the target tensor T as a sparse structure or dense array
    # T is of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
                
    # Function to verify a decomposition
    def verify(U, V, W):
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        R = U.shape[1]
        # Reconstruct T
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can do this efficiently using einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # Check if T_recon matches T
        # Since T is sparse, we only need to check the non-zero entries
        # And ensure that entries where T is 0 are also 0 in T_recon
        # Actually, T has 1s at specific spots and 0s elsewhere.
        # So we check np.allclose(T_recon, T)
        if not np.allclose(T_recon, T, atol=1e-9):
            return False
        # Check if entries are integer/half-integer
        # The problem says "integer/half-integer entries"
        # We can just return the float values, the evaluator likely handles the type
        # But we should try to snap to integers if close
        return True

    # Search for lower rank
    # We will try ranks from rank_schoolbook - 1 down to some limit
    # Time limit guard
    start_time = time.time()
    time_limit = 100  # seconds, keep margin
    
    # Try to find a decomposition of rank R
    for R in range(rank_schoolbook - 1, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Multiple restarts for each rank
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W randomly
            np.random.seed(42 + restart + R)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Iterative optimization
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                # Fix V, W -> Solve for U
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed a, T[a, :, :] = sum_r U[a, r] (V[:, r] W[:, r])
                # This is a linear system for U[a, :]
                # Let M_b_c = V[:, r] W[:, r]
                # We want U[a, :] such that U[a, :] @ M = T[a, :, :]
                # M is R x (m*p * p*n)
                # T[a, :, :] is (m*p * p*n)
                
                # Efficiently:
                # Reshape V to (m*p, R) and W to (p*n, R)
                # The term for a fixed a is:
                # sum_r U[a, r] V[:, r] W[:, r]
                # This is equivalent to:
                # U[a, :] @ (V.T * W.T) ? No.
                # Let's use einsum for the residual and gradient, or just solve linear systems.
                
                # Linear system for U:
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Let M_r = V[:, r] * W[:, r] (outer product? No, elementwise? No)
                # V[:, r] is a vector of length m*p.
                # W[:, r] is a vector of length p*n.
                # The product V[b, r] W[c, r] is the (b, c) entry of the outer product V[:, r] W[:, r].T?
                # Yes, outer product.
                # So T[a, :, :] = sum_r U[a, r] (V[:, r] outer W[:, r])
                # Let M be a matrix of shape (R, m*p, p*n) where M[r, b, c] = V[b, r] W[c, r]
                # Then T[a, :, :] = sum_r U[a, r] M[r, :, :]
                # This is a linear system: U[a, :] @ M_flat = T[a, :, :].flat
                # M_flat is (R, m*p*p*n)
                
                # This is too large for direct solving if R is large, but R is small (<60).
                # m*p = 20, p*n = 15. Size 300.
                # M_flat is R x 300.
                # T_flat is 300.
                # We solve U[a, :] (size R) from U[a, :] @ M_flat = T_flat.
                # This is an underdetermined system if R < 300.
                # We use least squares.
                
                M_flat = np.einsum('br,cr->bcr', V, W).reshape(R, -1)
                T_flat = T.reshape(-1)
                
                # Solve for U
                # U is (n*m, R)
                # We need to solve for each row a.
                # U[a, :] @ M_flat = T_flat
                # This is (R x 1) @ (R x 300) = 300 ? No.
                # U[a, :] is (1, R). M_flat is (R, 300).
                # (1, R) @ (R, 300) = (1, 300).
                # So we solve X @ M_flat = T_flat where X is (1, R).
                # This is equivalent to M_flat.T @ X.T = T_flat.T
                # M_flat.T is (300, R). X.T is (R, 1).
                # We use np.linalg.lstsq(M_flat.T, T_flat.T)
                
                # Do this for all a
                U_new = np.zeros((n*m, R))
                for a in range(n*m):
                    # M_flat is same for all a
                    # T_flat is same for all a? No, T[a, :, :] is different.
                    T_a_flat = T[a, :, :].reshape(-1)
                    sol, residuals, rank, sv = np.linalg.lstsq(M_flat.T, T_a_flat, rcond=None)
                    U_new[a, :] = sol
                    
                U = U_new
                
                # Similarly for V
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Fix U, W. Solve for V.
                # T[:, b, c] = sum_r U[:, r] V[b, r] W[c, r]
                # Let N_r = U[:, r] * W[:, r] (outer product)
                # N is (R, n*m, p*n)
                # T[:, b, c] = sum_r V[b, r] N[r, :, c] ?
                # T[a, b, c] = sum_r V[b, r] (U[a, r] W[c, r])
                # Let M2_flat be (R, n*m * p*n) where M2_flat[r, a, c] = U[a, r] W[c, r]
                # T[:, b, :] = sum_r V[b, r] M2_flat[r, :, :]
                # V[b, :] @ M2_flat = T[:, b, :].flat
                M2_flat = np.einsum('ar,cr->acr', U, W).reshape(R, -1)
                V_new = np.zeros((m*p, R))
                for b in range(m*p):
                    T_b_flat = T[:, b, :].reshape(-1)
                    sol, residuals, rank, sv = np.linalg.lstsq(M2_flat.T, T_b_flat, rcond=None)
                    V_new[b, :] = sol
                V = V_new
                
                # Similarly for W
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Fix U, V. Solve for W.
                # T[a, b, c] = sum_r W[c, r] (U[a, r] V[b, r])
                # Let M3_flat be (R, n*m * m*p) where M3_flat[r, a, b] = U[a, r] V[b, r]
                # T[:, :, c] = sum_r W[c, r] M3_flat[r, :, :]
                M3_flat = np.einsum('ar,br->abr', U, V).reshape(R, -1)
                W_new = np.zeros((p*n, R))
                for c in range(p*n):
                    T_c_flat = T[:, :, c].reshape(-1)
                    sol, residuals, rank, sv = np.linalg.lstsq(M3_flat.T, T_c_flat, rcond=None)
                    W_new[c, :] = sol
                W = W_new
                
            # Check if the solution is valid
            if verify(U, V, W):
                # Snap to integers if close
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                if verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # If no lower rank found, return schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
