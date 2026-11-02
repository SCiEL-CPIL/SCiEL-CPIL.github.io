"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """Return the best exact tensor decomposition found within a time budget."""
    start = time.time()
    deadline = start + 130.0  # safety margin below 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 84
    
    # Pre-compute the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] is 1 if i==i', j==j', k==k' else 0
    # Actually, T_{alpha, beta, gamma} = 1 if alpha=(i,j), beta=(j,k), gamma=(i,k)
    # Note: The index of W is k*n+i.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    def _schoolbook():
        """Return the classical rank-nmp decomposition."""
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def _verify_exact(U, V, W, seed=42, trials=3):
        """Verify exact decomposition using multiple random trials."""
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        rng = np.random.RandomState(seed)
        for trial in range(trials):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            C_approx = np.zeros((n, p))
            for r in range(R):
                u_r = Ua[:, r]
                v_r = Va[:, r]
                w_r = Wa[:, r]
                u_scalar = np.dot(u_r, A.flatten())
                v_scalar = np.dot(v_r, B.flatten())
                C_approx += u_scalar * v_scalar * w_r.reshape(p, n).T
            if not np.allclose(C_approx, C_true, atol=1e-9):
                return False
        return True

    def _snap_to_half_integer(arr, tol=1e-3):
        """Snap values to nearest half-integer if close, else keep as is."""
        snapped = np.round(arr * 2) / 2
        # Check if snapping was beneficial (i.e., values were close)
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return arr

    def _als_decompose(R, rng, max_iters=100):
        """
        Use Alternating Least Squares to find a rank-R decomposition.
        Returns (U, V, W) if a valid solution is found, else None.
        """
        # Initialize with random sparse half-integers
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        for r in range(R):
            # Sparse initialization
            nz_u = rng.randint(1, 4)
            nz_v = rng.randint(1, 4)
            nz_w = rng.randint(1, 4)
            
            idx_u = rng.choice(n * m, nz_u, replace=False)
            idx_v = rng.choice(m * p, nz_v, replace=False)
            idx_w = rng.choice(p * n, nz_w, replace=False)
            
            U[idx_u, r] = rng.choice(vals, nz_u)
            V[idx_v, r] = rng.choice(vals, nz_v)
            W[idx_w, r] = rng.choice(vals, nz_w)
            
        # Pre-compute flattened T for faster ALS
        # T shape (n*m, m*p, p*n)
        # For solving U: 
        # T[alpha, beta, gamma] = sum_r U[alpha, r] * V[beta, r] * W[gamma, r]
        # Let M_U[alpha, delta] = T[alpha, beta, gamma] where delta = beta * (p*n) + gamma
        # Let X_U[delta, r] = V[beta, r] * W[gamma, r]
        # Then T_alpha = X_U @ U_alpha^T
        # We can solve U_alpha = (X_U^T X_U)^-1 X_U^T T_alpha
        
        T_flat = T.reshape(n * m, m * p * p * n)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
                
            # Solve for U
            # X_U has shape (m*p*p*n, R)
            # X_U[beta*(p*n)+gamma, r] = V[beta, r] * W[gamma, r]
            # Vectorize: 
            # V is (m*p, R), W is (p*n, R)
            # We need outer product of V and W along the R dimension?
            # V[:, r] is vector of length m*p. W[:, r] is vector of length p*n.
            # X_U[:, r] = np.kron(V[:, r], W[:, r]) ? No.
            # Index delta = beta * (p*n) + gamma.
            # X_U[delta, r] = V[beta, r] * W[gamma, r].
            # This is exactly the Kronecker product of V[:, r] and W[:, r] if we flatten V and W appropriately.
            # Let's construct X_U efficiently.
            # X_U = np.einsum('br,cr->bc r', V, W) ? No, that gives (m*p, p*n, R).
            # We need shape (m*p*p*n, R).
            # X_U_flat = (V[:, :, None] * W[None, :, :]).reshape(m*p*p*n, R)
            X_U = (V[:, :, None] * W[None, :, :]).reshape(m * p * p * n, R)
            
            # Solve T_flat[alpha] = X_U @ U[alpha]
            # T_flat is (n*m, m*p*p*n)
            # U is (n*m, R)
            # We can solve the whole system at once using least squares.
            # T_flat^T = (X_U @ U)^T = U^T @ X_U^T
            # Let A = X_U^T (shape R, m*p*p*n)
            # Let B = T_flat^T (shape m*p*p*n, n*m)
            # We want U^T (n*m, R) such that U^T @ A = B? No.
            # T_flat[alpha, delta] = sum_r U[alpha, r] X_U[delta, r]
            # T_flat = U @ X_U^T
            # So U = T_flat @ pinv(X_U^T)
            # Or use lstsq: X_U @ U^T = T_flat^T
            # U^T = lstsq(X_U, T_flat^T)
            try:
                U_sol, _, _, _ = np.linalg.lstsq(X_U, T_flat.T, rcond=None)
                U = U_sol.T
            except:
                return None

            # Solve for V
            # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
            # Fix U, W. Solve for V.
            # T[beta, alpha, gamma] = sum_r V[beta, r] (U[alpha, r] W[gamma, r])
            # Let X_V[delta, r] = U[alpha, r] * W[gamma, r] where delta = alpha * (p*n) + gamma
            # T_flat_V[beta, delta] = T[alpha, beta, gamma]
            # We need to permute T to get T_flat_V.
            # T shape (n*m, m*p, p*n).
            # T_flat_V shape (m*p, n*m*p*n).
            # T_flat_V[beta, alpha*(p*n)+gamma] = T[alpha, beta, gamma]
            # T_flat_V = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
            T_flat_V = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            X_V = (U[:, :, None] * W[None, :, :]).reshape(n * m * p * n, R)
            
            try:
                V_sol, _, _, _ = np.linalg.lstsq(X_V, T_flat_V.T, rcond=None)
                V = V_sol.T
            except:
                return None

            # Solve for W
            # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
            # Fix U, V. Solve for W.
            # T[gamma, alpha, beta] = sum_r W[gamma, r] (U[alpha, r] V[beta, r])
            # Let X_W[delta, r] = U[alpha, r] * V[beta, r] where delta = alpha * (m*p) + beta
            # T_flat_W[gamma, delta] = T[alpha, beta, gamma]
            # T_flat_W = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
            T_flat_W = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            X_W = (U[:, :, None] * V[None, :, :]).reshape(n * m * m * p, R)
            
            try:
                W_sol, _, _, _ = np.linalg.lstsq(X_W, T_flat_W.T, rcond=None)
                W = W_sol.T
            except:
                return None
            
            # Check convergence / exactness
            # Calculate residual
            # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            # This is expensive. Let's check on a few random samples or just snap and verify.
            if it % 5 == 0:
                U_snapped = _snap_to_half_integer(U)
                V_snapped = _snap_to_half_integer(V)
                W_snapped = _snap_to_half_integer(W)
                
                # Quick check: does the snapped version work?
                # We can compute the tensor from snapped matrices and compare to T
                # T_approx = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
                # if np.allclose(T_approx, T, atol=1e-9):
                #    return U_snapped, V_snapped, W_snapped
                
                # To save time, only verify with random trials if residual is small
                # Let's compute residual norm on a subset
                # Actually, let's just do the full verification every 10 iterations if time permits
                if it % 10 == 0 and time.time() < deadline - 20:
                    U_list = [[float(x) for x in row] for row in U_snapped]
                    V_list = [[float(x) for x in row] for row in V_snapped]
                    W_list = [[float(x) for x in row] for row in W_snapped]
                    if _verify_exact(U_list, V_list, W_list, seed=rng.randint(10000), trials=3):
                        return U_snapped, V_snapped, W_snapped

        return None

    # Start with schoolbook as baseline
    best_result = _schoolbook()
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Lower bound is 28.
    # We will try ranks from 28 upwards.
    # Use multiple restarts for each rank.
    
    for R in range(28, min(best_rank, 40)):
        if time.time() > deadline - 5:
            break
        
        found = False
        # Try a few random initializations
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(R * 1000 + restart * 100 + 42)
            result = _als_decompose(R, rng, max_iters=50)
            
            if result is not None:
                U_arr, V_arr, W_arr = result
                U_list = [[float(x) for x in row] for row in U_arr]
                V_list = [[float(x) for x in row] for row in V_arr]
                W_list = [[float(x) for x in row] for row in W_arr]
                
                # Final verification
                if _verify_exact(U_list, V_list, W_list, seed=12345, trials=5):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = R
                    found = True
                    break
        
        if found:
            break # Found a valid lower rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
