"""Matrix multiplication 3x3 x 3x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36

    # Precompute target tensor
    # T[a,b,g] = 1 if (a,b,g) is a valid multiplication index
    # a = i*m + j, b = j*p + k, g = k*n + i
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                T_target[a, b, g] = 1.0

    def compute_tensor(U, V, W):
        """Compute T[alpha,beta,gamma] = sum_c U[alpha,c]*V[beta,c]*W[gamma,c]"""
        return np.einsum('ac,bc,dc->abd', U, V, W)

    def snap_to_half_integer(X):
        return np.round(X * 2) / 2

    def check_solution(U, V, W, tol=1e-9):
        U_s = snap_to_half_integer(U)
        V_s = snap_to_half_integer(V)
        W_s = snap_to_half_integer(W)
        T = compute_tensor(U_s, V_s, W_s)
        return np.max(np.abs(T - T_target)) < tol, U_s, V_s, W_s

    def schoolbook():
        rank = rank_schoolbook
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def to_lists(U, V, W):
        U = snap_to_half_integer(np.asarray(U, dtype=float))
        V = snap_to_half_integer(np.asarray(V, dtype=float))
        W = snap_to_half_integer(np.asarray(W, dtype=float))
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build a rank-22 initialization from the 4x4x4 structure
    # The 4x4x4 matrix multiplication tensor has rank 22 (Strassen-like)
    # We'll use a known rank-22 decomposition for 4x4x4 and restrict to 3x3x4
    
    # For 4x4x4, the indices are:
    # U: (4*4, 22) = (16, 22), V: (4*4, 22) = (16, 22), W: (4*4, 22) = (16, 22)
    # We need to extract the 3x3x4 sub-tensor
    
    # A simpler approach: use a structured initialization with half-integer values
    # and use ALS to refine
    
    def generate_structured_init(r, seed=42):
        """Generate a structured initial guess with half-integer values."""
        np.random.seed(seed)
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
        return U, V, W

    def als_optimize(U, V, W, max_iters=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        T_c = T_target.reshape(n*m, m*p, p*n)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update W: for each column c, solve (U[:,c] ⊗ V[:,c])^T W[:,c] = vec(T[:, :, c]_target)
            # Actually: T[a,b,g] = sum_c U[a,c] V[b,c] W[g,c]
            # For fixed U, V, we want to find W such that:
            # sum_c U[a,c] V[b,c] W[g,c] = T_target[a,b,g]
            # This is: for each g, sum_c (U[:,c] * V[:,c])_flat W[g,c] = T_target[:, :, g]_flat
            
            # Let's do it properly with matricization
            # T_mat_c: shape (n*m * m*p, p*n) where T_mat_c[(a,b), g] = T_target[a,b,g]
            T_mat_c = T_target.reshape(n*m * m*p, p*n)
            
            # For W: E_W[:, c] = vec(U[:, c] * V[:, c]) of shape (n*m * m*p,)
            E_W = np.zeros((n*m * m*p, W.shape[1]))
            for c in range(W.shape[1]):
                E_W[:, c] = (U[:, c].reshape(n, m) * V[:, c].reshape(m, p)).flatten()
            
            # Solve E_W @ W.T = T_mat_c.T
            try:
                W_new_T, _, _, _ = np.linalg.lstsq(E_W, T_mat_c.T, rcond=None)
                W_new = W_new_T.T
            except:
                break
            
            # Update U
            T_mat_a = T_target.transpose(1, 2, 0).reshape(m*p * p*n, n*m)
            E_U = np.zeros((m*p * p*n, U.shape[1]))
            for c in range(U.shape[1]):
                E_U[:, c] = (V[:, c].reshape(m, p) * W_new[:, c].reshape(p, n)).flatten()
            
            try:
                U_new_T, _, _, _ = np.linalg.lstsq(E_U, T_mat_a.T, rcond=None)
                U_new = U_new_T.T
            except:
                break
            
            # Update V
            T_mat_b = T_target.transpose(2, 0, 1).reshape(p*n * n*m, m*p)
            E_V = np.zeros((p*n * n*m, V.shape[1]))
            for c in range(V.shape[1]):
                E_V[:, c] = (W_new[:, c].reshape(p, n) * U_new[:, c].reshape(n, m)).flatten()
            
            try:
                V_new_T, _, _, _ = np.linalg.lstsq(E_V, T_mat_b.T, rcond=None)
                V_new = V_new_T.T
            except:
                break
            
            # Check convergence
            diff = max(np.max(np.abs(U_new - U)), np.max(np.abs(V_new - V)), np.max(np.abs(W_new - W)))
            U, V, W = U_new, V_new, W_new
            if diff < tol:
                break
        
        return U, V, W

    def refine_with_als(U, V, W, r):
        """Run ALS and then snap to half-integers, check validity."""
        U, V, W = als_optimize(U, V, W)
        valid, U_s, V_s, W_s = check_solution(U, V, W)
        if valid:
            return to_lists(U_s, V_s, W_s)
        return None

    best_result = schoolbook()
    
    # Try ranks from 22 to 35
    for r in range(22, rank_schoolbook):
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 30 if r <= 24 else 15
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            seed = 1000 + r * 100 + attempt
            U_init, V_init, W_init = generate_structured_init(r, seed=seed)
            
            result = refine_with_als(U_init, V_init, W_init, r)
            if result is not None:
                best_result = result
                found = True
                break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
