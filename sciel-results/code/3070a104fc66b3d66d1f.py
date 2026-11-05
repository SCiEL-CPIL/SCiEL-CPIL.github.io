"""Matrix multiplication 3x3 x 3x4 tensor decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    n, m, p = 3, 3, 4
    rank_schoolbook = n * m * p  # 36

    # Precompute T_target
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                T_target[a, b, g] = 1.0

    # Precompute index grids for ALS to speed up iterations
    # U update: beta = (b, g)
    b_vals = np.arange(m * p)
    g_vals = np.arange(p * n)
    B_grid, G_grid = np.meshgrid(b_vals, g_vals, indexing='ij')
    B_flat_ug = B_grid.flatten()
    G_flat_ug = G_grid.flatten()
    
    # T_flat for U: shape (m*p*p*n, n*m) -> (144, 9)
    # T_target[a, b, g]. We want T_target[:, B_flat, G_flat]
    # Using advanced indexing:
    a_indices = np.arange(n*m)
    T_flat_u = T_target[a_indices[:, np.newaxis], B_flat_ug[np.newaxis, :], G_flat_ug[np.newaxis, :]].T
    # T_flat_u[beta, a] = T_target[a, B_flat_ug[beta], G_flat_ug[beta]]

    # V update: alpha = (a, g)
    A_grid_v, G_grid_v = np.meshgrid(a_indices, g_vals, indexing='ij')
    A_flat_v = A_grid_v.flatten()
    G_flat_v = G_grid_v.flatten()
    
    # T_flat for V: shape (n*m*p*n, m*p) -> (108, 12)
    # T_target[a, b, g]. We want T_target[A_flat, :, G_flat]
    b_indices_v = np.arange(m*p)
    T_flat_v = T_target[A_flat_v[:, np.newaxis], b_indices_v[np.newaxis, :], G_flat_v[:, np.newaxis]].T
    # T_flat_v[alpha, b] = T_target[A_flat_v[alpha], b, G_flat_v[alpha]]

    # W update: beta = (a, b)
    A_grid_w, B_grid_w = np.meshgrid(a_indices, b_vals, indexing='ij')
    A_flat_w = A_grid_w.flatten()
    B_flat_w = B_grid_w.flatten()
    
    # T_flat for W: shape (n*m*m*p, p*n) -> (108, 12)
    # T_target[a, b, g]. We want T_target[A_flat, B_flat, :]
    g_indices_w = np.arange(p*n)
    T_flat_w = T_target[A_flat_w[:, np.newaxis], B_flat_w[:, np.newaxis], g_indices_w[np.newaxis, :]].T
    # T_flat_w[beta, g] = T_target[A_flat_w[beta], B_flat_w[beta], g]

    def compute_tensor(U, V, W):
        """Compute T[alpha,beta,gamma] = sum_c U[alpha,c]*V[beta,c]*W[gamma,c]"""
        return np.einsum('ac,bc,dc->abd', U, V, W)

    def snap_and_check(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T = compute_tensor(U_s, V_s, W_s)
        
        # Check max error
        if np.max(np.abs(T - T_target)) > 1e-9:
            return False
        return True

    def to_lists(U, V, W):
        U = np.round(np.asarray(U, dtype=float) * 2) / 2
        V = np.round(np.asarray(V, dtype=float) * 2) / 2
        W = np.round(np.asarray(W, dtype=float) * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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

    # ALS Implementation with Precomputed Indices
    def als_decompose(rank, max_iter=1000, tol=1e-10):
        # Random initialization with small integers
        np.random.seed(int(time.time() * 1000) % 10000) # Vary seed slightly per call? No, use global seed or fixed for reproducibility?
        # Let's use a fixed seed for the first few attempts to be deterministic if possible, or just random.
        U = np.random.randint(-2, 3, size=(n * m, rank)) * 0.5
        V = np.random.randint(-2, 3, size=(m * p, rank)) * 0.5
        W = np.random.randint(-2, 3, size=(p * n, rank)) * 0.5
        
        if np.allclose(U, 0): U += np.random.rand(n * m, rank) * 0.1
        if np.allclose(V, 0): V += np.random.rand(m * p, rank) * 0.1
        if np.allclose(W, 0): W += np.random.rand(p * n, rank) * 0.1

        prev_loss = np.inf
        loss = 0.0
        
        for it in range(max_iter):
            # Update U
            # P[beta, c] = V[B_flat_ug[beta], c] * W[G_flat_ug[beta], c]
            P = V[B_flat_ug, :] * W[G_flat_ug, :]
            
            # Solve P^T P U = P^T T_flat_u
            # U is (n*m, rank). We solve for U^T (rank, n*m)
            A_mat = P.T @ P
            b_vec = P.T @ T_flat_u
            A_mat += 1e-9 * np.eye(rank)
            try:
                U_new = np.linalg.solve(A_mat, b_vec).T
            except np.linalg.LinAlgError:
                return None, np.inf

            # Update V
            # Q[alpha, c] = U[A_flat_v[alpha], c] * W[G_flat_v[alpha], c]
            Q = U[A_flat_v, :] * W[G_flat_v, :]
            
            # Solve Q^T Q V = Q^T T_flat_v
            A_mat_v = Q.T @ Q
            b_vec_v = Q.T @ T_flat_v
            A_mat_v += 1e-9 * np.eye(rank)
            try:
                V_new = np.linalg.solve(A_mat_v, b_vec_v).T
            except np.linalg.LinAlgError:
                return None, np.inf

            # Update W
            # S[beta, c] = U[A_flat_w[beta], c] * V[B_flat_w[beta], c]
            S = U[A_flat_w, :] * V[B_flat_w, :]
            
            # Solve S^T S W = S^T T_flat_w
            A_mat_w = S.T @ S
            b_vec_w = S.T @ T_flat_w
            A_mat_w += 1e-9 * np.eye(rank)
            try:
                W_new = np.linalg.solve(A_mat_w, b_vec_w).T
            except np.linalg.LinAlgError:
                return None, np.inf

            U, V, W = U_new, V_new, W_new
            
            # Calculate loss
            T = compute_tensor(U, V, W)
            loss = np.sum((T - T_target) ** 2)
            
            if np.abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
            
        # Snap and check
        if snap_and_check(U, V, W):
            return to_lists(U, V, W), loss
        else:
            return None, loss

    best_result = schoolbook()
    
    # Try ranks from 26 to 35
    # Prioritize lower ranks
    for r in range(26, rank_schoolbook):
        if time.time() > deadline:
            break
            
        found = False
        # More restarts for lower ranks
        max_restarts = 50 if r < 30 else 20
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
                
            res, loss = als_decompose(r, max_iter=300)
            
            if res is not None:
                best_result = res
                found = True
                break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
