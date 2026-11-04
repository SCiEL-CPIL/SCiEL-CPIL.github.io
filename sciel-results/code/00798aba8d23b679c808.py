"""Matrix multiplication 3x3 x 3x4 tensor decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    n, m, p = DIMENSIONS
    # n=3 (rows of A, cols of C), m=3 (cols of A, rows of B), p=4 (cols of B, rows of C? No)
    # Task: Multiply 3x3 by 3x4.
    # A: 3x3. B: 3x4. C: 3x4.
    # Indices:
    # A[i, j] where i in 0..2, j in 0..2
    # B[j, k] where j in 0..2, k in 0..3
    # C[i, k] where i in 0..2, k in 0..3
    # 
    # Contract: sum_j A[i, j] B[j, k] = C[i, k]
    #
    # Tensor T[a, b, g]
    # a corresponds to (i, j) -> index i*m + j. (3*3=9)
    # b corresponds to (j, k) -> index j*p + k. (3*4=12)
    # g corresponds to (i, k) -> index i*p + k. (3*4=12) -- WAIT.
    #
    # Let's check the contract in the prompt:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies g index is related to k and i.
    # Usually, the output index is $(i, k)$.
    # If $g = k \cdot n + i$, then $g$ ranges $0 \dots (p \cdot n - 1) = 11$.
    # This matches the dimension of the output tensor $C$ which is $n \times p$ (3x4).
    #
    # Let's re-verify the dimensions in the code provided:
    # T_target = np.zeros((n*m, m*p, p*n))
    # a: i*m + j. i in 0..n-1, j in 0..m-1.
    # b: j*p + k. j in 0..m-1, k in 0..p-1.
    # g: k*n + i. k in 0..p-1, i in 0..n-1.
    #
    # This matches the standard multiplication tensor structure.
    
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

    # ALS Implementation
    def als_decompose(rank, max_iter=500, tol=1e-8):
        # Random initialization
        # Use small integers to help snapping
        U = np.random.randint(-1, 2, size=(n * m, rank)) * 0.5
        V = np.random.randint(-1, 2, size=(m * p, rank)) * 0.5
        W = np.random.randint(-1, 2, size=(p * n, rank)) * 0.5
        
        # Ensure non-trivial start
        if np.allclose(U, 0): U += np.random.rand(n * m, rank) * 0.1
        if np.allclose(V, 0): V += np.random.rand(m * p, rank) * 0.1
        if np.allclose(W, 0): W += np.random.rand(p * n, rank) * 0.1

        prev_loss = np.inf
        loss = 0.0
        
        for it in range(max_iter):
            # Update U
            # We need to solve for U given V, W
            # T[a,b,g] = sum_c U[a,c] V[b,c] W[g,c]
            # For fixed a, b, g, this is linear in U[a,c].
            # Vectorize: 
            # Let M_U be a matrix where each row corresponds to a specific (a, b, g) triple?
            # No, we can batch the (b,g) pairs for each a.
            
            # Construct P matrix for U update:
            # P[beta, c] = V[b(beta), c] * W[g(beta), c]
            # beta index corresponds to flattened (b, g).
            # b range: m*p = 12. g range: p*n = 12.
            # Total betas: 144.
            
            b_vals = np.arange(m * p)
            g_vals = np.arange(p * n)
            B_grid, G_grid = np.meshgrid(b_vals, g_vals, indexing='ij')
            B_flat = B_grid.flatten()
            G_flat = G_grid.flatten()
            
            # P shape: (144, rank)
            P = V[B_flat, :] * W[G_flat, :]
            
            # T_flat shape: (144, n*m) ? 
            # T_target is (n*m, m*p, p*n).
            # We want T_target[a, B_flat, G_flat] for all a.
            # Reshape T_target to (n*m, 144).
            T_flat = T_target.reshape(n * m, m * p, p * n).reshape(n * m, -1).T
            # T_flat is (144, n*m). Columns are a=0..8.
            
            # We want to solve: min_U sum_a || T_flat[:, a] - P U[:, a] ||^2
            # This is a linear least squares problem: P^T P U = P^T T_flat
            # U shape: (rank, n*m). Columns of U are the updates for each a.
            
            try:
                # P is 144 x rank. T_flat is 144 x (n*m).
                # Solve (P.T @ P) @ U = P.T @ T_flat
                A_mat = P.T @ P
                b_vec = P.T @ T_flat
                # Add small regularization for stability
                A_mat += 1e-9 * np.eye(rank)
                U_new = np.linalg.solve(A_mat, b_vec).T # (n*m, rank)
            except np.linalg.LinAlgError:
                return None, np.inf

            # Update V
            # T[a,b,g] = sum_c U[a,c] V[b,c] W[g,c]
            # For fixed b, linear in V[b,c].
            # Q[a, c] = U[a, c] * W[g, c] ? No, g varies.
            # For fixed b, we sum over a, g.
            # Let's construct Q matrix.
            # Q[alpha, c] = U[a(alpha), c] * W[g(alpha), c]
            # alpha corresponds to flattened (a, g).
            # a range: n*m = 9. g range: p*n = 12.
            # Total alphas: 108.
            
            a_vals = np.arange(n * m)
            G_grid2, A_grid2 = np.meshgrid(g_vals, a_vals, indexing='ij')
            A_flat2 = A_grid2.flatten()
            G_flat2 = G_grid2.flatten()
            
            Q = U[A_flat2, :] * W[G_flat2, :]
            
            # T_target reshape for V:
            # We need T_target[A_flat2, b, G_flat2].
            # This is tricky to extract directly without loops or advanced indexing.
            # Let's use einsum or advanced indexing.
            # T_target shape: (n*m, m*p, p*n).
            # We want a matrix R of shape (108, m*p).
            # R[i, b] = T_target[A_flat2[i], b, G_flat2[i]].
            
            R = np.zeros((len(A_flat2), m * p))
            for idx in range(len(A_flat2)):
                R[idx, :] = T_target[A_flat2[idx], :, G_flat2[idx]]
            
            try:
                A_mat_v = Q.T @ Q
                b_vec_v = Q.T @ R
                A_mat_v += 1e-9 * np.eye(rank)
                V_new = np.linalg.solve(A_mat_v, b_vec_v).T # (m*p, rank)
            except np.linalg.LinAlgError:
                return None, np.inf

            # Update W
            # For fixed g, linear in W[g,c].
            # S[beta, c] = U[a(beta), c] * V[b(beta), c]
            # beta corresponds to flattened (a, b).
            # a range: 9. b range: 12.
            # Total betas: 108.
            
            B_grid3, A_grid3 = np.meshgrid(b_vals, a_vals, indexing='ij')
            A_flat3 = A_grid3.flatten()
            B_flat3 = B_grid3.flatten()
            
            S = U[A_flat3, :] * V[B_flat3, :]
            
            # T_target[A_flat3, B_flat3, g]
            # We want a matrix M of shape (108, p*n).
            # M[i, g] = T_target[A_flat3[i], B_flat3[i], g].
            
            M = np.zeros((len(A_flat3), p * n))
            for idx in range(len(A_flat3)):
                M[idx, :] = T_target[A_flat3[idx], B_flat3[idx], :]
            
            try:
                A_mat_w = S.T @ S
                b_vec_w = S.T @ M
                A_mat_w += 1e-9 * np.eye(rank)
                W_new = np.linalg.solve(A_mat_w, b_vec_w).T # (p*n, rank)
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
            # If snap fails, check if the raw loss is very small (might be numerical)
            if loss < 1e-6:
                 # Try a few more iterations with stricter tolerance? 
                 # Or just return None
                 pass
            return None, loss

    best_result = schoolbook()
    
    # Try ranks from 25 to 35
    for r in range(25, rank_schoolbook):
        if time.time() > deadline:
            break
            
        found = False
        # More restarts for lower ranks
        max_restarts = 30 if r < 30 else 10
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
                
            res, loss = als_decompose(r, max_iter=200)
            
            if res is not None:
                best_result = res
                found = True
                break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
