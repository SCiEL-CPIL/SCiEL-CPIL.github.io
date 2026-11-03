import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T
    # T[w, u, v] = 1 if w = k*n + i, u = i*m + j, v = j*p + k
    # u in [0, n*m), v in [0, m*p), w in [0, p*n)
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[w, u, v] = 1

    # Store schoolbook solution as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()

    start_time = time.time()
    time_limit = 115  # Leave 5s margin for safety

    # We want to find rank < 90.
    # Try ranks from 89 down to 85. 
    # If we find one, we return it.
    # If not, we return schoolbook.
    
    # To make the search effective, we can try to construct solutions based on known low-rank blocks.
    # E.g., M_{3,5,6} can be split into two M_{3,5,3} if we can find low rank M_{3,5,3}.
    # Or just random search.
    
    # Randomized Search Strategy
    # 1. Pick a rank R.
    # 2. Initialize U, V with random integers in {-1, 0, 1} or similar.
    # 3. Optimize W for fixed U, V (Linear least squares).
    # 4. Optimize U for fixed V, W (Linear least squares).
    # 5. Optimize V for fixed U, W (Linear least squares).
    # 6. Repeat a few times.
    # 7. Round U, V, W to half-integers.
    # 8. Verify exactness.
    
    def verify(U, V, W, T):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if sum_r W[:, r] * (U[:, r] outer V[:, r]) == T
        # T is (p*n, n*m, m*p)
        R = U.shape[1]
        recon = np.zeros_like(T)
        for r in range(R):
            # Term is W[:, r] * (U[:, r] outer V[:, r])
            # W[:, r] is (p*n,)
            # U[:, r] outer V[:, r] is (n*m, m*p)
            # Outer product of W[:, r] and (U[:, r] outer V[:, r])?
            # No, the term is sum_r W[w, r] * U[u, r] * V[v, r]
            # So recon[w, u, v] = sum_r W[w, r] U[u, r] V[v, r]
            # This is an Einstein summation: 'wr,ur,vr -> wuv'
            # Or simply:
            # For each r, add W[:, r][:, None, None] * U[:, r][None, :, None] * V[:, r][None, None, :]
            # Wait, shapes:
            # W[:, r] is (p*n,) -> reshape (p*n, 1, 1)
            # U[:, r] is (n*m,) -> reshape (1, n*m, 1)
            # V[:, r] is (m*p,) -> reshape (1, 1, m*p)
            recon += W[:, r][:, None, None] * U[:, r][None, :, None] * V[:, r][None, None, :]
        
        return np.allclose(recon, T, atol=1e-9)

    def find_solution_for_rank(R, attempts=10):
        for attempt in range(attempts):
            if time.time() - start_time > time_limit:
                return None
            
            # Initialize with random integers
            # Using small integers helps rounding
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            W = np.zeros((p * n, R))
            
            # ALS iterations
            for it in range(50):
                if time.time() - start_time > time_limit:
                    return None
                
                # Update W: For each w, T[w] = sum_r W[w, r] U[:, r] V[:, r]^T
                # vec(T[w]) = sum_r W[w, r] vec(U[:, r] V[:, r]^T)
                # Let A_w be matrix with columns vec(U[:, r] V[:, r]^T)
                # Then vec(T[w]) = A_w W[w]
                # A is same for all w if U, V fixed.
                # A shape: (n*m * m*p, R)
                A = np.zeros((n * m * m * p, R))
                for r in range(R):
                    A[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                # Solve for W[w] for all w
                # T_flat is (p*n, n*m*m*p)
                T_flat = T.reshape(p * n, -1)
                # W = T_flat @ A^+ (pseudo inverse)
                # Since R is small, we can use lstsq
                try:
                    W, _, _, _ = np.linalg.lstsq(A, T_flat.T, rcond=None)
                    W = W.T
                except np.linalg.LinAlgError:
                    return None
                    
                # Update U: For each u, T[:, u, :] = sum_r U[u, r] W[:, r] V[:, r]^T?
                # T[w, u, v] = sum_r U[u, r] V[v, r] W[w, r]
                # Fix u. We want to find U[u, :] such that 
                # T[:, u, :] (which is (p*n, m*p)) is approx sum_r U[u, r] (W[:, r] outer V[:, r])
                # Let B_u be matrix with columns (W[:, r] outer V[:, r]).flatten()
                # Then vec(T[:, u, :]) = B_u U[u]
                # B shape: (p*n * m*p, R)
                # This B depends on W, V.
                # We can precompute B for all u? No, B is same for all u?
                # Wait, the term is W[:, r] outer V[:, r]. This does not depend on u.
                # So B is the same for all u!
                # Let's call it B_mat.
                B_mat = np.zeros((p * n * m * p, R))
                for r in range(R):
                    B_mat[:, r] = np.outer(W[:, r], V[:, r]).flatten()
                
                # Now for each u, solve B_mat U[u] = vec(T[:, u, :])
                # T_u_flat is (p*n * m*p,)
                # We have n*m such systems.
                # We can stack them: T_stack is (n*m, p*n*m*p)
                # U is (n*m, R)
                # T_stack = U @ B_mat.T ?
                # vec(T[:, u, :]) = sum_r U[u, r] B_mat[:, r]
                # So T_stack = U @ B_mat.T
                # U = T_stack @ (B_mat.T)^+
                T_stack = np.zeros((n * m, p * n * m * p))
                for u in range(n * m):
                    T_stack[u] = T[:, u, :].flatten()
                
                try:
                    U, _, _, _ = np.linalg.lstsq(B_mat.T, T_stack.T, rcond=None)
                    U = U.T
                except np.linalg.LinAlgError:
                    return None

                # Update V: Symmetric to U.
                # T[w, u, v] = sum_r U[u, r] V[v, r] W[w, r]
                # Fix v. Find V[v, :].
                # T[:, :, v] (p*n, n*m) = sum_r V[v, r] (W[:, r] outer U[:, r])
                # C_mat columns: (W[:, r] outer U[:, r]).flatten()
                C_mat = np.zeros((p * n * n * m, R))
                for r in range(R):
                    C_mat[:, r] = np.outer(W[:, r], U[:, r]).flatten()
                
                T_stack_v = np.zeros((m * p, p * n * n * m))
                for v in range(m * p):
                    T_stack_v[v] = T[:, :, v].flatten()
                
                try:
                    V, _, _, _ = np.linalg.lstsq(C_mat.T, T_stack_v.T, rcond=None)
                    V = V.T
                except np.linalg.LinAlgError:
                    return None
            
            # Round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Verify
            if verify(U_round, V_round, W_round, T):
                return U_round, V_round, W_round
        
        return None

    # Try to find lower rank
    # Start from 89 down to 80
    for R in range(rank_schoolbook - 1, max(70, rank_schoolbook - 20), -1):
        if time.time() - start_time > time_limit:
            break
        
        sol = find_solution_for_rank(R, attempts=20)
        if sol is not None:
            U_found, V_found, W_found = sol
            best_rank = R
            best_U = U_found.tolist()
            best_V = V_found.tolist()
            best_W = W_found.tolist()
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
