import time
import numpy as np

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Construct Schoolbook Decomposition (Rank 42)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook

    # 2. Attempt to find lower rank (41) using ALS
    target_rank = 41
    if time.time() < deadline - 10:
        # Pre-compute target tensor T as a list of vectors for each a
        # T[a, b, c] = 1 if valid, 0 else
        # We need T_a (vector of size m*p) for each a in 0..n*m-1
        # Actually, for ALS, we need T reshaped.
        
        # Let's create the full target tensor T of shape (n*m, m*p, p*n)
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        # Reshape T_target to (n*m, m*p*p*n) for linear algebra?
        # Or just use the structure.
        
        def run_als(U, V, W, T, max_iters=50):
            # T is (A, B, C)
            A, B, C = T.shape
            R = U.shape[1]
            for _ in range(max_iters):
                # Update U
                # For each a, solve D_a @ u_a = t_a
                # D_a[b*C + c, r] = V[b, r] * W[c, r]
                # t_a[b*C + c] = T[a, b, c]
                for a in range(A):
                    # Construct D_a
                    # Vectorize V and W?
                    # D_a is (B*C) x R
                    # D_a[idx, r] = V[b, r] * W[c, r] where idx = b*C + c
                    # We can compute D_a by broadcasting V and W?
                    # V is (B, R), W is (C, R)
                    # We want an array of shape (B, C, R)
                    # V_exp = V[:, np.newaxis, :] # (B, 1, R)
                    # W_exp = W[np.newaxis, :, :] # (1, C, R)
                    # Prod = V_exp * W_exp # (B, C, R)
                    # Reshape to (B*C, R)
                    D_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, R)
                    t_a = T[a].reshape(-1)
                    # Solve D_a @ u_a = t_a
                    # Use least squares
                    u_a, _, _, _ = np.linalg.lstsq(D_a, t_a, rcond=None)
                    U[a, :] = u_a
                    
                # Update V
                for b in range(B):
                    # D_b[a*C + c, r] = U[a, r] * W[c, r]
                    # t_b[a*C + c] = T[a, b, c]
                    # Note: T is (A, B, C). We need to extract T[:, b, :]
                    # Reshape to (A*C)
                    t_b = T[:, b, :].reshape(-1)
                    D_b = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, R)
                    v_b, _, _, _ = np.linalg.lstsq(D_b, t_b, rcond=None)
                    V[b, :] = v_b
                    
                # Update W
                for c in range(C):
                    # D_c[a*B + b, r] = U[a, r] * V[b, r]
                    # t_c[a*B + b] = T[a, b, c]
                    # Extract T[:, :, c]
                    t_c = T[:, :, c].reshape(-1)
                    D_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(-1, R)
                    w_c, _, _, _ = np.linalg.lstsq(D_c, t_c, rcond=None)
                    W[c, :] = w_c
                    
            # Calculate final error
            # T_rec = sum_r U[:, r] V[:, r] W[:, r]
            # Efficiently:
            # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            return np.linalg.norm(T_rec - T)

        # Try a few restarts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            # Scale to avoid huge numbers
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            err = run_als(U_rand, V_rand, W_rand, T_target, max_iters=20)
            
            if err < 1e-8:
                # Check if close to half-integers
                # 2*U, 2*V, 2*W should be close to integers
                U_half = 2 * U_rand
                V_half = 2 * V_rand
                W_half = 2 * W_rand
                
                U_rounded = np.round(U_half)
                V_rounded = np.round(V_half)
                W_rounded = np.round(W_half)
                
                # Check distance to integers
                if np.max(np.abs(U_half - U_rounded)) < 1e-6 and \
                   np.max(np.abs(V_half - V_rounded)) < 1e-6 and \
                   np.max(np.abs(W_half - W_rounded)) < 1e-6:
                   
                    # Convert to half-integers
                    U_final = U_rounded / 2.0
                    V_final = V_rounded / 2.0
                    W_final = W_rounded / 2.0
                    
                    # Verify exactness
                    T_rec = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
                    if np.allclose(T_rec, T_target, atol=1e-9):
                        return {
                            "U": U_final.tolist(),
                            "V": V_final.tolist(),
                            "W": W_final.tolist()
                        }
                        
    return best_result
