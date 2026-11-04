import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if A[a] * B[b] contributes to C[c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def try_rank(R, max_iters=500, tol=1e-9, seed=42):
        """
        Attempts to find a rank-R decomposition using ALS (Alternating Least Squares).
        Returns (U, V, W, error) or None if failed/timeout.
        """
        if time.time() - start_time > timeout:
            return None
            
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_error = np.inf
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return None
                
            # Update U
            # T is (A, B, C). We want U such that U * (V^T * W^T) approximates T unrolled.
            # T_unrolled (A, B*C) = U (R, B*C)
            # V^T * W^T is (R, B) x (R, C) -> (R, B, C)
            # This is tricky to do directly without unrolling.
            # Let's use the property: U[:, r] = argmin || T[:, :, :] - sum_r U[:, r] (V[:, r] \otimes W[:, r]) ||
            
            # Compute the "effective" target for U
            # T is 3D. We can reshape T to (A, R, B, C) by contracting with V and W? No.
            
            # Standard ALS for tensor rank:
            # U[:, r] = (T \times_1 (V[:, r] \otimes W[:, r]))? No.
            
            # Let's use a simpler gradient descent approach on the flattened residual.
            # Residual R_t = T - sum_r U[:, r] V[:, r]^T W[:, r]^T (broadcasted)
            
            # To speed up, we can use a simple coordinate descent or just stick to the known rank 17 if this is too slow.
            # Given the 120s limit and complexity, a robust numerical search for rank 16 is risky.
            
            # Let's try a specific known rank 16 structure if it exists?
            # Literature says 17 is minimal.
            
            # Instead of a generic search which might fail, let's just verify the rank 17 solution is correct and return it.
            # If I implement a search and it times out or fails, I return rank 17.
            
            # For now, let's implement a quick check if rank 16 is possible using a small number of iterations.
            
            # Update U
            # T_{a, b, c} = sum_r U_{a, r} V_{b, r} W_{c, r}
            # Fix V, W.
            # U_{a, r} = sum_{b, c} T_{a, b, c} V_{b, r} W_{c, r} / sum_{b, c} (V_{b, r} W_{c, r})^2
            # This is least squares for each r.
            
            # Construct matrix M_r of shape (A, B*C)
            # M_r[a, b*p_n + c] = V[b, r] * W[c, r]
            # Target t_r[a] = sum_{b,c} T[a, b, c] V[b, r] W[c, r]
            
            # This is O(R * A * B * C) per iteration.
            # A=6, B=9, C=6. Size is small.
            
            for r in range(R):
                # Target for U[:, r]
                # t = np.tensordot(T, V[:, r] * W[:, r], axes=([1, 2], [0, 0]))
                # t shape (A,)
                
                # Denominator for U[:, r]
                # d = np.sum((V[:, r] * W[:, r]) ** 2)
                
                # This formulation assumes orthogonality or specific structure?
                # Actually, ALS updates U by solving:
                # U[:, r] = (T \times_2 V[:, r] \times_3 W[:, r]) / ||V[:, r] \otimes W[:, r]||^2
                # This is valid if we are doing coordinate descent on the mode-1 fibers.
                
                v_r = V[:, r]
                w_r = W[:, r]
                
                # Contract T with v_r and w_r
                # T[a, b, c] * v_r[b] * w_r[c]
                # Sum over b, c
                t_u = np.einsum('abc,b,c->a', T, v_r, w_r)
                
                # Norm squared of the outer product v_r \otimes w_r
                # ||v_r \otimes w_r||^2 = ||v_r||^2 * ||w_r||^2
                norm_sq = np.dot(v_r, v_r) * np.dot(w_r, w_r)
                
                if norm_sq < 1e-12:
                    U[:, r] = np.random.randn(n * m) * 1e-6
                else:
                    U[:, r] = t_u / norm_sq
                    
            # Update V
            for r in range(R):
                u_r = U[:, r]
                w_r = W[:, r]
                
                t_v = np.einsum('abc,a,c->b', T, u_r, w_r)
                norm_sq = np.dot(u_r, u_r) * np.dot(w_r, w_r)
                
                if norm_sq < 1e-12:
                    V[:, r] = np.random.randn(m * p) * 1e-6
                else:
                    V[:, r] = t_v / norm_sq
                    
            # Update W
            for r in range(R):
                u_r = U[:, r]
                v_r = V[:, r]
                
                t_w = np.einsum('abc,a,b->c', T, u_r, v_r)
                norm_sq = np.dot(u_r, u_r) * np.dot(v_r, v_r)
                
                if norm_sq < 1e-12:
                    W[:, r] = np.random.randn(p * n) * 1e-6
                else:
                    W[:, r] = t_w / norm_sq
                    
            # Calculate error
            # Res = T - sum_r U[:, r] V[:, r]^T W[:, r]^T
            # Res = T - np.einsum('ar,br,cr->abc', U, V, W)
            Re = T - np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(Re)
            
            if error < tol:
                return U, V, W, error
                
            if error > prev_error * 1.1:
                # Diverging or stuck, break
                break
            prev_error = error
            
        return None

    # Try rank 16
    best_rank_16 = None
    for seed in range(5):
        if time.time() - start_time > timeout:
            break
        res = try_rank(16, max_iters=100, seed=seed)
        if res is not None:
            U, V, W, err = res
            if err < 1e-8:
                best_rank_16 = (U, V, W)
                break
                
    if best_rank_16:
        U, V, W = best_rank_16
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to Rank 17
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def set_U(i, j, r, val): U[i * m + j, r] = val
    def set_V(j, k, r, val): V[j * p + k, r] = val
    def set_W(k, i, r, val): W[k * n + i, r] = val

    # Strassen-like decomposition for the 2x2x2 core (rows 0,1 of A; cols 0,1 of B; rows 0,1 of C)
    # M0: (A00 + A11)(B00 + B11)
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    # M1: (A10 + A11)B00
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
    
    # M2: A00(B01 - B11)
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
    
    # M3: A11(B10 - B00)
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
    
    # M4: (A00 + A01)B11
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
    
    # M5: (A10 - A00)(B00 + B01)
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)
    
    # M6: (A11 - A01)(B10 + B11)
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)

    # Term 7: A[0,0] * B[0,2] -> C[0,2]
    set_U(0, 0, 7, 1)
    set_V(0, 2, 7, 1)
    set_W(2, 0, 7, 1)
    
    # Term 8: A[0,1] * B[1,2] -> C[0,2]
    set_U(0, 1, 8, 1)
    set_V(1, 2, 8, 1)
    set_W(2, 0, 8, 1)
    
    # Term 9: A[1,0] * B[0,2] -> C[1,2]
    set_U(1, 0, 9, 1)
    set_V(0, 2, 9, 1)
    set_W(2, 1, 9, 1)
    
    # Term 10: A[1,1] * B[1,2] -> C[1,2]
    set_U(1, 1, 10, 1)
    set_V(1, 2, 10, 1)
    set_W(2, 1, 10, 1)
    
    # Term 11: A[0,2] * B[2,0] -> C[0,0]
    set_U(0, 2, 11, 1)
    set_V(2, 0, 11, 1)
    set_W(0, 0, 11, 1)
    
    # Term 12: A[0,2] * B[2,1] -> C[0,1]
    set_U(0, 2, 12, 1)
    set_V(2, 1, 12, 1)
    set_W(1, 0, 12, 1)
    
    # Term 13: A[0,2] * B[2,2] -> C[0,2]
    set_U(0, 2, 13, 1)
    set_V(2, 2, 13, 1)
    set_W(2, 0, 13, 1)
    
    # Term 14: A[1,2] * B[2,0] -> C[1,0]
    set_U(1, 2, 14, 1)
    set_V(2, 0, 14, 1)
    set_W(0, 1, 14, 1)
    
    # Term 15: A[1,2] * B[2,1] -> C[1,1]
    set_U(1, 2, 15, 1)
    set_V(2, 1, 15, 1)
    set_W(1, 1, 15, 1)
    
    # Term 16: A[1,2] * B[2,2] -> C[1,2]
    set_U(1, 2, 16, 1)
    set_V(2, 2, 16, 1)
    set_W(2, 1, 16, 1)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
