import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Precompute target tensor
    TARGET_TENSOR = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TARGET_TENSOR[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def check_decomposition(U, V, W, T_target):
        """Check if U, V, W form a valid exact decomposition with half-integer entries."""
        U_check = U * 2
        V_check = V * 2
        W_check = W * 2
        
        U_int = np.rint(U_check)
        V_int = np.rint(V_check)
        W_int = np.rint(W_check)
        
        err_U = np.max(np.abs(U_check - U_int))
        err_V = np.max(np.abs(V_check - V_int))
        err_W = np.max(np.abs(W_check - W_int))
        
        if err_U < 1e-5 and err_V < 1e-5 and err_W < 1e-5:
            U_r = U_int / 2.0
            V_r = V_int / 2.0
            W_r = W_int / 2.0
            
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_check, T_target, atol=1e-8):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        return None

    def try_rank(r):
        """Try to find a decomposition of rank r using ALS."""
        if time.time() > deadline:
            return None
            
        num_attempts = 200 if r <= 25 else 100
        found = None
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + r * 1000 + attempt)
            
            # Sparse initialization
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Random sparse init with +/- 1
            density = 0.3
            for _ in range(int(n*m*r*density)):
                i = np.random.randint(n*m)
                j = np.random.randint(r)
                U[i, j] = np.random.choice([-1, 1])
            for _ in range(int(m*p*r*density)):
                i = np.random.randint(m*p)
                j = np.random.randint(r)
                V[i, j] = np.random.choice([-1, 1])
            for _ in range(int(p*n*r*density)):
                i = np.random.randint(p*n)
                j = np.random.randint(r)
                W[i, j] = np.random.choice([-1, 1])
            
            # Add small noise to avoid symmetry breaking issues
            U += np.random.randn(n*m, r) * 0.1
            V += np.random.randn(m*p, r) * 0.1
            W += np.random.randn(p*n, r) * 0.1
            
            # ALS Loop
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # M[b*c, r] = V[b, r] W[c, r]
                M = np.einsum('br,cr->bcr', V, W).reshape(m*p, r)
                # T[a, bc] = U[a, r] M[bc, r]
                # T_a = M^T U_a^T
                # U_a M = T_a^T
                # Solve U_a for each a
                for a in range(n*m):
                    T_a = TARGET_TENSOR[a].reshape(m*p)
                    # U[a, :] M = T_a
                    # M is (m*p, r), U[a] is (r,), T_a is (m*p,)
                    # We want U[a] such that M^T U[a]^T = T_a
                    # Or U[a] M = T_a^T? No.
                    # T_a(bc) = sum_r U[a,r] M[bc,r]
                    # This is T_a = (M^T U[a]^T) ?
                    # Let x = U[a]^T (r x 1).
                    # M^T (r x m*p) * x (r x 1) -> (m*p x 1).
                    # So T_a = M^T x.
                    # x = (M M^T)^{-1} M T_a
                    # Using lstsq: M^T x = T_a
                    # x, _ = np.linalg.lstsq(M.T, T_a, rcond=None)
                    # U[a, :] = x
                    x, _ = np.linalg.lstsq(M.T, T_a, rcond=None)
                    U[a, :] = x

                # Update V
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Fix U, W.
                # T[a, b, c] = V[b, r] (U[a, r] W[c, r])
                # Let N[a*c, r] = U[a, r] W[c, r]
                N = np.einsum('ar,cr->acr', U, W).reshape(n*m, r)
                # T[a, b, c] = V[b, r] N[a, c, r]?
                # T[ab, c] = sum_r V[b, r] (U[a, r] W[c, r])
                # Let's fix b.
                # T[:, b, :] = V[b, r] (U[:, r] W[:, r])
                # T_a_c = sum_r V[b, r] U[a, r] W[c, r]
                # For fixed b, T[:, b, :] is (n*m, p*n).
                # Let P = U * W (elementwise)? No.
                # T[a, c] = sum_r U[a, r] W[c, r] V[b, r]
                # This is T_{a,c} = (U * V_b^T)_{a,r} W_{c,r}?
                # Let's reshape.
                # T[a, c] = sum_r (U[a, r] V[b, r]) W[c, r]
                # Let K[a, r] = U[a, r] V[b, r].
                # Then T[a, c] = sum_r K[a, r] W[c, r].
                # T_vec = W^T K^T?
                # T[a, c] = K[a, :] W[c, :]^T
                # T (n*m x p*n) = K (n*m x r) * W^T (r x p*n)
                # So K = T W^T (W W^T)^{-1}
                # K[a, r] = U[a, r] V[b, r]
                # We need to solve for V[b, r].
                # K = U * V[b]^T (elementwise product of rows of U and V[b])
                # This is not linear in V[b] because U is fixed but K is linear in V[b].
                # K[a, r] = U[a, r] V[b, r].
                # We have T = K W^T.
                # So K = T (W W^T)^{-1} W
                # Let K_target = T (W W^T)^{-1} W
                # Then U[a, r] V[b, r] = K_target[a, r]
                # This is linear in V[b, r] for each r?
                # For each r, V[b, r] * U[:, r] = K_target[:, r]
                # So V[b, r] = <U[:, r], K_target[:, r]> / <U[:, r], U[:, r]>
                
                # Compute K_target
                # T is (n*m, p*n). W is (p*n, r).
                # K_target = T @ W @ inv(W.T @ W) ?
                # T = K W^T => T W = K W W^T => K = T W (W W^T)^{-1}
                WW_t = W.T @ W # r x r
                # K_target = T @ W @ inv(WW_t)
                # T is (nm, pn), W is (pn, r). T@W is (nm, r).
                TW = TARGET_TENSOR.reshape(n*m, -1) @ W
                # K_target = TW @ inv(WW_t)
                try:
                    K_target = TW @ np.linalg.inv(WW_t)
                except np.linalg.LinAlgError:
                    K_target = TW @ np.linalg.pinv(WW_t)
                
                # Solve for V[b]
                # V[b, r] = sum_a U[a, r] K_target[a, r] / sum_a U[a, r]^2
                denom = np.sum(U**2, axis=0) # r
                denom[denom < 1e-10] = 1e-10
                numer = np.sum(U * K_target, axis=0) # r
                V[b, :] = numer / denom

                # Update W
                # Symmetric to V
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Fix U, V.
                # T[a, c] = W[c, r] (U[a, r] V[b, r])
                # Let L[c, r] = W[c, r].
                # T[a, c] = sum_r L[c, r] M[a, r] where M[a, r] = U[a, r] V[b, r]? No, sum over b.
                # T[a, c] = sum_r W[c, r] (sum_b U[a, r] V[b, r])? No.
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
                # For fixed c, T[:, :, c] = W[c, r] (U[:, r] V[:, r])?
                # T[a, b] = sum_r W[c, r] U[a, r] V[b, r]
                # T (nm x mp) = (U * W_c^T) (V * W_c^T)^T?
                # T[a, b] = sum_r (U[a, r] W[c, r]) (V[b, r] W[c, r])
                # This is not linear in W[c] easily.
                # Wait, T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
                # Fix U, V.
                # T[a, b, c] = W[c, r] (U[a, r] V[b, r]).
                # Let Q[c, r] = W[c, r].
                # T[a, b, c] = sum_r Q[c, r] P[a, b, r] where P[a, b, r] = U[a, r] V[b, r].
                # This is linear in Q[c, r].
                # T_vec (nm*mp) = Q[c, :] P_vec?
                # T[a, b, c] = sum_r Q[c, r] P[a, b, r].
                # For fixed c, T[:, :, c] is a matrix (nm x mp).
                # P[:, :, r] is a matrix (nm x mp).
                # T_c = sum_r Q[c, r] P_r.
                # So T_c = P Q[c]^T?
                # T_c (nm x mp) = P (nm x mp x r) contracted with Q[c] (r).
                # This is a linear system for Q[c].
                # Flatten T_c and P_r.
                # T_c_vec = sum_r Q[c, r] P_r_vec.
                # So T_c_vec = P_mat Q[c]^T?
                # P_mat is (nm*mp, r).
                # T_c_vec = P_mat Q[c]^T.
                # Q[c] = (P_mat^T P_mat)^{-1} P_mat^T T_c_vec.
                
                P = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, r)
                T_c = TARGET_TENSOR[:, :, c].reshape(-1)
                # Q[c] is row vector.
                # T_c_vec = P Q[c]^T
                # Q[c]^T = (P^T P)^{-1} P^T T_c_vec
                PP_t = P.T @ P
                try:
                    Q_c, _ = np.linalg.lstsq(P, T_c, rcond=None)
                except:
                    Q_c = np.linalg.pinv(P) @ T_c
                W[c, :] = Q_c

            # After ALS, check for half-integers
            solution = check_decomposition(U, V, W, TARGET_TENSOR)
            if solution is not None:
                return solution

        return None

    # Schoolbook fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Search ranks
    for r in [25, 24, 26]:
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            return result

    return {"U": U_sb, "V": V_sb, "W": W_sb}
