import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[a, b, c] where:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T[a, b, c] = 1 if indices match, 0 otherwise
    
    N = n * m
    M = m * p
    P = p * n
    
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_rank = N * M * P # Fallback
    best_U = None
    best_V = None
    best_W = None
    
    # Helper to check if solution is valid
    def check_solution(U, V, W, tol=1e-6):
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Efficient check:
        # Reshape T to (N, M*P) ? No.
        # Use einsum: T_calc[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        # return np.allclose(T_calc, T, atol=tol)
        
        # To save time in the loop, we can just check the error norm.
        # But for final verification, we need exactness or high precision.
        # The problem asks for "integer/half-integer entries". 
        # ALS usually produces floats. We need to round?
        # "exactly decompose" implies we might need to find a decomposition with rational numbers?
        # Or does it accept floats that are close?
        # "score is the rank". "invalid result ... scores 0".
        # Usually, these tasks accept floats with a tolerance, OR they require exact integers.
        # Given "integer/half-integer entries", it strongly suggests we should round to nearest 0.5 and verify.
        
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return np.allclose(T_calc, T, atol=1e-9), U_r, V_r, W_r

    # Search ranks
    # Start from low rank. 3x3x3 is 23. 3x3x4 might be 24, 25...
    # Let's try 23 to 35.
    for R in range(23, 36):
        if time.time() - start_time > timeout:
            break
            
        for attempt in range(5): # Reduced attempts due to faster ALS
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + R * 100 + attempt)
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # Pre-compute reshaped T for faster access
            # T_flat_c: shape (N*M, P) -> T.reshape(N*M, P)
            # T_flat_a: shape (M*P, N) -> T.transpose(1, 2, 0).reshape(M*P, N)
            # T_flat_b: shape (P*N, M) -> T.transpose(2, 0, 1).reshape(P*N, M)
            
            T_c = T.reshape(N * M, P)
            T_a = T.transpose(1, 2, 0).reshape(M * P, N)
            T_b = T.transpose(2, 0, 1).reshape(P * N, M)
            
            converged = False
            for it in range(100): # Max iterations per attempt
                if time.time() - start_time > timeout:
                    break
                
                # Update W
                # We need E_W of shape (N*M, R)
                # E_W[:, r] = vec(U[:, r] * V[:, r]^T)
                # U[:, r] is (N, 1), V[:, r] is (M, 1)
                # Outer product is (N, M). Flatten is (N*M,).
                # We can compute this as:
                # E_W = np.einsum('ar,br->ab r', U, V).reshape(N*M, R)
                
                E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
                # Solve E_W @ W_c^T = T_c[:, c] for each c?
                # Actually, T_c is (N*M, P).
                # We want W (P, R) such that for each c, E_W @ W[c, :]^T = T_c[:, c]
                # This is equivalent to:
                # Let W_T be W.T (R, P).
                # Then E_W @ W_T = T_c.T (R, P)? No.
                # E_W (N*M, R) @ W (P, R).T?
                # Let's check dimensions.
                # We want sum_r E_W[i, r] W[c, r] = T_c[i, c]
                # Let W_T be (R, P) where W_T[r, c] = W[c, r].
                # Then (E_W @ W_T)[i, c] = sum_r E_W[i, r] W_T[r, c] = sum_r E_W[i, r] W[c, r].
                # This matches T_c[i, c].
                # So we solve E_W @ W_T = T_c.T
                # W_T = lstsq(E_W, T_c.T)
                
                try:
                    W_T, residuals, rank, s = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                    W = W_T.T
                except:
                    break
                
                # Update U
                # E_U of shape (M*P, R)
                # E_U[:, r] = vec(V[:, r] * W[:, r]^T)
                # V is (M, R), W is (P, R)
                E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                # Solve E_U @ U_T = T_a.T
                # T_a is (M*P, N). T_a.T is (N, M*P).
                # U_T should be (R, N).
                # (E_U @ U_T)[j, n] = sum_r E_U[j, r] U_T[r, n] = sum_r E_U[j, r] U[n, r].
                # We want sum_r V[j, r] W[k, r] U[i, r] = T_a[j, i] where j=(j,k) in B, i in A?
                # T_a index: first is B index (m*p), second is A index (n*m).
                # T_a[b_idx, a_idx] = T[a_idx, b_idx, c_idx] summed over c? No, T is 0/1.
                # T[a, b, c] = 1 only for specific a,b,c.
                # T_a[b, a] = T[a, b, c] for the specific c?
                # Wait, T is a 3-tensor. T_a is a matricization.
                # T_a[b, a] = sum_c T[a, b, c] ? No, that would lose information.
                # The matricization T_a has entries T_a[b, a] = T[a, b, c] ??
                # No, T is 3D. T_a is (M*P, N).
                # T_a[b_idx, a_idx] = T[a_idx, b_idx, c_idx] is not well-defined unless we fix c.
                # Ah, the matricization T_a is defined such that:
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Fixing b and a, we have T[a, b, :] = sum_r (U[a, r] V[b, r]) W[:, r]
                # This is a vector in c.
                # My previous derivation for W used T_c which is (N*M, P).
                # T_c[i, c] where i = a*M + b?
                # T_c[i, c] = T[a, b, c] with i indexing (a, b).
                # Then E_W[i, r] = U[a, r] V[b, r].
                # Then sum_r E_W[i, r] W[c, r] = T[a, b, c]. Correct.
                
                # Now for U:
                # We want to solve for U.
                # Fix V, W.
                # T[a, b, c] = sum_r U[a, r] (V[b, r] W[c, r])
                # For a fixed a, T[a, :, :] = sum_r U[a, r] (V[:, r] W[:, r]^T)
                # Flatten T[a, :, :] to vector of size M*P.
                # Let T_a_vec[a] = vec(T[a, :, :]).
                # Let E_U be (M*P, R) where E_U[j, r] = V[b, r] W[c, r] with j indexing (b, c).
                # Then T_a_vec[a] = sum_r E_U[:, r] U[a, r] = (E_U.T @ U[a, :])?
                # No, T_a_vec[a, j] = sum_r E_U[j, r] U[a, r].
                # This is (E_U.T @ U.T)[j, a]?
                # Let U_T be (R, N). U_T[r, a] = U[a, r].
                # (E_U.T @ U_T)[j, a] = sum_r E_U.T[j, r] U_T[r, a] = sum_r E_U[r, j] U[a, r].
                # We want this to equal T_a_vec[a, j] = T[a, b, c] where j=(b,c).
                # So we solve E_U.T @ U_T = T_a_vec.T
                # T_a_vec is (N, M*P). T_a_vec.T is (M*P, N).
                
                # Construct T_a_vec:
                # T is (N, M, P).
                # T_a_vec[a, j] where j = b*P + c.
                # T_a_vec = T.reshape(N, M * P)
                
                T_a_vec = T.reshape(N, M * P)
                E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                
                try:
                    U_T, residuals, rank, s = np.linalg.lstsq(E_U.T, T_a_vec.T, rcond=None)
                    U = U_T.T
                except:
                    break

                # Update V
                # Fix U, W.
                # T[a, b, c] = sum_r V[b, r] (U[a, r] W[c, r])
                # For fixed b, T[:, b, :] = sum_r V[b, r] (U[:, r] W[:, r]^T)
                # Flatten T[:, b, :] to vector of size N*P.
                # T_b_vec[b, i] where i = a*P + c.
                # T_b_vec = T.transpose(1, 0, 2).reshape(M, N * P)
                
                T_b_vec = T.transpose(1, 0, 2).reshape(M, N * P)
                E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                
                # T_b_vec[b, i] = sum_r E_V[i, r] V[b, r]
                # (E_V.T @ V.T)[i, b] = sum_r E_V[r, i] V.T[r, b] = sum_r E_V[i, r] V[b, r].
                # So solve E_V.T @ V.T = T_b_vec.T
                
                try:
                    V_T, residuals, rank, s = np.linalg.lstsq(E_V.T, T_b_vec.T, rcond=None)
                    V = V_T.T
                except:
                    break

                # Check convergence
                # Calculate error
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_calc - T)
                
                if error < 1e-8:
                    # Try to round to half-integers
                    valid, U_r, V_r, W_r = check_solution(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                    else:
                        # If not valid after rounding, we might need more iterations or different init
                        # But if error is low, it should be close to integers.
                        # Sometimes we need to push it closer.
                        pass
                        
            if converged:
                break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
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
