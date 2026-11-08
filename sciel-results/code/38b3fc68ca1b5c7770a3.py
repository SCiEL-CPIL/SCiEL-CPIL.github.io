import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T as a list of non-zero indices
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    def check_decomposition(U, V, W):
        # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
        # Check if sum_t U[a,t] V[b,t] W[c,t] == 1 for (a,b,c) in target
        # and 0 otherwise? 
        # Actually, the tensor is exactly defined by these 45 terms.
        # We need to ensure that for any (a,b,c), the sum is 1 if in target, 0 otherwise.
        # Since the tensor is sparse, we can just check the target indices.
        # But we also need to ensure no "crosstalk" creates non-zero terms elsewhere.
        # The tensor T is zero everywhere else.
        # So we must check ALL (a,b,c)?
        # dim_u * dim_v * dim_w = 9 * 15 * 15 = 2025.
        # We can construct the full tensor and compare.
        
        r = U.shape[1]
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
        T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
        for a, b, c in target_indices:
            T_target[a, b, c] = 1
            
        if not np.allclose(T_decomp, T_target, atol=1e-9):
            return False
        return True

    def round_to_half_integer(X):
        return np.round(X * 2) / 2.0

    def als_decomposition(rank, max_iters=50, tol=1e-8):
        # Random initialization
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_error = float('inf')
        
        for _ in range(max_iters):
            # Update W
            # T_decomp[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # For fixed a,b, we have sum_t (U[a,t] V[b,t]) W[c,t] = T_target[a,b,c]
            # This is a linear system for W[:, t]? No, W is (dim_w, rank).
            # Let M_ab = U[a,:] * V[b,:] (element-wise product).
            # We want sum_t M_ab[t] W[c,t] = T[a,b,c].
            # This is W^T M_ab^T = T[a,b,:].
            # We can solve for W by stacking all (a,b) pairs.
            # There are dim_u * dim_v pairs.
            # Each gives dim_w equations.
            # Total equations: dim_u * dim_v * dim_w.
            # Unknowns: dim_w * rank.
            
            # Construct A matrix for W
            # A has shape (dim_u * dim_v * dim_w, dim_w * rank)
            # This is too big.
            
            # Better: Use the fact that T is sparse.
            # But ALS usually solves dense least squares.
            # Let's use a simpler approach: Gradient Descent?
            # Or just use the matricization.
            
            # Let's try a different update rule.
            # W[c, t] = (sum_ab T[a,b,c] U[a,t] V[b,t]) / (sum_ab (U[a,t] V[b,t])^2)
            # This is the coordinate descent step for W[c,t].
            
            # Precompute U*V
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :] # (dim_u, dim_v, rank)
            
            # Update W
            # For each c, t:
            # num = sum_ab T[a,b,c] UV[a,b,t]
            # den = sum_ab UV[a,b,t]^2
            # Since T is sparse, we can sum over target indices.
            
            W_new = np.zeros_like(W)
            for c in range(dim_w):
                for t in range(rank):
                    num = 0
                    den = 0
                    for a, b, cc in target_indices:
                        if cc == c:
                            num += UV[a, b, t]
                            den += UV[a, b, t] ** 2
                    if den > 1e-9:
                        W_new[c, t] = num / den
                    else:
                        W_new[c, t] = 0
            W = W_new
            
            # Similarly for U and V
            # This is slow in Python.
            # Let's use NumPy vectorization.
            
            # Flatten target to vector
            # T_flat of length dim_u * dim_v * dim_w
            # This is getting complicated to vectorize the sparse sum efficiently without building the full tensor.
            
            # Given the small size, building the full tensor is fine.
            T_full = np.zeros((dim_u, dim_v, dim_w))
            for a, b, c in target_indices:
                T_full[a, b, c] = 1
            
            # Update W
            # W[c, t] = sum_ab T[a,b,c] U[a,t] V[b,t] / sum_ab (U[a,t] V[b,t])^2
            # Numerator: einsum('abc,at,bt->ct', T_full, U, V)
            # Denominator: einsum('ab,at,bt->t', (U*V).sum(axis=2) ... no)
            # Denominator for W[c,t] is sum_ab (U[a,t] V[b,t])^2.
            # This does not depend on c!
            # Den_t = sum_ab (U[a,t] V[b,t])^2
            
            Num_W = np.einsum('abc,at,bt->ct', T_full, U, V)
            Den_W = np.einsum('ab,at,bt->t', (U[:, :, np.newaxis] * V[np.newaxis, :, :]).sum(axis=(0,1)) ** 2, np.ones((dim_u, dim_v))) # This is wrong
            # Den_t = sum_{a,b} (U[a,t] V[b,t])^2
            Den_W = np.einsum('at,bt->t', U, V) ** 2 # No, this is (sum U V)^2
            # Correct: sum_{a,b} U[a,t]^2 V[b,t]^2
            Den_W = (U**2).sum(axis=0) * (V**2).sum(axis=0)
            
            W = Num_W / Den_W
            
            # Update U
            # U[a, t] = sum_bc T[a,b,c] V[b,t] W[c,t] / sum_bc (V[b,t] W[c,t])^2
            Num_U = np.einsum('abc,bt,ct->at', T_full, V, W)
            Den_U = (V**2).sum(axis=0) * (W**2).sum(axis=0)
            U = Num_U / Den_U
            
            # Update V
            # V[b, t] = sum_ac T[a,b,c] U[a,t] W[c,t] / sum_ac (U[a,t] W[c,t])^2
            Num_V = np.einsum('abc,at,ct->bt', T_full, U, W)
            Den_V = (U**2).sum(axis=0) * (W**2).sum(axis=0)
            V = Num_V / Den_V
            
            # Calculate error
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
            error = np.linalg.norm(T_decomp - T_full)
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
            
        return U, V, W

    best_result = None
    start_time = time.time()
    
    # Try ranks from 30 down to 23
    for rank in range(30, 22, -1):
        if time.time() - start_time > 100:
            break
            
        found = False
        for attempt in range(20):
            U, V, W = als_decomposition(rank, max_iters=20)
            
            # Round to half-integers
            U_r = round_to_half_integer(U)
            V_r = round_to_half_integer(V)
            W_r = round_to_half_integer(W)
            
            if check_decomposition(U_r, V_r, W_r):
                best_result = (U_r, V_r, W_r)
                found = True
                break
        
        if found:
            break

    if best_result:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
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
