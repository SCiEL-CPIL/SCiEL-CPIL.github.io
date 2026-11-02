import numpy as np
import itertools
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target Tensor Shape: (n*m, m*p, p*n)
    # Indexing:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # Build the target tensor T
    # T is sparse, only 1s where indices are consistent
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Try ranks from a lower bound up to n*m*p
    # A known lower bound for 2x5x5 might be significantly less than 50.
    # We try ranks in increasing order to find the minimal one.
    # However, ALS often finds local minima. We might need to retry with different seeds or start from higher rank and project?
    # For simplicity and robustness within time limits, we'll try ranks from 20 to 50.
    # If we find one with error ~0, we snap to half-integers.
    
    def is_valid_decomposition(U, V, W, T):
        # Calculate tensor product sum
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Sum over R: U[..., r] * V[..., r] * W[..., r]
        # Using einsum: 'ar, br, cr -> abc'
        try:
            decomposed = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(decomposed - T)
            return error, decomposed
        except Exception:
            return float('inf'), None

    def snap_to_half(x):
        return np.round(x * 2.0) / 2.0

    # Try different ranks
    # We expect the rank to be lower than 50.
    # Let's try a range.
    for R in range(15, 50):
        # Try multiple initializations for each rank
        for _ in range(5):
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            # ALS iterations
            iters = 50
            for it in range(iters):
                # Fix V, W, update U
                # U = (T x V x W) / (V x W)
                # Efficient update using einsum
                # U_new[a, r] = sum_b,c T[a,b,c] V[b,r] W[c,r] / sum_b,c V[b,r] W[c,r]
                # denom = np.einsum('br,cr->bc', V, W) -> sum over b,c? No, sum over b,c for each r?
                # We need U[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / (sum_{b,c} V[b,r] * W[c,r])
                # Let M_r[b,c] = V[b,r] * W[c,r]
                # num = sum_{b,c} T[a,b,c] M_r[b,c]
                # den = sum_{b,c} M_r[b,c]
                
                # Vectorized approach:
                # Compute V W^T ? No, V is (mp, R), W is (pn, R).
                # We need for each r, the outer product V[:, r] and W[:, r].
                # Sum over r is implicit in the update?
                
                # Update U:
                # For each r, let v_r = V[:, r], w_r = W[:, r]
                # numerator for column r of U: einsum('abc, br, cr -> a', T, V[:, r], W[:, r])
                # We can compute the whole tensor at once?
                # Num[a, r] = sum_b,c T[a, b, c] V[b, r] W[c, r]
                # This is equivalent to: 
                # Temp1 = np.einsum('abc, br -> acr', T, V)  # (nm, pn, R)
                # Num = np.einsum('acr, cr -> ar', Temp1, W) # (nm, R)
                
                Temp1 = np.einsum('abc, br -> acr', T, V)
                Num_U = np.einsum('acr, cr -> ar', Temp1, W)
                
                # Denominator for U: sum_{b,c} V[b, r] W[c, r]
                # This is sum_b V[b,r] * sum_c W[c,r] ? No, it's sum_{b,c} product.
                # Den[r] = sum_b,c V[b,r] W[c,r] = (sum_b V[b,r]) * (sum_c W[c,r])
                # Because V and W are independent in the sum? Yes, sum_{b,c} V[b,r] W[c,r] = (sum_b V[b,r]) (sum_c W[c,r])
                sum_V = np.sum(V, axis=0)
                sum_W = np.sum(W, axis=0)
                Den_U = sum_V * sum_W
                
                # Avoid division by zero
                Den_U[Den_U < 1e-10] = 1e-10
                U = Num_U / Den_U[np.newaxis, :]
                
                # Update V
                # V[b, r] = sum_a,c T[a,b,c] U[a,r] W[c,r] / sum_a,c U[a,r] W[c,r]
                Temp2 = np.einsum('abc, ar -> bcr', T, U)
                Num_V = np.einsum('bcr, cr -> br', Temp2, W)
                
                sum_U = np.sum(U, axis=0)
                Den_V = sum_U * sum_W
                Den_V[Den_V < 1e-10] = 1e-10
                V = Num_V / Den_V[np.newaxis, :]
                
                # Update W
                # W[c, r] = sum_a,b T[a,b,c] U[a,r] V[b,r] / sum_a,b U[a,r] V[b,r]
                Temp3 = np.einsum('abc, ar -> bcr', T, U) # Reuse? No, T is fixed.
                # Temp3[c, b, r] = sum_a T[a,b,c] U[a,r]
                # Num_W[c, r] = sum_b Temp3[c, b, r] V[b, r]
                Temp3_trans = np.einsum('abc, ar -> cbr', T, U)
                Num_W = np.einsum('cbr, br -> cr', Temp3_trans, V)
                
                Den_W = sum_U * sum_V
                Den_W[Den_W < 1e-10] = 1e-10
                W = Num_W / Den_W[np.newaxis, :]

            # Check error
            error, decomposed = is_valid_decomposition(U, V, W, T)
            if error < 1e-8:
                # Try to snap to half-integers
                U_snap = snap_to_half(U)
                V_snap = snap_to_half(V)
                W_snap = snap_to_half(W)
                
                error_snap, _ = is_valid_decomposition(U_snap, V_snap, W_snap, T)
                if error_snap < 1e-8:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                # If snap fails but float works, try snapping to integers?
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                error_int, _ = is_valid_decomposition(U_int, V_int, W_int, T)
                if error_int < 1e-8:
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }

    # Fallback to schoolbook if no lower rank found quickly
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
# EVOLVE-BLOCK-END
