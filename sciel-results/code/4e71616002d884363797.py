"""Matrix multiplication 5x5x5: Attempt lower-rank exact tensor decomposition via ALS."""
import time
import numpy as np
from scipy.optimize import minimize

_DIM = 5

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication.
    
    The tensor T[i*m+j, j*p+k, k*n+i] should equal 1 for all i,j,k in [0,n),[0,m),[0,p)
    and 0 elsewhere (when indices are inconsistent).
    
    We check: for random vectors u, v, w:
      sum_c U[a,c] * V[b,c] * W[c_idx,c] = T_mult[a,b,c_idx]
    where T_mult is the multiplication tensor.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    r = U_arr.shape[1]
    
    rng = np.random.default_rng(12345)
    for _ in range(20):
        u = rng.standard_normal(n * m)
        v = rng.standard_normal(m * p)
        w = rng.standard_normal(p * n)
        
        # Left side: sum_c (U^T u)_c * (V^T v)_c * (W^T w)_c
        left = np.sum((U_arr.T @ u) * (V_arr.T @ v) * (W_arr.T @ w))
        
        # Right side: <T_mult, u ⊗ v ⊗ w>
        # T_mult[i*m+j, j*p+k, k*n+i] = 1
        # So <T_mult, u⊗v⊗w> = sum_{i,j,k} u[i*m+j] * v[j*p+k] * w[k*n+i]
        A_mat = u.reshape(n, m)
        B_mat = v.reshape(m, p)
        # w[k*n+i] corresponds to C[i,k], so C_mat[i,k] = w[k*n+i]
        C_mat = w.reshape(p, n).T  # C_mat[i,k] = w[k*n+i]
        right = np.sum(A_mat * B_mat * C_mat)
        
        if abs(left - right) > tol:
            return False
    return True

def _als_decompose(target_tensor_func, n, m, p, rank, max_iter=500, tol=1e-10, seed=42):
    """ALS tensor decomposition to approximate the multiplication tensor.
    
    target_tensor_func(a, b, c) returns the target tensor value at flat indices (a, b, c)
    where a = i*m+j, b = j'*p+k', c = k''*n+i''
    """
    rng = np.random.default_rng(seed)
    N, M, P = n * m, m * p, p * n
    
    # Initialize
    U = rng.standard_normal((N, rank)) * 0.1
    V = rng.standard_normal((M, rank)) * 0.1
    W = rng.standard_normal((P, rank)) * 0.1
    
    def compute_residual(U, V, W):
        """Compute Frobenius norm of (T_factored - T_target)."""
        # T_factored[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need to compare with T_target[a,b,c]
        # For the multiplication tensor: T_target[i*m+j, j*p+k, k*n+i] = 1
        # Efficient computation via random projections or direct
        
        # Direct: build the full tensor difference
        # T_factored as a 3D array would be (n*m, m*p, p*n) = 25*25*25 = 15625 elements
        # T_factored[i*m+j, j'*p+k', k''*n+i''] = sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i'', r]
        
        # We only care about entries where indices are "consistent": 
        # a = i*m+j, b = j*p+k, c = k*n+i (same i,j,k)
        # For these, target = 1. For others, target = 0.
        
        # Compute sum of squares over ALL entries
        # T_factored[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # ||T_factored - T_target||^2 = sum_{a,b,c} (T_factored[a,b,c] - T_target[a,b,c])^2
        
        # This is expensive to compute directly. Use the fact that:
        # sum_{a,b,c} T_factored[a,b,c]^2 = sum_r (sum_a U[a,r]^2)(sum_b V[b,r]^2)(sum_c W[c,r]^2)
        # No, that's not right either.
        
        # Let's just compute it directly for 25x25x25
        # T_factored[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can compute this as: for each r, add outer(U[:,r], outer(V[:,r], W[:,r]))
        # But that's 15625 * rank ops
        
        # Alternative: use the random projection trick for the residual
        # But for exactness checking, let's compute directly
        
        # Build T_factored efficiently
        # T_factored[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        # = (U @ diag? No)
        # For fixed b: T_factored[:, b, :] = sum_r U[:,r] * V[b,r] * W[:,r]^T
        # = sum_r (U[:,r] * V[b,r]) @ W[:,r]^T
        
        # Let's just do it with numpy broadcasting for 25x25x25
        # U: (25, r), V: (25, r), W: (25, r)
        # T_factored[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Reshape and use einsum
        # T_factored = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 25x25x25 array = 15625 elements. Fine.
        
        T_f = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Build T_target: T_target[i*m+j, j*p+k, k*n+i] = 1
        T_t = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_t[i*m+j, j*p+k, k*n+i] = 1.0
        
        diff = T_f - T_t
        return np.sum(diff * diff)
    
    def solve_for_U(V, W):
        """Solve for U given V, W."""
        # Minimize ||sum_r U[:,r] V[:,r]^T W[:,r]^T - T_target||^2 w.r.t. U
        # This is a linear least squares problem.
        # T_factored[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed b,c: T_factored[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
        # So for each (b,c), we have: T_target[:, b, c] = U @ (V[b,:] * W[c,:])^T
        # Stacking all (b,c): T_target_reshaped = U @ G where G[r, (b,c)] = V[b,r] * W[c,r]
        
        # G shape: (rank, M*P)
        G = np.zeros((rank, M * P))
        for b in range(M):
            for c in range(P):
                G[:, b * P + c] = V[b, :] * W[c, :]
        
        # T_target reshaped: (N, M*P)
        T_t = np.zeros((N, M * P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_t[i*m+j, j*p+k*1 + (k*n+i - (k*n+i)) ]  # This is getting messy
                    # a = i*m+j, b = j*p+k, c = k*n+i
                    T_t[i*m+j, (j*p+k) * P + (k*n+i)] = 1.0
        
        # Solve: T_t = U @ G => U = T_t @ G^T (G G^T)^{-1} if G is full rank
        # Or use lstsq: U = lstsq(G.T, T_t.T).T
        U_new, _, _, _ = np.linalg.lstsq(G.T, T_t.T, rcond=None)
        return U_new.T  # (N, rank)
    
    def solve_for_V(U, W):
        """Solve for V given U, W."""
        # T_factored[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed a,c: T_factored[a,b,c] = sum_r V[b,r] * (U[a,r] * W[c,r])
        # G[r, (a,c)] = U[a,r] * W[c,r]
        G = np.zeros((rank, N * P))
        for a in range(N):
            for c in range(P):
                G[:, a * P + c] = U[a, :] * W[c, :]
        
        T_t = np.zeros((M, N * P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_t[j*p+k, (i*m+j) * P + (k*n+i)] = 1.0
        
        V_new, _, _, _ = np.linalg.lstsq(G.T, T_t.T, rcond=None)
        return V_new.T  # (M, rank)
    
    def solve_for_W(U, V):
        """Solve for W given U, V."""
        # T_factored[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed a,b: T_factored[a,b,c] = sum_r W[c,r] * (U[a,r] * V[b,r])
        # G[r, (a,b)] = U[a,r] * V[b,r]
        G = np.zeros((rank, N * M))
        for a in range(N):
            for b in range(M):
                G[:, a * M + b] = U[a, :] * V[b, :]
        
        T_t = np.zeros((P, N * M))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_t[k*n+i, (i*m+j) * M + (j*p+k)] = 1.0
        
        W_new, _, _, _ = np.linalg.lstsq(G.T, T_t.T, rcond=None)
        return W_new.T  # (P, rank)
    
    prev_res = float('inf')
    for it in range(max_iter):
        U = solve_for_U(V, W)
        V = solve_for_V(U, W)
        W = solve_for_W(U, V)
        
        res = compute_residual(U, V, W)
        if abs(prev_res - res) < tol:
            break
        prev_res = res
    
    return U, V, W, prev_res

def _try_rank(rank, n, m, p, deadline, seed=42):
    """Try to find an exact rank-r decomposition. Returns (U, V, W) or None."""
    if time.time() > deadline:
        return None
    
    try:
        U, V, W, res = _als_decompose(None, n, m, p, rank, max_iter=300, tol=1e-12, seed=seed)
    except Exception:
        return None
    
    # Check if it's exact
    if res < 1e-8:
        # Verify with the proper check
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-7):
            return U, V, W
    
    return None

def construct():
    """Construct tensor decomposition for 5x5x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n = m = p = _DIM
    
    # First, build the schoolbook as fallback
    rank_school = n * m * p  # 125
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find lower-rank decompositions
    # Known: for 5x5, the rank might be less than 125
    # Try ranks: 124, 123, ..., down to some minimum
    # But ALS is expensive, so try a few key ranks
    
    # Based on literature, 5x5 might have rank around 100-120
    # Let's try a range of ranks with different seeds
    
    ranks_to_try = [124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:  # Stop if less than 20s left
            break
        
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 15:
                break
            
            result = _try_rank(rank, n, m, p, deadline, seed=seed)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_result = (U, V, W)
                    best_rank = rank
                    print(f"Found rank {rank} decomposition!")
                break  # Found valid for this rank, try lower
    
    # Return the best result
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
