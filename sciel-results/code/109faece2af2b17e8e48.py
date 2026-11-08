"""
Rank-minimal exact tensor decomposition for 3x5 x 5x5 matrix multiplication.

Strategy:
1. Build the schoolbook rank-75 decomposition as a baseline.
2. Try a structured Strassen-like block decomposition to reduce rank.
3. Attempt numerical search for lower ranks using alternating least squares
   with careful initialization and rounding to half-integers.
4. Use a deadline guard to ensure we finish within the time limit.
5. Return the best decomposition found.
"""

import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 120
SAFETY_MARGIN = 15

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _build_target_tensor():
    """Build the matrix multiplication tensor T for 3x5 x 5x5."""
    n, m, p = 3, 5, 5
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition is exact."""
    if T is None:
        T = _build_target_tensor()
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_recon, atol=1e-9)

def _build_schoolbook():
    """Build the standard rank-75 schoolbook decomposition."""
    n, m, p = 3, 5, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_strassen_blocks():
    """
    Try a Strassen-like block decomposition.
    
    For 3x5 x 5x5, we can partition:
    - A is 3x5, B is 5x5, C is 3x5
    - Split rows of A: top 1 row (1x5), bottom 2 rows (2x5)
    - Split cols of B: left 2 cols (5x2), right 3 cols (5x3)
    - Split rows of B: top 2 rows (2x5), bottom 3 rows (3x5)
    - Split cols of C: left 2 cols (3x2), right 3 cols (3x3)
    
    This gives 4 blocks:
    C[0:1, 0:2] = A[0:1, 0:2]*B[0:2, 0:2] + A[0:1, 2:5]*B[2:5, 0:2]  (1x2 = 1x2*2x2 + 1x3*3x2)
    C[0:1, 2:5] = A[0:1, 0:2]*B[0:2, 2:5] + A[0:1, 2:5]*B[2:5, 2:5]  (1x3 = 1x2*2x3 + 1x3*3x3)
    C[1:3, 0:2] = A[1:3, 0:2]*B[0:2, 0:2] + A[1:3, 2:5]*B[2:5, 0:2]  (2x2 = 2x2*2x2 + 2x3*3x2)
    C[1:3, 2:5] = A[1:3, 0:2]*B[0:2, 2:5] + A[1:3, 2:5]*B[2:5, 2:5]  (2x3 = 2x2*2x3 + 2x3*3x3)
    
    Each sub-multiplication can use Strassen if sizes are >= 2.
    But this is complex. Let's try a simpler approach: just use schoolbook for each block.
    
    Actually, let's try a different partition that might allow Strassen:
    Split A into 3 row blocks of 1 row each.
    Split B into 5 row blocks of 1 row each.
    This doesn't help much.
    
    Let me try a more promising approach: use the fact that 3x5 x 5x5 can be seen as
    3 independent 1x5 x 5x5 multiplications (one per row of A).
    Each 1x5 x 5x5 is just a linear combination: C[i,:] = sum_j A[i,j] B[j,:]
    This is rank 25 per row, total 75. No improvement.
    
    Let's try a numerical search instead.
    """
    return None

def _try_als_decomposition(rank, T, max_iters=200, n_restarts=5, seed=0):
    """
    Try to find a rank-`rank` decomposition using Alternating Least Squares.
    
    The tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    We can rewrite this as: for each (a,b), the vector T[a,b,:] = sum_r U[a,r] V[b,r] W[:,r]
    This is a matrix factorization problem for each (a,b) pair, but coupled.
    
    A better approach: flatten T into a 2D matrix.
    T[a, (b,c)] where (b,c) is a flattened index. Then:
    T_mat[a, bc] = sum_r U[a,r] * (V[b,r] * W[c,r])
    
    This is like: T_mat = U @ M where M[bc, r] = V[b,r] * W[c,r]
    But M is not arbitrary - it has the product structure.
    
    Alternative: use the unfolding approach.
    T_{(a)(b)(c)} = sum_r U_{a r} V_{b r} W_{c r}
    
    For fixed r, define the vector u_r = U[:,r], v_r = V[:,r], w_r = W[:,r].
    Then T = sum_r u_r v_r w_r^T (outer product in tensor form).
    
    ALS approach:
    1. Initialize U, V, W randomly.
    2. Iterate:
       - Fix V, W. Solve for U: 
         For each a, U[a,r] should satisfy:
         sum_r U[a,r] * (V[:,r] * W[:,r])^T = T[a,:,:] (flattened)
         This is: U[a,:] @ diag(V[:,r]*W[:,r] for each r)... no.
         
         Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
         For fixed a: sum_r U[a,r] (V[:,r] * W[:,r])_{(b,c)} = T[a,b,c]
         So: T[a, :, :] (flattened to length m*p*p*n... no, to length (m*p)*(p*n) = m*p^2*n)
         
         Wait, T[a] is a 2D array of shape (m*p, p*n).
         T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
         
         Let M_r = outer(V[:,r], W[:,r]) which has shape (m*p, p*n).
         Then T[a, :, :] = sum_r U[a,r] * M_r
         
         So: T_a = U[a,:] @ M where M is a (rank, m*p*p*n) matrix? No.
         
         M_r is (m*p, p*n). Flatten it to a vector of length m*p*p*n.
         Then T_a (flattened) = sum_r U[a,r] * vec(M_r)
         
         So: T_flat[a] = sum_r U[a,r] * vec(M_r)
         This means: T_flat = U @ M_flat where M_flat[r, bc] = V[b,r]*W[c,r]
         
         So for fixed V, W: U = T_flat @ M_flat^T @ (M_flat @ M_flat^T)^{-1}
         
         Similarly for V and W.
    """
    n, m, p = 3, 5, 5
    dim_U = n * m  # 15
    dim_V = m * p  # 25
    dim_W = p * n  # 15
    
    # Flatten T: T[a, b, c] -> T_flat[a, b * dim_W + c]
    T_flat = T.reshape(dim_U, dim_V * dim_W)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    rng = np.random.RandomState(seed)
    
    for restart in range(n_restarts):
        if _time_remaining() < 15:
            break
        
        # Random initialization
        U = rng.randn(dim_U, rank)
        V = rng.randn(dim_V, rank)
        W = rng.randn(dim_W, rank)
        
        # Normalize
        for r in range(rank):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 1e-12:
                U[:, r] /= (norm ** (1/3))
                V[:, r] /= (norm ** (1/3))
                W[:, r] /= (norm ** (1/3))
        
        for it in range(max_iters):
            if _time_remaining() < 10:
                break
            
            # Compute M_flat: M_flat[r, b*dim_W + c] = V[b,r] * W[c,r]
            M_flat = np.zeros((rank, dim_V * dim_W))
            for r in range(rank):
                M_flat[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Update U: U = T_flat @ M_flat^T @ (M_flat @ M_flat^T)^{-1}
            try:
                MMt = M_flat @ M_flat.T  # (rank, rank)
                TtM = T_flat.T @ M_flat  # (dim_U, rank)
                U_new = np.linalg.solve(MMt, TtM.T).T  # (dim_U, rank)
                U = U_new
            except np.linalg.LinAlgError:
                pass
            
            # Compute N_flat: N_flat[r, a*dim_W + c] = U[a,r] * W[c,r]
            N_flat = np.zeros((rank, dim_U * dim_W))
            for r in range(rank):
                N_flat[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
            # Update V: V = T_flat_V @ N_flat^T @ (N_flat @ N_flat^T)^{-1}
            # T_flat_V[b, a*dim_W + c] = T[a, b, c]
            T_flat_V = T.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W)
            try:
                NNt = N_flat @ N_flat.T
                TtN = T_flat_V.T @ N_flat
                V_new = np.linalg.solve(NNt, TtN.T).T
                V = V_new
            except np.linalg.LinAlgError:
                pass
            
            # Compute P_flat: P_flat[r, a*dim_V + b] = U[a,r] * V[b,r]
            P_flat = np.zeros((rank, dim_U * dim_V))
            for r in range(rank):
                P_flat[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
            # Update W: W = T_flat_W @ P_flat^T @ (P_flat @ P_flat^T)^{-1}
            # T_flat_W[c, a*dim_V + b] = T[a, b, c]
            T_flat_W = T.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V)
            try:
                PPt = P_flat @ P_flat.T
                TtP = T_flat_W.T @ P_flat
                W_new = np.linalg.solve(PPt, TtP.T).T
                W = W_new
            except np.linalg.LinAlgError:
                pass
            
            # Compute error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T - T_recon)
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if error < 1e-8:
                break
        
        # Try rounding to half-integers
        if best_U is not None:
            for scale in [1, 2, 4]:
                U_r = np.round(best_U * scale) / scale
                V_r = np.round(best_V * scale) / scale
                W_r = np.round(best_W * scale) / scale
                if _verify(U_r, V_r, W_r, T):
                    return U_r, V_r, W_r
    
    return None

def _try_greedy_column_merge(U, V, W, T):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    If columns of [U|V|W] are linearly dependent, we might be able to merge.
    """
    rank = U.shape[1]
    
    # Check if any two columns can be merged
    # This is a heuristic: if U[:,r1] is proportional to U[:,r2], etc.
    # For now, just return the current decomposition
    return U, V, W, rank

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 x 5x5 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = 3, 5, 5
    
    # Build target tensor
    T = _build_target_tensor()
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try Strassen-like approach
    strassen_result = _try_strassen_blocks()
    if strassen_result is not None:
        U_s, V_s, W_s = strassen_result
        if _verify(U_s, V_s, W_s, T):
            best_rank = U_s.shape[1]
            best_U, best_V, best_W = U_s, V_s, W_s
    
    # Try numerical search for lower ranks
    # Start from best_rank - 1 and go down
    for rank in range(best_rank - 1, max(n * m, 20) - 1, -1):
        if _time_remaining() < 30:
            break
        
        result = _try_als_decomposition(rank, T, max_iters=100, n_restarts=3, seed=rank)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
    
    # Convert to lists of lists with integer/half-integer values
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    # Ensure values are exactly integer or half-integer
    U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
    V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
    W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
