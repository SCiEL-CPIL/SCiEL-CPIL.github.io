import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _try_find_decomposition(n, m, p, rank, max_iters=5000, tol=1e-8):
    """
    Attempts to find a rank decomposition for n x m x p tensor (all ones) 
    using numerical optimization.
    Returns U, V, W if successful, else None.
    """
    # Variables: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Total variables: rank * (n*m + m*p + p*n)
    nm = n * m
    mp = m * p
    pn = p * n
    total_vars = rank * (nm + mp + pn)
    
    # Initialize randomly
    np.random.seed(42)
    x0 = np.random.randn(total_vars) * 0.1
    
    def objective(x):
        U = x[:rank*nm].reshape(rank, nm).T
        V = x[rank*nm:rank*nm + rank*mp].reshape(rank, mp).T
        W = x[rank*nm + rank*mp:].reshape(rank, pn).T
        
        # Compute tensor T = sum_c U_c (x) V_c (y) W_c (z)
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i, j, k] = 1 for all i, j, k
        
        errors = np.zeros((n, m, p))
        for c in range(rank):
            # u_vec is nm x 1, v_vec is mp x 1, w_vec is pn x 1
            # We need to compute outer products and contract?
            # Simpler: loop over i, j, k? Too slow.
            # Use einsum or reshape tricks.
            pass
        
        # Efficient computation:
        # T is n x m x p
        # Term c: U[:, c] is nm x 1 -> reshape to n x m
        # V[:, c] is mp x 1 -> reshape to m x p
        # W[:, c] is pn x 1 -> reshape to p x n
        # This is not a simple outer product.
        # The definition is T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        
        # Let's just loop over c and add to a pre-allocated tensor.
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            
            # Reshape to 2D
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            
            # For each i, j, k:
            # T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # This can be done with einsum:
            # term = u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # Sum over i, j, k? No, we need element-wise addition to T.
            
            # Let's use a loop for i, j, k if n,m,p are small (2,3,3).
            # 2*3*3 = 18 iterations per c. 15 c's. 270 ops. Very fast.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                        
        diff = T - 1.0
        return np.sum(diff**2)

    # Use scipy.optimize.minimize
    from scipy.optimize import minimize
    
    start_time = time.time()
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'ftol': tol})
    
    if time.time() - start_time > 60:
        return None

    if result.fun < 1e-6:
        x_opt = result.x
        U = x_opt[:rank*nm].reshape(rank, nm).T
        V = x_opt[rank*nm:rank*nm + rank*mp].reshape(rank, mp).T
        W = x_opt[rank*nm + rank*mp:].reshape(rank, pn).T
        
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                        
        if np.allclose(T, 1.0, atol=1e-9):
            return U, V, W
            
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Try to find a decomposition for 2x3x3 with rank 15
    # and use it to build 2x3x6 with rank 30
    target_sub_n, target_sub_m, target_sub_p = 2, 3, 3
    target_sub_rank = 15
    
    U_sub, V_sub, W_sub = None, None, None
    
    # Only attempt if we have time
    if time.time() < 10: # Just a sanity check, we have 120s
        U_sub, V_sub, W_sub = _try_find_decomposition(target_sub_n, target_sub_m, target_sub_p, target_sub_rank)
        
    if U_sub is not None:
        # Construct full decomposition
        rank = target_sub_rank * 2
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # First half (k=0,1,2)
        # U, V, W are same for both halves?
        # For 2x3x6, the first 3 cols of B are B1, next 3 are B2.
        # The decomposition for AB1 uses U, V1, W1.
        # The decomposition for AB2 uses U, V2, W2.
        # Note: U is shared?
        # In the tensor decomposition, the index i (row of A) is the same.
        # So U can be the same.
        # V has index j (col of A / row of B).
        # For B1, j is 0..2, k is 0..2.
        # For B2, j is 0..2, k is 3..5.
        # So V1 is (m x p1) and V2 is (m x p2).
        # In the flattened V matrix of shape (m*p, rank), 
        # The first block (rows 0..mp1-1) corresponds to k=0..2.
        # The second block (rows mp1..mp-1) corresponds to k=3..5.
        
        # W has index k (col of B) and i (row of C).
        # For B1, k is 0..2.
        # For B2, k is 3..5.
        # So W1 is (p1 x n) and W2 is (p2 x n).
        
        # U is (n x m). Same for both.
        
        # Copy U
        # U_sub is (n*m, rank_sub)
        # We need U to be (n*m, rank_full)
        # The first rank_sub columns of U_full should be U_sub.
        # The next rank_sub columns of U_full should be U_sub.
        U[:, :target_sub_rank] = U_sub
        U[:, target_sub_rank:] = U_sub
        
        # Copy V
        # V_sub is (m*p_sub, rank_sub) where p_sub=3.
        # V is (m*p, rank_full) where p=6.
        # Rows 0..5 of V correspond to k=0..2 (j=0..2).
        # Rows 6..11 of V correspond to k=3..5 (j=0..2).
        
        # V_sub has shape (3*3, 15) = (9, 15).
        # We place V_sub in V[0:9, 0:15].
        # And V_sub in V[9:18, 15:30]?
        # Wait, the index j is the same.
        # V[j*p+k, c].
        # For first block, k=0,1,2. Index = j*6 + k.
        # For second block, k=3,4,5. Index = j*6 + k.
        
        # V_sub was computed for p=3.
        # V_sub[j*3+k, c].
        # We need to map this to V[j*6+k, c].
        
        V_block = np.zeros((m * p, target_sub_rank))
        for j in range(m):
            for k in range(target_sub_p):
                V_block[j * p + k, :] = V_sub[j * target_sub_p + k, :]
                
        V[:, :target_sub_rank] = V_block
        V[:, target_sub_rank:] = V_block
        
        # Copy W
        # W_sub is (p_sub*n, rank_sub) = (3*2, 15) = (6, 15).
        # W is (p*n, rank_full) = (6*2, 30) = (12, 30).
        # W index = k*n + i.
        # For first block, k=0,1,2.
        # For second block, k=3,4,5.
        
        W_block = np.zeros((p * n, target_sub_rank))
        for k in range(target_sub_p):
            for i in range(n):
                W_block[k * n + i, :] = W_sub[k * target_sub_p + i, :]
                
        W[:, :target_sub_rank] = W_block
        W[:, target_sub_rank:] = W_block
        
        # Verify full decomposition
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                        
        if np.allclose(T, 1.0, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    rank = rank_school
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
