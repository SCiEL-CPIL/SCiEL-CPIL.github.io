"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Uses vectorized Alternating Least Squares to find a low-rank decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    if _time_left() < 5:
        return _schoolbook(n, m, p)
    
    # Try ranks from 63 down to find the minimum valid rank
    # We start from a reasonable guess. Schoolbook is 63.
    # For 3x3x3, min rank is 23. For 3x3x7, let's try to find something < 63.
    best_result = _schoolbook(n, m, p)
    best_rank = 63
    
    # Try a few specific ranks that might be achievable
    # Note: Finding the *exact* minimum is NP-hard, so we search for a valid low rank.
    for rank in range(62, 40, -1):
        if _time_left() < 10:
            break
        
        result = _try_rank_fast(rank, n, m, p)
        if result is not None:
            best_result = result
            best_rank = rank
            # If we find a very low rank, we might try to go lower, 
            # but for safety and time, we stop if we find a good one.
            # Let's continue to try to lower it if time permits.
            if rank < 50:
                # Try to go lower
                for r2 in range(rank - 1, 30, -1):
                    if _time_left() < 10:
                        break
                    res2 = _try_rank_fast(r2, n, m, p)
                    if res2 is not None:
                        best_result = res2
                        best_rank = r2
                break
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank n*m*p)."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
    """
    Verify the decomposition is correct using NumPy.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Reshape U, V, W to 3D tensors
    # U[i*m+j, r] -> U_tensor[i, j, r]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute the tensor T[i, k, i2, j2, k2] = sum_r W[k, i, r] * U[i2, j, r] * V[j2, k2, r]
    # Wait, the indices in the problem description:
    # W[k*n+i, r]
    # U[i*m+j, r]
    # V[j*p+k, r]
    # The multiplication is C = A * B.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, k, i2, j2, k2] is 1 if i==i2, j==j2, k==k2 else 0?
    # No, the tensor decomposition of matrix multiplication is:
    # T_{i, j, k, i', j', k'} = delta_{i, i'} delta_{j, j'} delta_{k, k'}
    # And the decomposition is sum_r U_{i, j, r} V_{j', k', r} W_{k, i', r} ?
    # Let's check the standard convention.
    # Usually: T = sum_r U_r \otimes V_r \otimes W_r
    # T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r} ... this is cyclic.
    # The problem states:
    # U shape (n*m, rank) -> index i*m+j
    # V shape (m*p, rank) -> index j*p+k
    # W shape (p*n, rank) -> index k*n+i
    # The tensor is T[i, j, k, i2, j2, k2] = 1 if i=i2, j=j2, k=k2.
    # The decomposition must satisfy:
    # T[i, j, k, i2, j2, k2] = sum_r U[i*m+j, r] * V[j2*p+k2, r] * W[k*n+i2, r]
    # Wait, usually the indices for the factors correspond to the input/output indices.
    # Let's assume the standard:
    # U corresponds to A indices (i, j)
    # V corresponds to B indices (j, k) -> but V is indexed by j2, k2?
    # W corresponds to C indices (i, k) -> but W is indexed by k, i2?
    
    # Let's construct the expected tensor E of shape (n, m, p, n, m, p)
    # E[i, j, k, i2, j2, k2] = 1 if i==i2 and j==j2 and k==k2 else 0.
    
    # Compute actual tensor A from U, V, W
    # A[i, j, k, i2, j2, k2] = sum_r U[i, j, r] * V[j2, k2, r] * W[k, i2, r]
    # Note: W is indexed by k*n+i2 in the matrix, so W[k, i2, r].
    # V is indexed by j2*p+k2 in the matrix, so V[j2, k2, r].
    # U is indexed by i*m+j in the matrix, so U[i, j, r].
    
    # We can compute this using einsum
    # U_t: (n, m, rank)
    # V_t: (m, p, rank)  (indices j2, k2)
    # W_t: (p, n, rank)  (indices k, i2)
    
    # A[i, j, k, i2, j2, k2] = sum_r U[i, j, r] * V[j2, k2, r] * W[k, i2, r]
    # This is a 6-index tensor.
    
    # Let's verify a few random indices to be fast, or compute the whole thing if small.
    # n=3, m=3, p=7. Total size 3*3*7*3*3*7 = 41157. This is small.
    
    # Reshape
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute A
    # We need to sum over r
    # A = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t) ? No, that collapses i,j,k.
    # We need the full 6D tensor to compare with E.
    # But E is sparse.
    # E[i, j, k, i2, j2, k2] is 1 only when i==i2, j==j2, k==k2.
    # So we only need to check A[i, j, k, i, j, k] == 1 for all i,j,k
    # and A[i, j, k, i2, j2, k2] == 0 for all other combinations.
    
    # Check 1s:
    # For all i, j, k:
    # Sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] should be 1.
    # Note: V index is j*p+k. So V_t[j, k, r].
    # W index is k*n+i. So W_t[k, i, r].
    # U index is i*m+j. So U_t[i, j, r].
    
    diag_vals = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    if not np.allclose(diag_vals, 1.0, atol=tol):
        return False
        
    # Check 0s:
    # This is harder to vectorize directly for "all other combinations".
    # But we can check the norm of the whole tensor minus the diagonal part?
    # Or just check a random sample of off-diagonal elements.
    # Given the small size, let's compute the full tensor and check.
    
    # A_full[i, j, k, i2, j2, k2]
    # We can do this by outer products.
    # But 41k elements is fine.
    
    # Let's use a different approach: 
    # The tensor T is the identity tensor on the product space.
    # We can check if the decomposition matches the identity.
    
    # Construct the full tensor T
    # T = np.zeros((n, m, p, n, m, p))
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       T[i, j, k, i, j, k] = 1
    
    # Compute A_full
    # A_full = np.einsum('ijr,jkr,kir->ijk', ...) only gives the diagonal.
    # We need A[i, j, k, i2, j2, k2].
    # This is equivalent to:
    # For each r, the term is U[:, :, r] (n,m) x V[:, :, r] (m,p) x W[:, :, r] (p,n)
    # This forms a rank-1 tensor in the 6D space? No.
    
    # Let's just loop over the small dimensions to check zeros, it's fast enough.
    # n=3, m=3, p=7.
    # Number of checks: 3*3*7*3*3*7 = 41157.
    # In Python, this is ~40k operations, which is < 1ms.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            val = 0.0
                            # sum_r U[i, j, r] * V[j2, k2, r] * W[k, i2, r]
                            # U[i*m+j, r]
                            # V[j2*p+k2, r]
                            # W[k*n+i2, r]
                            u_row = U[i*m+j, :]
                            v_row = V[j2*p+k2, :]
                            w_row = W[k*n+i2, :]
                            val = np.dot(u_row, v_row * w_row)
                            if abs(val) > tol:
                                return False
    return True

def _try_rank_fast(rank, n, m, p, iters=50, seed=42):
    """
    Fast ALS using NumPy.
    """
    if _time_left() < 5:
        return None
        
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Precompute indices for speed? No, just use vectorized ops.
    
    for it in range(iters):
        if _time_left() < 5:
            break
            
        # Solve for U
        # For each i, j:
        # U[i, j, :] minimizes || sum_r U[i, j, r] V[j', k', r] W[k, i', r] - T ||^2
        # This is a linear system.
        # Let's flatten the target tensor for each (i, j).
        
        # To speed up, we can solve for all U rows at once?
        # The system for U[i, j, :] is:
        # sum_r U[i, j, r] * (V[:, :, r] * W[:, :, r]) ... no.
        
        # Let's stick to the previous logic but vectorized inner product.
        # For U[i, j]:
        # Target T_{i, j, k, i2, j2, k2} is 1 if i==i2, j==j2, k==k2.
        # For fixed i, j, the non-zero targets are at (i, j, k, i, j, k) for all k.
        # So we only need to satisfy:
        # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all k.
        # And 0 for all other (i2, j2, k2).
        
        # This is a linear system A u = b.
        # A has rows corresponding to all (i2, j2, k2) tuples.
        # b is 1 if (i2, j2, k2) == (i, j, k) for the specific k? No.
        # b is 1 if i2==i, j2==j, k2==k.
        
        # Since n, m, p are small, we can construct A for each (i, j) quickly.
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Construct A (size n*m*p*n*m*p x rank)
                # This is large: 3*3*7*3*3*7 = 41157 rows.
                # 41157 x rank. Rank ~ 50.
                # 41157 * 50 * 50 * 3 * 3 * 7 ... might be slow in Python loops.
                # Let's use a smarter way.
                
                # We know the target is sparse.
                # We only care about the diagonal entries for the "1"s.
                # But we must also ensure the "0"s are 0.
                
                # Let's use the full system but build it with NumPy broadcasting.
                # A_{(i2, j2, k2), r} = V[j2, k2, r] * W[k, i2, r] ? No, W depends on k from the U index?
                # No, U is indexed by i, j.
                # The term in the sum is U[i, j, r] * V[j2, k2, r] * W[k, i2, r].
                # Here k is the output index from A (which is input to W? No).
                # Let's re-read the index mapping.
                # U[i*m+j, r]
                # V[j*p+k, r]  -> j is row of B, k is col of B.
                # W[k*n+i, r]  -> k is col of B (row of C?), i is row of C.
                
                # The product is C[i, k] = sum_j A[i, j] B[j, k].
                # Tensor T_{i, j, k, i2, j2, k2} = delta_{i, i2} delta_{j, j2} delta_{k, k2}.
                # Decomposition: sum_r U[i, j, r] V[j2, k2, r] W[k, i2, r]
                
                # For fixed i, j (U index), we vary r.
                # The equation is:
                # sum_r U[i, j, r] * (V[j2, k2, r] * W[k, i2, r]) = T_{i, j, k, i2, j2, k2}
                
                # Let X_{(j2, k2, k, i2), r} = V[j2, k2, r] * W[k, i2, r]
                # Then sum_r U[i, j, r] * X_{(j2, k2, k, i2), r} = T_{i, j, k, i2, j2, k2}
                
                # Note that T depends on k, i2, j2, k2.
                # T is 1 only if i2==i, j2==j, k2==k.
                
                # So for each (i, j), we have a system:
                # A u = b
                # where A has rows indexed by (j2, k2, k, i2).
                # A[(j2, k2, k, i2), r] = V[j2, k2, r] * W[k, i2, r]
                # b[(j2, k2, k, i2)] = 1 if (j2==j and k2==k and i2==i) else 0.
                
                # Number of rows: m * p * p * n = 3 * 7 * 7 * 3 = 441.
                # This is small!
                
                # Build A for this (i, j)
                # V is (m, p, rank)
                # W is (p, n, rank)
                
                # We need to form V[j2, k2, r] * W[k, i2, r] for all j2, k2, k, i2, r.
                # This is an outer product of V and W, but with different indices.
                
                # V_exp: (m, p, 1, 1, rank)
                # W_exp: (1, 1, p, n, rank)
                # Product: (m, p, p, n, rank)
                
                V_exp = V.reshape(m, p, 1, 1, rank)
                W_exp = W.reshape(1, 1, p, n, rank)
                A_full = V_exp * W_exp  # (m, p, p, n, rank)
                
                # Flatten the first 4 dims to get (441, rank)
                A_mat = A_full.reshape(-1, rank)
                
                # Construct b
                # b is 1 if j2==j, k2==k, i2==i.
                # The index in A_mat corresponds to (j2, k2, k, i2).
                # We need to find the index where j2=j, k2=k, i2=i.
                # Note that k is the 3rd index (0-based) in the 4D tuple (j2, k2, k, i2).
                # And i is the 4th index.
                
                # Let's create b as zeros and set the right spot.
                b_vec = np.zeros(441)
                # Index calculation:
                # j2 in [0, m), k2 in [0, p), k in [0, p), i2 in [0, n)
                # Linear index = ((j2 * p + k2) * p + k) * n + i2
                # We want j2=j, k2=k, i2=i.
                # But k is a variable in the tensor, and also the column of B.
                # In the target T, the non-zero is at k2=k.
                # So we set b for all k?
                # No, T_{i, j, k, i2, j2, k2} is 1 if i2=i, j2=j, k2=k.
                # So for a fixed (i, j), the non-zero entries in b are at positions
                # where j2=j, i2=i, and k2=k.
                # There are p such entries (one for each k).
                
                for k_val in range(p):
                    idx = ((j * p + k_val) * p + k_val) * n + i
                    b_vec[idx] = 1.0
                    
                # Solve A_mat u = b_vec
                u_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U_new[i*m+j, :] = u_sol
                
        U = U_new
        
        # Similarly for V and W
        # V[j2, k2, :]
        # System: sum_r V[j2, k2, r] * (U[i, j, r] * W[k, i2, r]) = T
        # A rows: (i, j, k, i2). Size n*m*p*n = 3*3*7*3 = 189.
        
        V_new = np.zeros_like(V)
        for j2 in range(m):
            for k2 in range(p):
                # A[(i, j, k, i2), r] = U[i, j, r] * W[k, i2, r]
                # U_exp: (n, m, 1, 1, rank)
                # W_exp: (1, 1, p, n, rank)
                # Product: (n, m, p, n, rank)
                
                U_exp = U.reshape(n, m, 1, 1, rank)
                W_exp = W.reshape(1, 1, p, n, rank)
                A_full = U_exp * W_exp
                A_mat = A_full.reshape(-1, rank)
                
                b_vec = np.zeros(189)
                # Non-zero if i==i2, j==j2, k==k2.
                # Indices: i in [0,n), j in [0,m), k in [0,p), i2 in [0,n)
                # Linear index = ((i * m + j) * p + k) * n + i2
                # We want j==j2, k==k2, i==i2.
                for i_val in range(n):
                    idx = ((i_val * m + j2) * p + k2) * n + i_val
                    b_vec[idx] = 1.0
                    
                v_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V_new[j2*p+k2, :] = v_sol
                
        V = V_new
        
        # W[k, i2, :]
        # System: sum_r W[k, i2, r] * (U[i, j, r] * V[j2, k2, r]) = T
        # A rows: (i, j, j2, k2). Size n*m*m*p = 3*3*3*7 = 189.
        
        W_new = np.zeros_like(W)
        for k in range(p):
            for i2 in range(n):
                # A[(i, j, j2, k2), r] = U[i, j, r] * V[j2, k2, r]
                # U_exp: (n, m, 1, 1, rank)
                # V_exp: (1, m, p, 1, rank) ? No, V is (m, p, rank).
                # We need V[j2, k2, r].
                # U[i, j, r] * V[j2, k2, r]
                # U shape (n, m, rank) -> (n, m, 1, 1, rank)
                # V shape (m, p, rank) -> (1, m, p, 1, rank)
                
                U_exp = U.reshape(n, m, 1, 1, rank)
                V_exp = V.reshape(1, m, p, 1, rank)
                A_full = U_exp * V_exp
                A_mat = A_full.reshape(-1, rank)
                
                b_vec = np.zeros(189)
                # Non-zero if i==i2, j==j2, k==k2.
                # Indices: i in [0,n), j in [0,m), j2 in [0,m), k2 in [0,p)
                # Linear index = ((i * m + j) * m + j2) * p + k2
                # We want i==i2, j==j2, k2==k.
                for j_val in range(m):
                    idx = ((i2 * m + j_val) * m + j_val) * p + k
                    b_vec[idx] = 1.0
                    
                w_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_new[k*n+i2, :] = w_sol
                
        W = W_new
        
        # Check convergence
        if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-6):
            # Try to round
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_decomposition_fast(U_r, V_r, W_r, n, m, p, tol=1e-5):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if _verify_decomposition_fast(U_r, V_r, W_r, n, m, p, tol=1e-5):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
            # Return float if exact rounding fails but float is valid
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def _try_rank(rank, n, m, p):
    """Wrapper to try multiple seeds."""
    if _time_left() < 10:
        return None
    for seed in range(3):
        if _time_left() < 10:
            break
        res = _try_rank_fast(rank, n, m, p, seed=seed)
        if res:
            return res
    return None
# EVOLVE-BLOCK-END
