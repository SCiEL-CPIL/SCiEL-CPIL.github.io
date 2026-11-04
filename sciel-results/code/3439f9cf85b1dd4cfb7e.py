import numpy as np
import time

DIMENSIONS = [2, 4, 7]
DEADLINE = 150  # seconds, with safety margin

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Build the tensor constraint matrix
    # For each rank-1 term r: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # The tensor entry at (a,b,c) is sum_r U[a,r]*V[b,r]*W[c,r]
    # We need this to equal T[a,b,c]
    
    # Non-zero tensor entries: T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # All other entries = 0
    
    # Let's set up the problem differently.
    # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is equivalent to: for each c, the matrix (U*V^T)[:,:,c] ... no.
    
    # Actually: for each c, define M_c[a,b] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Then M_c should equal the slice of T at index c.
    
    # T slice at c = k*n+i: T[(i*m+j),(j*p+k),(k*n+i)] = 1 for the specific j
    # So for fixed (k,i), the non-zero entries in slice c=k*n+i are at positions (i*m+j, j*p+k) for j=0..m-1
    
    # Let me build the target tensor slices
    # T_slice[c][a,b] = 1 iff c=k*n+i, a=i*m+j, b=j*p+k for some j
    
    # For given c, let i = c % n, k = c // n
    # Then a = i*m + j, b = j*p + k for j = 0,1,...,m-1
    # So T_slice[c] has ones at (i*m+j, j*p+k) for j=0..m-1
    
    # This is a permutation-like structure!
    
    # Now, for the decomposition: M_c = U * diag(W[c,:]) * V^T = sum_r W[c,r] * U[:,r] * V[:,r]^T
    # So M_c is a linear combination of rank-1 matrices U[:,r]*V[:,r]^T with coefficients W[c,r]
    
    # The target M_c has rank at most m (since it's a sum of m rank-1 terms: for each j, e_{i*m+j} * e_{j*p+k}^T)
    # Actually M_c = sum_j e_{i*m+j} * e_{j*p+k}^T, which has rank m (if the vectors are linearly independent)
    
    # For n=2, m=4, p=7: each M_c has rank 4 (since the 4 vectors e_{i*m+j} for fixed i are distinct, and e_{j*p+k} for fixed k are distinct)
    
    # So we need: M_c = sum_r W[c,r] * (U[:,r] * V[:,r]^T)
    # where U[:,r]*V[:,r]^T is a rank-1 matrix.
    
    # The space of all M_c (for c=0..p*n-1) spans a subspace of R^{(n*m) x (m*p)}
    # The dimension of this subspace is at most p*n = 14
    
    # Each rank-1 term U[:,r]*V[:,r]^T contributes to all M_c with coefficient W[c,r]
    
    # So we're looking for a low-rank decomposition of a set of matrices that are linearly related.
    
    # Alternative approach: think of this as a tensor network problem.
    # The multiplication tensor can be seen as: for each j, we have a "link" between A's j-th column and B's j-th row.
    
    # Let me try a direct optimization approach.
    # Fix r, and try to find U, V, W that minimize the error.
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor."""
        r = U.shape[1]
        # For each non-zero tensor entry, verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check that zero entries are indeed zero (sample some)
        # This is expensive, so we'll check a subset
        for _ in range(100):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            # Check if this should be zero
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            i_c = c % n
            k_c = c // n
            if not (j_a == j_b and i_a == i_c and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    # Try to reduce rank by exploiting structure
    # Key insight: for fixed j, the contribution to the tensor involves A[i,j] and B[j,k]
    # We can group by j
    
    # For each j, define U_j[i] = U[i*m+j, :], V_j[k] = V[j*p+k, :]
    # Then for the tensor entry (i,j),(j,k),(k*n+i):
    # sum_r U_j[i,r] * V_j[k,r] * W[k*n+i, r] = 1
    
    # And for zero entries where j doesn't match:
    # sum_r U[j1*m+i,r] * V[j2*p+k,r] * W[k*n+i, r] = 0 for j1 != j2
    
    # This suggests that for different j values, the vectors U_j and V_j should be "orthogonal" in some sense.
    
    # Let me try a structured decomposition:
    # For each j, use a rank-r_j decomposition of the m_j x p_j sub-problem
    # But here m_j = n = 2 and p_j = p = 7 for each j
    
    # So for each j, we need to decompose the 2x7 matrix multiplication (which is just outer product structure)
    
    # Actually, for fixed j, the constraint is:
    # U_j[i,r] * V_j[k,r] * W[k*n+i, r] summed over r = 1 for all i,k
    # This means: sum_r W[k*n+i, r] * U_j[i,r] * V_j[k,r] = 1 for all i,k
    
    # This is saying that the matrix M_j[i,k] = sum_r W[k*n+i, r] * U_j[i,r] * V_j[k,r] = 1 for all i,k
    
    # So M_j is the all-ones matrix of size n x p = 2 x 7
    
    # And M_j = U_j * diag-ish * V_j^T where the "diag" is replaced by W
    
    # Specifically: M_j[i,k] = sum_r U_j[i,r] * V_j[k,r] * W[k*n+i, r]
    
    # If W were identity-like, this would be U_j * V_j^T, but W depends on (k,i)
    
    # Let me try: assume W[k*n+i, r] = w_r for all k,i (constant across k,i for each r)
    # Then M_j[i,k] = sum_r w_r * U_j[i,r] * V_j[k,r] = 1 for all i,k
    # This means sum_r w_r * U_j[i,r] * V_j[k,r] = 1
    
    # For this to hold for all i,k, we need: sum_r w_r * U_j[i,r] * V_j[k,r] = 1
    # One solution: r=1, w_1=1, U_j[i,1]=1, V_j[k,1]=1 for all i,k
    # But then we need to satisfy the zero-entry constraints
    
    # The zero-entry constraint for j1 != j2:
    # sum_r U[j1*m+i, r] * V[j2*p+k, r] * W[k*n+i, r] = 0
    
    # With the constant W assumption: sum_r w_r * U[j1*m+i, r] * V[j2*p+k, r] = 0
    
    # This is getting complex. Let me just try numerical optimization.
    
    # Approach: for a given rank r, initialize U, V, W randomly and use alternating least squares
    # to minimize the tensor reconstruction error
    
    def tensor_error(U, V, W):
        """Compute the squared error of the tensor decomposition."""
        r = U.shape[1]
        err = 0.0
        # Non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err += (val - 1.0) ** 2
        # Zero entries (check all for exactness)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    for j2 in range(m):
                        if j2 != j:
                            for k2 in range(p):
                                b = j2 * p + k2
                                c = k2 * n + i
                                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                err += val ** 2
                    # Also need to check when i or k don't match
        # This is too slow. Let me use a different error computation.
        return err
    
    # Better: compute the full tensor and compare
    def full_tensor_error(U, V, W):
        """Compute squared error over all tensor entries."""
        r = U.shape[1]
        # Build the reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We can compute this efficiently
        
        # For each r, the contribution is U[:,r] * V[:,r]^T * W[:,r]
        # So T_recon = sum_r np.einsum('ar,br,cr->abc', U, V, W)
        
        # But this is 8*28*14 = 3136 elements, which is fine
        
        T_recon = np.zeros((n * m, m * p, p * n))
        for r in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Build target tensor
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        return np.sum((T_recon - T_target) ** 2)
    
    def als_decompose(rank, max_iters=100, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        # Initialize
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Scale to help convergence
        U *= 2.0
        V *= 2.0
        W *= 2.0
        
        prev_err = np.inf
        for it in range(max_iters):
            # Update U: for each a, U[a,:] should minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # This is a linear system in U[a,:]
            
            # For fixed a: min_Ua sum_{b,c} (T[a,b,c] - Ua^T * (V[b,:] * W[c,:]))^2
            # = min_Ua ||T[a,:,:].flatten() - (V*W)^T * Ua||^2
            # where (V*W)[b,c,r] = V[b,r]*W[c,r]
            
            # So: A_U[a] = sum_{b,c} T[a,b,c] * V[b,r]*W[c,r] for each r
            # And B_U[a] = sum_{b,c} V[b,r1]*V[b,r2]*W[c,r1]*W[c,r2]
            
            # Let me build the normal equations
            
            # Precompute VW_outer[r1,r2] = sum_{b,c} V[b,r1]*V[b,r2]*W[c,r1]*W[c,r2]
            # = (sum_b V[b,r1]*V[b,r2]) * (sum_c W[c,r1]*W[c,r2])
            # = (V^T V)[r1,r2] * (W^T W)[r1,r2]
            
            VtV = V.T @ V
            WtW = W.T @ W
            B_U = np.einsum('ij,kj->ik', VtV, WtW)  # B_U[r1,r2] = VtV[r1,r2]*WtW[r1,r2]
            
            # A_U[a,r] = sum_{b,c} T[a,b,c] * V[b,r]*W[c,r]
            # For given a, T[a,b,c] is non-zero only for specific (b,c)
            # a = i*m+j, so i=a//m, j=a%m
            # T[a,b,c]=1 iff b=j*p+k and c=k*n+i for some k
            
            A_U = np.zeros((n * m, rank))
            for a in range(n * m):
                i = a // m
                j = a % m
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    A_U[a, :] += V[b, :] * W[c, :]
            
            # Solve B_U @ U[a,:] = A_U[a,:] for each a
            try:
                U = np.linalg.solve(B_U, A_U.T).T
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(B_U, A_U.T, rcond=None)[0].T
            
            # Similarly for V
            UtU = U.T @ U
            WtW = W.T @ W
            B_V = np.einsum('ij,kj->ik', UtU, WtW)
            
            A_V = np.zeros((m * p, rank))
            for b in range(m * p):
                j = b // p
                k = b % p
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    A_V[b, :] += U[a, :] * W[c, :]
            
            try:
                V = np.linalg.solve(B_V, A_V.T).T
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(B_V, A_V.T, rcond=None)[0].T
            
            # Similarly for W
            UtU = U.T @ U
            VtV = V.T @ V
            B_W = np.einsum('ij,kj->ik', UtU, VtV)
            
            A_W = np.zeros((p * n, rank))
            for c in range(p * n):
                i = c % n
                k = c // n
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    A_W[c, :] += U[a, :] * V[b, :]
            
            try:
                W = np.linalg.solve(B_W, A_W.T).T
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(B_W, A_W.T, rcond=None)[0].T
            
            # Check convergence
            if it % 5 == 0 or it == max_iters - 1:
                err = full_tensor_error(U, V, W)
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            if time.time() - start_time > DEADLINE - 10:
                break
        
        return U, V, W, prev_err
    
    # Try ranks from low to high
    # Lower bound: the tensor has rank at least max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Actually, the rank is at least the maximum dimension of any mode, so at least 28
    
    # But we know schoolbook is 56, so let's try to beat that
    
    # Try rank 28 first (theoretical lower bound)
    for rank in range(28, min(57, 45)):  # Try up to rank 44
        if time.time() - start_time > DEADLINE - 30:
            break
        
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            U, V, W, err = als_decompose(rank, max_iters=50, tol=1e-8)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U, V, W
            
            if best_err < 1e-6:
                break
        
        if best_err < 1e-4:
            # Verify exactness
            if check_decomposition(best_U, best_V, best_W, tol=1e-8):
                # Convert to list format
                U_list = best_U.tolist()
                V_list = best_V.tolist()
                W_list = best_W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        if time.time() - start_time > DEADLINE - 20:
            break
    
    # Return schoolbook as fallback
    U_list = best_U.tolist() if best_U is not None and best_err < 1 else U_school.tolist()
    V_list = best_V.tolist() if best_V is not None and best_err < 1 else V_school.tolist()
    W_list = best_W.tolist() if best_W is not None and best_err < 1 else W_school.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
