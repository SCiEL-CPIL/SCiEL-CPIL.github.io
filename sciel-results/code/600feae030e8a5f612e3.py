import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _strassen_222():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) such that
    sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Also verify with random matrices
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(rank):
            L_r = 0.0
            for i in range(n):
                for j in range(m):
                    L_r += U[i*m+j, r] * A[i, j]
            M_r = 0.0
            for j in range(m):
                for k in range(p):
                    M_r += V[j*p+k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k*n+i, r] * L_r * M_r
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _als_decompose(n, m, p, rank, max_iter=100, tol=1e-8, deadline=None):
    """ALS decomposition for the matrix multiplication tensor.
    Returns U (n*m, rank), V (m*p, rank), W (p*n, rank) or None if not converged.
    """
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Normalize to avoid scaling issues
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        total = norm_u * norm_v * norm_w
        if total > 1e-12:
            U[:, r] /= total ** (1/3)
            V[:, r] /= total ** (1/3)
            W[:, r] /= total ** (1/3)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Update U: for each r, solve min_U sum_{i,j,k} (T[i,j,k] - sum_{r'} U[i,j,r']*V[j,k,r']*W[k,i,r'])^2
        # This is a least squares problem for each r
        for r in range(rank):
            # Fix V and W, solve for U[:, r]
            # The residual for U[:, r] is: T[:, :, :] - sum_{r' != r} U[:, r'] outer V[:, r'] outer W[:, r']
            # But we can use the ALS trick: for each (i,j), the coefficient is sum_k W[k,i,r] * V[j,k,r]
            # So U[i,j,r] should be such that sum_{r'} U[i,j,r'] * (sum_k V[j,k,r']*W[k,i,r']) = T[i,j,k] for all k
            # Actually, the standard ALS approach:
            # For fixed V, W, the optimal U[:, r] minimizes ||T - sum_r' U_r' V_r' W_r'||_F
            # This is a linear system in U[:, r]
            
            # Build the system: for each (i,j), we have equations for each k:
            # sum_{r'} U[i,j,r'] * V[j,k,r'] * W[k,i,r'] = T[i,j,k]
            # But since we're updating one r at a time, we can write:
            # U[i,j,r] * (V[j,k,r] * W[k,i,r]) + sum_{r' != r} U[i,j,r'] * V[j,k,r'] * W[k,i,r'] = T[i,j,k]
            # This is coupled across k, so we need to solve a least squares problem.
            
            # Alternative: use the matrix form.
            # Let's define for each r:
            # A_r[i,j,k] = V[j,k,r] * W[k,i,r]
            # Then we want U[:, r] to satisfy: sum_{r'} U[:, r'] * A_r' = T (element-wise)
            # For a single r, this is: U[:, r] * A_r + sum_{r' != r} U[:, r'] * A_r' = T
            # => U[:, r] * A_r = T - sum_{r' != r} U[:, r'] * A_r'
            
            # This is a tensor equation. We can vectorize:
            # Let u_r = vec(U[:, r]) (length n*m)
            # Let t = vec(T) (length n*m*p)
            # The equation is: sum_{r'} (u_r' \otimes a_r') = t, where a_r' = vec(A_r')
            # For updating u_r: (a_r \otimes I) u_r = t - sum_{r' != r} (a_r' \otimes I) u_r'
            # This is a large system. Instead, let's use a simpler approach.
            
            # Simpler ALS: treat it as a 3-mode tensor unfolding.
            # U-mode unfolding: U_(1) is (n*m) x (p*rank)
            # But this is getting complex. Let's use a different approach.
            
            # Actually, for matrix multiplication tensor, the ALS update is:
            # U[:, r] = argmin ||T - sum_r' U_r' V_r' W_r'||_F
            # This can be solved by noticing that for each (i,j), we have:
            # sum_{r'} U[i,j,r'] * (V_r' * W_r')[i,j,:] = T[i,j,:]
            # where (V_r' * W_r')[i,j,k] = V[j,k,r'] * W[k,i,r']
            
            # So for each (i,j), we have a linear system:
            # U[i,j,:] @ M[i,j] = T[i,j,:]
            # where M[i,j] is a (p x rank) matrix with M[i,j,k,r'] = V[j,k,r'] * W[k,i,r']
            
            # But we're updating one r at a time, so:
            # U[i,j,r] * M[i,j,k,r] + sum_{r' != r} U[i,j,r'] * M[i,j,k,r'] = T[i,j,k]
            
            # Let's just do a full LSQ update for all r at once.
            pass
        
        # Let's use a simpler approach: coordinate descent on r
        for r in range(rank):
            if deadline is not None and time.time() > deadline:
                return None
            
            # Build the matrix M[i,j,k] = V[j,k,r] * W[k,i,r] for the current r
            # We want to find U[:, r] to minimize ||T - sum_r' U_r' * M_r'||_F
            # For the r-th component, we can write:
            # U[:, r] should be such that for all (i,j,k):
            # U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] - sum_{r' != r} U[i,j,r'] * V[j,k,r'] * W[k,i,r']
            
            # Let's define the residual:
            residual = np.zeros((n, m, p))
            for r_prime in range(rank):
                if r_prime == r:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] += U[i*m+j, r_prime] * V[j*p+k, r_prime] * W[k*n+i, r_prime]
            
            # Now we need to solve: U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] - residual[i,j,k]
            # for all (i,j,k). This is a linear system in U[:, r].
            # For each (i,j), we have p equations:
            # sum_{k} ... no, it's element-wise.
            # U[i,j,r] = (T[i,j,k] - residual[i,j,k]) / (V[j,k,r] * W[k,i,r]) for each k
            # But this must be consistent across k. So we do a least squares fit.
            
            # For each (i,j), solve: U[i,j,r] * v_k = t_k for k=0..p-1
            # where v_k = V[j,k,r] * W[k,i,r], t_k = T[i,j,k] - residual[i,j,k]
            # This is a 1D least squares problem: U[i,j,r] = sum(v_k * t_k) / sum(v_k^2)
            
            for i in range(n):
                for j in range(m):
                    v = np.array([V[j*p+k, r] * W[k*n+i, r] for k in range(p)])
                    t = np.array([1.0 if (i, j, k) in [(a, b, c) for a in range(n) for b in range(m) for c in range(p)] else 0.0 for k in range(p)])
                    # T[i,j,k] = 1 for all i,j,k in the multiplication tensor
                    t = np.ones(p)
                    t = t - np.array([residual[i, j, k] for k in range(p)])
                    denom = np.dot(v, v)
                    if denom > 1e-12:
                        U[i*m+j, r] = np.dot(v, t) / denom
                    else:
                        U[i*m+j, r] = 0
        
        # Update V similarly
        for r in range(rank):
            if deadline is not None and time.time() > deadline:
                return None
            
            residual = np.zeros((n, m, p))
            for r_prime in range(rank):
                if r_prime == r:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] += U[i*m+j, r_prime] * V[j*p+k, r_prime] * W[k*n+i, r_prime]
            
            for j in range(m):
                for k in range(p):
                    v = np.array([U[i*m+j, r] * W[k*n+i, r] for i in range(n)])
                    t = np.ones(n) - np.array([residual[i, j, k] for i in range(n)])
                    denom = np.dot(v, v)
                    if denom > 1e-12:
                        V[j*p+k, r] = np.dot(v, t) / denom
                    else:
                        V[j*p+k, r] = 0
        
        # Update W similarly
        for r in range(rank):
            if deadline is not None and time.time() > deadline:
                return None
            
            residual = np.zeros((n, m, p))
            for r_prime in range(rank):
                if r_prime == r:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] += U[i*m+j, r_prime] * V[j*p+k, r_prime] * W[k*n+i, r_prime]
            
            for k in range(p):
                for i in range(n):
                    v = np.array([U[i*m+j, r] * V[j*p+k, r] for j in range(m)])
                    t = np.ones(m) - np.array([residual[i, j, k] for j in range(m)])
                    denom = np.dot(v, v)
                    if denom > 1e-12:
                        W[k*n+i, r] = np.dot(v, t) / denom
                    else:
                        W[k*n+i, r] = 0
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (val - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < tol:
            break
    
    # Final verification
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 120.0
    deadline = start_time + time_limit - 5
    
    # Schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Block decomposition using Strassen
    def build_2x4x7_strassen():
        """Build 2x4 @ 4x7 by splitting the 4-dim middle into two 2-dim parts."""
        U1, V1, W1 = _strassen_222()  # 2x2x2, rank 7
        U2, V2, W2 = _strassen_222()  # 2x2x2, rank 7
        
        # For 2x2 @ 2x3, use schoolbook
        n3, m3, p3 = 2, 2, 3
        U3 = np.zeros((n3*m3, n3*m3*p3), dtype=float)
        V3 = np.zeros((m3*p3, n3*m3*p3), dtype=float)
        W3 = np.zeros((p3*n3, n3*m3*p3), dtype=float)
        col3 = 0
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    U3[i*m3+j, col3] = 1
                    V3[j*p3+k, col3] = 1
                    W3[k*n3+i, col3] = 1
                    col3 += 1
        
        # Build 2x2 @ 2x7
        total_rank_227 = U1.shape[1] + U2.shape[1] + U3.shape[1]
        U_227 = np.zeros((2*2, total_rank_227), dtype=float)
        V_227 = np.zeros((2*7, total_rank_227), dtype=float)
        W_227 = np.zeros((7*2, total_rank_227), dtype=float)
        
        col = 0
        # Block 1: B cols 0-1
        rank1 = U1.shape[1]
        U_227[:, col:col+rank1] = U1
        V_227[:2*2, col:col+rank1] = V1  # V1 is (4,7), but we need (2*2, rank) for j=0-1, k=0-1
        # Actually V1[j*2+k, r] for j in 0-1, k in 0-6 -> but we only want k=0-1 for block 1
        # This is getting messy. Let me redo this more carefully.
        
        # Let's just use the full schoolbook for the non-Strassen parts and be done.
        # The Strassen block decomposition is complex to implement correctly.
        # Let's focus on ALS instead.
        return None
    
    # Try ALS with various ranks
    for rank in range(30, 57):
        if time.time() > deadline:
            break
        
        for trial in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + rank + trial)
            result = _als_decompose(n, m, p, rank, max_iter=50, tol=1e-7, deadline=deadline)
            if result is not None:
                U_try, V_try, W_try = result
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                        print(f"ALS found valid decomposition with rank {rank}")
                        break  # Move to next rank
    
    return best_result
