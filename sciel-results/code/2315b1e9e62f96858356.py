import time
import numpy as np
from scipy.optimize import minimize
import itertools
import random

n, m, p = 4, 4, 8
TOL = 1e-9
TIME_LIMIT = 145.0


def verify(U, V, W, n=4, m=4, p=8, tol=TOL):
    """
    Check if U,V,W is a valid rank-R tensor decomposition for matrix multiplication.
    
    U: (n*m, R) - left factor
    V: (m*p, R) - middle factor
    W: (p*n, R) - right factor
    
    For each (i,j,k) we need:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    R = U.shape[1]
    # Build the three mode-3 unfoldings
    # For each r, we form:
    #   u_r = U[:,r] reshaped as (n,m)
    #   v_r = V[:,r] reshaped as (m,p)
    #   w_r = W[:,r] reshaped as (p,n)
    # The tensor T = sum_r u_r * v_r * w_r (outer product)
    # T[i,j,k] should equal 1 for all i,j,k
    
    # Efficient check: for each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We can vectorize this
    
    # For each pair (i,j), we need a vector of length p:
    #   For fixed (i,j), result[k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Let's build it as:
    # A = U reshaped to (n, m, R)
    # B = V reshaped to (m, p, R)
    # C = W reshaped to (p, n, R)
    
    A = U.reshape(n, m, R)
    B = V.reshape(m, p, R)
    C = W.reshape(p, n, R)
    
    # T[i,j,k] = sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
    # This is a contraction over r
    
    # For each i, j: 
    #   T[i,j,:] = sum_r A[i,j,r] * (B[j,:,r] * C[:,i,r])  # element-wise over k, then sum over r
    # B[j,:,r] is a vector of length p, C[:,i,r] is a vector of length p
    # So for fixed (i,j): T[i,j,k] = sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
    
    # We can compute this as:
    # For fixed i, j: 
    #   Let a = A[i,j,:]  (length R)
    #   Let B_j = B[j,:,:]  (p x R)
    #   Let C_i = C[:,i,:]  (p x R)
    #   T[i,j,:] = sum_k... wait, for each k: sum_r a[r] * B_j[k,r] * C_i[k,r]
    #   = a^T @ (B_j * C_i)  where * is element-wise, and @ is the column sum
    
    # Actually: T[i,j,:] = for each k, sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
    # Let M[i,j] = A[i,j,:] (vector of length R)
    # Let N[j,k] = B[j,k,:] (vector of length R)
    # Let P[k,i] = C[k,i,:] (vector of length R)
    
    # T[i,j,k] = M[i,j] . (N[j,k] * P[k,i])  where * is element-wise and . is dot
    
    # We can compute:
    # For each i, j:
    #   T[i,j,:] = sum over k? No, for each k independently:
    #   T[i,j,k] = np.dot(A[i,j,:], B[j,k,:] * C[k,i,:])
    
    # Vectorized: 
    # Create a 3D array D where D[j,k,i] = sum_r B[j,k,r] * C[k,i,r]
    # D[j,k,i] = np.dot(B[j,k,:], C[k,i,:])
    # Then T[i,j,k] = sum_r A[i,j,r] * (B[j,k,r] * C[k,i,r])  -- wait, that's not right.
    
    # T[i,j,k] = sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
    
    # Let E[i,j,k,r] = A[i,j,r] * B[j,k,r] * C[k,i,r], then T[i,j,k] = sum_r E[i,j,k,r]
    
    # We can compute:
    # For each r:
    #   u_r = A[:,:,r]  (n x m)
    #   v_r = B[:,:,r]  (m x p)
    #   w_r = C[:,:,r]  (p x n)
    #   T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
    #   T = sum_r T_r
    
    # This is equivalent to:
    # T[i,j,k] = sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
    
    # Vectorized approach:
    # For fixed (i,j):
    #   T[i,j,:] = sum_r A[i,j,r] * (B[j,:,r] * C[:,i,r])
    #   = np.sum(A[i,j,:] * (B[j,:,:] * C[:,i,:]), axis=1)
    # Wait, B[j,:,:] is (p,R), C[:,i,:] is (p,R), element-wise product is (p,R)
    # Then multiply by A[i,j,:] which is (R,), and sum over axis=1 (the R axis)
    
    # Actually:
    # T[i,j,k] = sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
    # For fixed (i,j,k): this is a scalar
    # For fixed (i,j): T[i,j,:] is a vector of length p
    
    # Vectorized:
    # B[j,:,:] * C[:,i,:]  -> (p, R)
    # Multiply by A[i,j,:] -> (1, R) broadcast -> (p, R)
    # Sum over axis=1 -> (p,)
    
    # So: T[i,j,:] = np.sum(A[i,j, None, :] * (B[j, :, :] * C[:, i, :]), axis=1)
    # where A[i,j, None, :] is (1, R)
    
    # Let's verify shapes:
    # A[i,j,:] has shape (R,)
    # A[i,j, None, :] has shape (1, R)
    # B[j,:,:] has shape (p, R)
    # C[:,i,:] has shape (p, R)
    # B[j,:,:] * C[:,i,:] has shape (p, R)
    # A[i,j, None, :] * (B[j,:,:] * C[:,i,:]) has shape (p, R)
    # np.sum(..., axis=1) has shape (p,)
    # Yes!
    
    # Build T:
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            T[i, j, :] = np.sum(A[i, j, None, :] * (B[j, :, :] * C[:, i, :]), axis=1)
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True


def verify_fast(U, V, W, n=4, m=4, p=8, tol=TOL):
    """Fast verification using tensor contraction."""
    R = U.shape[1]
    A = U.reshape(n, m, R)
    B = V.reshape(m, p, R)
    C = W.reshape(p, n, R)
    
    # T[i,j,k] = sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', A, B, C)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True


def schoolbook(n=4, m=4, p=8):
    """Schoolbook decomposition with rank n*m*p."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def strassen_2x2():
    """Strassen's algorithm for 2x2x2, rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def kron_product(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    n1, r1 = U1.shape
    m1, _ = V1.shape
    p1, _ = W1.shape
    n2, r2 = U2.shape
    m2, _ = V2.shape
    p2, _ = W2.shape
    
    R = r1 * r2
    U = np.zeros((n1 * n2, R))
    V = np.zeros((m1 * m2, R))
    W = np.zeros((p1 * p2, R))
    
    for a in range(r1):
        for b in range(r2):
            r = a * r2 + b
            U[:, r] = np.kron(U1[:, a], U2[:, b])
            V[:, r] = np.kron(V1[:, a], V2[:, b])
            W[:, r] = np.kron(W1[:, a], W2[:, b])
    
    return U, V, W


def direct_sum(U1, V1, W1):
    """Direct sum: doubles m and p dimensions."""
    n1, r = U1.shape
    m1, _ = V1.shape
    p1, _ = W1.shape
    
    R = 2 * r
    U = np.zeros((n1, R))
    V = np.zeros((2 * m1, R))
    W = np.zeros((2 * p1, R))
    
    U[:, :r] = U1
    V[:m1, :r] = V1
    W[:p1, :r] = W1
    
    U[:, r:] = U1
    V[m1:, r:] = V1
    W[p1:, r:] = W1
    
    return U, V, W


def build_2x2x4():
    """2x2 times 2x4 = 2x4, rank 14 (two Strassen blocks)."""
    Us, Vs, Ws = strassen_2x2()
    return direct_sum(Us, Vs, Ws)


def build_4x4x8_kron():
    """4x4x8 via Kronecker of 2x2x4 and 2x2x2."""
    U224, V224, W224 = build_2x2x4()
    U222, V222, W222 = strassen_2x2()
    return kron_product(U224, V224, W224, U222, V222, W222)


def build_4x4x4_strassen_iterated():
    """4x4x4 via iterated Strassen: (2x2x2) kron (2x2x2), rank 49."""
    U2, V2, W2 = strassen_2x2()
    return kron_product(U2, V2, W2, U2, V2, W2)


def build_4x4x8_from_444():
    """4x4x8 from 4x4x4 by direct sum (doubling p)."""
    U444, V444, W444 = build_4x4x4_strassen_iterated()
    return direct_sum(U444, V444, W444)


def random_decomposition(R, n=4, m=4, p=8, seed=0):
    """Generate a random rank-R decomposition."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n * m, R))
    V = rng.normal(size=(m * p, R))
    W = rng.normal(size=(p * n, R))
    return U, V, W


def try_remove_column(U, V, W, idx, n, m, p):
    """Try removing column idx; return new (U,V,W) or None."""
    R = U.shape[1]
    if R <= 1:
        return None
    
    mask = np.ones(R, dtype=bool)
    mask[idx] = False
    U2 = U[:, mask]
    V2 = V[:, mask]
    W2 = W[:, mask]
    
    if verify_fast(U2, V2, W2, n, m, p):
        return U2, V2, W2
    return None


def greedy_reduce(U, V, W, n, m, p, time_limit=30):
    """Greedy column removal."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    start = time.time()
    
    changed = True
    while changed:
        changed = False
        R = U.shape[1]
        if R <= 1 or time.time() - start > time_limit:
            break
        
        # Score by contribution norm
        scores = np.zeros(R)
        for r in range(R):
            scores[r] = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
        
        order = np.argsort(scores)
        
        for col in order:
            if time.time() - start > time_limit:
                break
            result = try_remove_column(U, V, W, col, n, m, p)
            if result is not None:
                U, V, W = result
                changed = True
                break
    
    return U, V, W


def pair_merge(U, V, W, n, m, p, time_limit=30):
    """Try merging pairs of columns via linear combinations."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    start = time.time()
    rng = np.random.default_rng(42)
    
    improved = True
    while improved:
        improved = False
        R = U.shape[1]
        if R <= 2 or time.time() - start > time_limit:
            break
        
        # Try all pairs
        found = False
        for i in range(R):
            if found or time.time() - start > time_limit:
                break
            for j in range(R):
                if found or time.time() - start > time_limit:
                    break
                if i == j:
                    continue
                
                # Try U[:,j] += alpha * U[:,i], then remove column i
                # The idea: if there's a linear dependency, we can merge
                # We need: for all (a,b,c):
                #   U[a,i]*V[b,i]*W[c,i] + alpha*U[a,j]*V[b,j]*W[c,j] = 0  (after removing i)
                # This is hard to check directly. Instead, just try a few alpha values.
                
                for alpha in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
                    U_try = U.copy()
                    V_try = V.copy()
                    W_try = W.copy()
                    
                    U_try[:, j] = U_try[:, i] + alpha * U_try[:, j]
                    V_try[:, j] = V_try[:, i] + alpha * V_try[:, j]
                    W_try[:, j] = W_try[:, i] + alpha * W_try[:, j]
                    
                    mask = np.ones(R, dtype=bool)
                    mask[i] = False
                    U_try2 = U_try[:, mask]
                    V_try2 = V_try[:, mask]
                    W_try2 = W_try[:, mask]
                    
                    if verify_fast(U_try2, V_try2, W_try2, n, m, p):
                        U, V, W = U_try2, V_try2, W_try2
                        improved = True
                        found = True
                        break
    
    return U, V, W


def random_restart_reduce(U, V, W, n, m, p, time_limit=30):
    """Randomized reduction with multiple restarts."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    start = time.time()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 1: Greedy
    U_g, V_g, W_g = greedy_reduce(U, V, W, n, m, p, min(10, time_limit * 0.3))
    if U_g.shape[1] < best_rank:
        best_rank = U_g.shape[1]
        best_U, best_V, best_W = U_g.copy(), V_g.copy(), W_g.copy()
    
    # Phase 2: Pair merge
    remaining = time_limit - (time.time() - start)
    if remaining > 10 and best_rank > 2:
        U_m, V_m, W_m = pair_merge(best_U, best_V, best_W, n, m, p, min(10, remaining * 0.5))
        if U_m.shape[1] < best_rank:
            best_rank = U_m.shape[1]
            best_U, best_V, best_W = U_m.copy(), V_m.copy(), W_m.copy()
    
    # Phase 3: Randomized pair merges
    remaining = time_limit - (time.time() - start)
    if remaining > 5 and best_rank > 2:
        rng = np.random.default_rng(123)
        U_r = best_U.copy()
        V_r = best_V.copy()
        W_r = best_W.copy()
        
        max_iters = int(remaining * 2)
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            R_cur = U_r.shape[1]
            if R_cur <= 2:
                break
            
            i, j = rng.choice(R_cur, size=2, replace=False)
            alpha = rng.uniform(-3, 3)
            
            U_try = U_r.copy()
            V_try = V_r.copy()
            W_try = W_r.copy()
            
            U_try[:, j] = U_try[:, i] + alpha * U_try[:, j]
            V_try[:, j] = V_try[:, i] + alpha * V_try[:, j]
            W_try[:, j] = W_try[:, i] + alpha * W_try[:, j]
            
            mask = np.ones(R_cur, dtype=bool)
            mask[i] = False
            U_try2 = U_try[:, mask]
            V_try2 = V_try[:, mask]
            W_try2 = W_try[:, mask]
            
            if verify_fast(U_try2, V_try2, W_try2, n, m, p):
                U_r, V_r, W_r = U_try2, V_try2, W_try2
                if U_r.shape[1] < best_rank:
                    best_rank = U_r.shape[1]
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
    
    return best_U, best_V, best_W


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return deadline - time.time()
    
    # Start with multiple initial decompositions
    candidates = []
    
    # 1. Schoolbook (rank 128)
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    candidates.append((U_sb, V_sb, W_sb, 128))
    
    # 2. Kronecker: 2x2x4 * 2x2x2 (rank 14*7 = 98)
    if remaining() > 5:
        U_k, V_k, W_k = build_4x4x8_kron()
        if verify_fast(U_k, V_k, W_k, n, m, p):
            candidates.append((U_k, V_k, W_k, U_k.shape[1]))
    
    # 3. From 4x4x4 Strassen (rank 49) + direct sum (rank 98)
    if remaining() > 5:
        U_444, V_444, W_444 = build_4x4x4_strassen_iterated()
        if verify_fast(U_444, V_444, W_444, n, m, n):
            U_ds, V_ds, W_ds = direct_sum(U_444, V_444, W_444)
            if verify_fast(U_ds, V_ds, W_ds, n, m, p):
                candidates.append((U_ds, V_ds, W_ds, U_ds.shape[1]))
    
    # Pick the best initial candidate (lowest rank)
    candidates.sort(key=lambda x: x[3])
    
    if not candidates:
        # Should not happen
        U_s, V_s, W_s = schoolbook(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_U, best_V, best_W = candidates[0][0].copy(), candidates[0][1].copy(), candidates[0][2].copy()
    best_rank = candidates[0][3]
    
    # Try reducing each candidate
    for idx, (U_c, V_c, W_c, rank_c) in enumerate(candidates):
        if remaining() < 20:
            break
        if rank_c >= best_rank and idx > 0:
            continue
        
        time_alloc = min(remaining() * 0.3, 40)
        if time_alloc < 10:
            break
        
        U_r, V_r, W_r = random_restart_reduce(U_c, V_c, W_c, n, m, p, time_alloc)
        
        if verify_fast(U_r, V_r, W_r, n, m, p):
            if U_r.shape[1] < best_rank:
                best_rank = U_r.shape[1]
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
    
    # Additional: try more aggressive search on the best
    if remaining() > 30 and best_rank > 50:
        U_ag, V_ag, W_ag = random_restart_reduce(best_U, best_V, best_W, n, m, p, min(remaining() * 0.4, 50))
        if verify_fast(U_ag, V_ag, W_ag, n, m, p):
            if U_ag.shape[1] < best_rank:
                best_rank = U_ag.shape[1]
                best_U, best_V, best_W = U_ag.copy(), V_ag.copy(), W_ag.copy()
    
    # Final verification
    if not verify_fast(best_U, best_V, best_W, n, m, p):
        U_s, V_s, W_s = schoolbook(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
