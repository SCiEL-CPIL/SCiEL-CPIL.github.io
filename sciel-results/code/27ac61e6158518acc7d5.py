import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _schoolbook(n, m, p):
    """Return the standard schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    # Reshape to 3D for einsum
    # U_3d: (n, m, R)
    # V_3d: (m, p, R)
    # W_3d: (p, n, R)
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    
    # Target is 1 for all i, j, k
    # S[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
    # Note: W index is k*n + i. In W_3d, this is W_3d[k, i, r].
    # So we want: S[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    
    # Einsum: 'ijr,jkr,kir->ijk'
    S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(S, np.ones((n, m, p)), atol=tol):
        return False
    
    # Check half-integer property
    # The task says "integer/half-integer entries".
    # This usually means 2*x is an integer.
    if not np.allclose(U * 2, np.round(U * 2), atol=1e-8):
        return False
    if not np.allclose(V * 2, np.round(V * 2), atol=1e-8):
        return False
    if not np.allclose(W * 2, np.round(W * 2), atol=1e-8):
        return False
        
    return True

def _als_step(U, V, W, n, m, p, deadline):
    """
    Perform one step of Alternating Least Squares.
    U, V, W are 2D arrays (dim, R).
    """
    R = U.shape[1]
    
    # 1. Update U
    # For each (i, j), we want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
    # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product of vectors of length R)
    # We want U[i*m+j, :] @ A_k = 1 for all k.
    # This is a linear system: M @ u = b, where M is p x R, u is R x 1, b is p x 1.
    # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
    # b = ones(p)
    
    U_new = U.copy()
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((p, R))
            for k in range(p):
                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                # Solve M @ u = 1
                # M is p x R. We want u (R x 1).
                # If p >= R, we can use lstsq.
                # If p < R, it's underdetermined. lstsq gives min-norm solution.
                U_new[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
    
    U = U_new
    
    # 2. Update V
    # For each (j, k), we want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
    # Let B_i = U[i*m+j, :] * W[k*n+i, :]
    # We want V[j*p+k, :] @ B_i = 1 for all i.
    # M is n x R.
    
    V_new = V.copy()
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((n, R))
            for i in range(n):
                M[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
                
    V = V_new
    
    # 3. Update W
    # For each (k, i), we want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
    # Let C_j = U[i*m+j, :] * V[j*p+k, :]
    # We want W[k*n+i, :] @ C_j = 1 for all j.
    # M is m x R.
    
    W_new = W.copy()
    for k in range(p):
        for i in range(n):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((m, R))
            for j in range(m):
                M[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
                
    W = W_new
    
    return U, V, W

def _try_rank(n, m, p, rank, seed, deadline):
    """
    Try to find a rank-r decomposition using ALS.
    Returns (U, V, W) if successful, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Scale down to avoid overflow/instability initially?
    # The target is 1. If R is large, random sum might be large.
    # Let's just run ALS.
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    max_iters = 200
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        U, V, W = _als_step(U, V, W, n, m, p, deadline)
        
        # Calculate error
        # S = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
        # err = np.sum((S - 1)**2)
        
        # Efficient error calculation
        # We can just check if it's close to 1.
        # But for convergence tracking, let's compute.
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.sum((S - 1.0)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-8:
            # Found a solution. Now try to polish it to half-integers.
            break
            
    if best_err < 1e-6:
        # Try to snap to half-integers
        # The solution might be very close to half-integers.
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        
        if _verify(U_round, V_round, W_round, n, m, p, tol=1e-8):
            return U_round, V_round, W_round
        
        # If snapping fails, maybe the solution isn't exactly half-integer yet.
        # Try a few more iterations or a different initialization?
        # For now, if it's not half-integer, we can't use it directly if the requirement is strict.
        # The task says "integer/half-integer entries".
        # If we can't snap, this rank might not have a half-integer solution, or we need better precision.
        # Let's check if the non-snapped solution is valid (it won't be if the check requires half-integers).
        # The verify function checks half-integers.
        
        # Perhaps we can try to refine the snapped values?
        # Or just return None and try another seed/rank.
        
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    
    # Known lower bounds:
    # Rank >= n*m = 9
    # Rank >= m*p = 18
    # Rank >= p*n = 18
    # So min rank is at least 18.
    
    # Schoolbook is 54.
    # Let's try ranks from 20 up to 53.
    # We want the lowest rank.
    
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try a few specific ranks first, then a range.
    # 3x3x3 is 19.
    # 3x3x6 is likely related.
    # Maybe 2 * 19 = 38? Or 3x3x6 can be done in 3x3x3 + 3x3x3 = 38?
    # If we split B into two 3x3 matrices, we can use two 3x3x3 decompositions.
    # That gives rank 38.
    # Can we do better?
    
    # Let's try to construct a rank 38 solution explicitly from two 3x3x3 solutions if we can find one.
    # But we don't have a hardcoded 3x3x3 solution here.
    
    # Let's search.
    # Start from lower ranks.
    # 18 is the lower bound.
    # Try 18, 19, 20, ...
    
    # To save time, we can jump.
    # Let's try 38 first, as it's a plausible candidate (2 * 19).
    # Then try lower.
    
    ranks_to_try = [38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple seeds
        for seed in range(50):
            if time.time() > deadline:
                break
                
            U, V, W = _try_rank(n, m, p, R, seed, deadline)
            
            if U is not None:
                best_rank = R
                best_U, best_V, best_W = U, V, W
                # If we found a solution, we can stop searching for this rank.
                # But we want the MINIMAL rank.
                # So we should break out of the seed loop and continue to next (lower) rank.
                break
                
        # If we found a solution for this rank, we continue to lower ranks.
        # If not, we continue to lower ranks.
        
    # Convert to lists
    # Ensure they are lists of lists of floats (or ints if integer)
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# Helper to run the program
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
