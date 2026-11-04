import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

def _build_rank33():
    """Return (U, V, W) as lists for the known rank-33 Strassen solution."""
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for k in range(p):
        U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
        U[3][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
    for x in range(3):
        cs = 2*x
        U[1][col]=1;U[5][col]=1;V[6+cs][col]=1;V[12+cs+1][col]=1;W[2*cs][col]=1;W[2*cs+3][col]=1;col+=1
        U[4][col]=1;U[5][col]=1;V[6+cs][col]=1;W[2*cs+1][col]=1;W[2*cs+3][col]=-1;col+=1
        U[1][col]=1;V[6+cs+1][col]=1;V[12+cs+1][col]=-1;W[2*cs+2][col]=1;W[2*cs+3][col]=1;col+=1
        U[5][col]=1;V[12+cs][col]=1;V[6+cs][col]=-1;W[2*cs][col]=1;W[2*cs+1][col]=1;col+=1
        U[1][col]=1;U[2][col]=1;V[12+cs+1][col]=1;W[2*cs][col]=-1;W[2*cs+2][col]=1;col+=1
        U[4][col]=1;U[1][col]=-1;V[6+cs][col]=1;V[6+cs+1][col]=1;W[2*cs+3][col]=1;col+=1
        U[2][col]=1;U[5][col]=-1;V[12+cs][col]=1;V[12+cs+1][col]=1;W[2*cs][col]=1;col+=1
    return U, V, W

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the 2x3*3x6 multiplication tensor."""
    n, m, p = 2, 3, 6
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T_est = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    T_target = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T_target[r*m+s, s*p+c, c*n+r] = 1
    return np.max(np.abs(T_est - T_target)) < 1e-9

def _find_2x3x3_rank9(deadline):
    """
    Find a rank-9 decomposition of the 2x3 * 3x3 -> 2x3 multiplication tensor.
    
    The tensor T has T[i, s, j] = 1 if the (i,j) entry of the product depends on A[i,s]*B[s,j].
    More precisely, T[A_idx, B_idx, C_idx] = 1 where:
      A_idx = i*3 + s (i in 0..1, s in 0..2)
      B_idx = s*3 + j (s in 0..2, j in 0..2)  
      C_idx = j*2 + i (j in 0..2, i in 0..1)
    
    We seek U (6 x r), V (9 x r), W (6 x r) such that
      sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    Known: rank of 2x3x3 is 9.
    """
    n_sub, m_sub, p_sub = 2, 3, 3
    r = 9
    U = np.zeros((n_sub * m_sub, r))
    V = np.zeros((m_sub * p_sub, r))
    W = np.zeros((p_sub * n_sub, r))
    
    # Construct a rank-9 decomposition.
    # For each column j of B (j=0,1,2), we compute A * B[:,j] -> 2x1 vector.
    # This is a 2x3 * 3x1 -> 2x1 multiplication with rank 3 (one per row of A).
    # But we need to be more careful.
    
    # Let's use the following approach:
    # The product C[i,j] = sum_s A[i,s] * B[s,j]
    # We can write this as 9 rank-1 terms, one for each (i,s) pair, but that's 6 terms.
    # Wait, we need to cover all (i,s,j) triples.
    
    # Actually, the naive decomposition has 2*3*3 = 18 terms.
    # To get rank 9, we need a clever decomposition.
    
    # One known rank-9 decomposition for 2x3x3:
    # Use the fact that we can factor through the middle dimension.
    
    # Let me try: for each s in 0..2, we have A[:,s] (2x1) and B[s,:] (1x3).
    # Their outer product gives a 2x3 matrix, and C = sum_s A[:,s] * B[s,:]
    # This is a sum of 3 rank-1 matrices, but in the tensor sense, each term
    # A[:,s] * B[s,:] is a 2x3 matrix, not a scalar.
    
    # In the tensor decomposition framework, we need scalar rank-1 terms.
    # Each scalar term u_a * v_b * w_c contributes to T[a,b,c].
    
    # Let me just use an optimization-based approach to find the rank-9 decomposition.
    
    # Initialize randomly
    np.random.seed(42)
    U = np.random.randn(n_sub * m_sub, r)
    V = np.random.randn(m_sub * p_sub, r)
    W = np.random.randn(p_sub * n_sub, r)
    
    # Build target tensor
    T_target = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for s in range(m_sub):
            for j in range(p_sub):
                a_idx = i * m_sub + s
                b_idx = s * p_sub + j
                c_idx = j * n_sub + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    
    # Use alternating least squares to minimize ||T_est - T_target||^2
    # where T_est[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    best_cost = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(2000):
        if time.time() > deadline - 20:
            break
        
        # Update U
        # For each row i of U, minimize sum_j,k (T_target[i,j,k] - sum_r U[i,r]*V[j,r]*W[k,r])^2
        # This is a linear system: for fixed i, we want U[i,:] such that
        # sum_r U[i,r] * (V[:,r] * W[:,r])^T approx T_target[i,:,:]
        # Let M_i be a (m_sub*p_sub)*(p_sub*n_sub) matrix where M_i[j,k] = sum_r V[j,r]*W[k,r]
        # But we want to solve for U[i,:], so we can write:
        # T_target[i,:,:] = sum_r U[i,r] * (V[:,r] outer W[:,r])
        # This is a linear system in U[i,:].
        
        # More efficiently: for each r, the contribution to T_target[i,:,:] is U[i,r] * (V[:,r] * W[:,r]^T)
        # So we can set up: T_target[i].flatten() = sum_r U[i,r] * (V[:,r] * W[:,r]^T).flatten()
        # This is a least squares problem.
        
        # Build the design matrix
        # For each r, feature_r[j,k] = V[j,r] * W[k,r]
        # We want to find U[i,:] such that sum_r U[i,r] * feature_r = T_target[i]
        
        # This is a linear system: A_r * U[i,r] = T_target[i] where A_r are the features.
        # Specifically, if we flatten: sum_r U[i,r] * (V[:,r] * W[:,r]^T).flatten() = T_target[i].flatten()
        
        # Let's build the matrix B where B[:,r] = (V[:,r] * W[:,r]^T).flatten()
        # Then we solve B^T * U[i,:] = T_target[i].flatten()... no.
        # Actually: T_target[i].flatten() = sum_r U[i,r] * B[:,r]
        # So: T_target[i].flatten() = B @ U[i,:]  (where B is (m*p * p*n) x r)
        # Wait, B[:,r] is a vector of length m*p * p*n, and U[i,:] is length r.
        # So T_target[i].flatten() = B @ U[i,:] means B is (m*p*p*n) x r and U[i,:] is r x 1.
        # But that's not quite right dimensionally. Let me re-derive.
        
        # T_target[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For fixed i: T_target[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r]^T)
        # Let P_r = V[:,r] * W[:,r]^T (a (m*p) x (p*n) matrix)
        # Then T_target[i] = sum_r U[i,r] * P_r
        # Flatten: T_target[i].flatten() = sum_r U[i,r] * P_r.flatten()
        # So if we let Q be a (m*p*p*n) x r matrix with Q[:,r] = P_r.flatten(),
        # then T_target[i].flatten() = Q @ U[i,:]
        # We solve: U[i,:] = (Q^T Q)^{-1} Q^T T_target[i].flatten()
        
        # This is expensive. Let's use a simpler approach: coordinate descent or just
        # do a few iterations of ALS.
        
        # Simpler: use the formula for ALS in tensor decomposition.
        # For U: U[i,r] = sum_{j,k} T_target[i,j,k] * V[j,r] * W[k,r] / (sum_{j,k} V[j,r]^2 * W[k,r]^2)
        # This is the standard ALS update for CP decomposition.
        
        # Update U
        for i in range(n_sub * m_sub):
            for r in range(r):
                num = np.sum(T_target[i, :, :] * V[:, r] * W[:, r])
                den = np.sum(V[:, r] ** 2 * W[:, r] ** 2)
                if den > 1e-10:
                    U[i, r] = num / den
        
        # Update V
        for j in range(m_sub * p_sub):
            for r in range(r):
                num = np.sum(T_target[:, j, :] * U[:, r] * W[:, r])
                den = np.sum(U[:, r] ** 2 * W[:, r] ** 2)
                if den > 1e-10:
                    V[j, r] = num / den
        
        # Update W
        for k in range(p_sub * n_sub):
            for r in range(r):
                num = np.sum(T_target[:, :, k] * U[:, r] * V[:, r])
                den = np.sum(U[:, r] ** 2 * V[:, r] ** 2)
                if den > 1e-10:
                    W[k, r] = num / den
        
        # Compute cost
        T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        cost = np.sum((T_est - T_target) ** 2)
        if cost < best_cost:
            best_cost = cost
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if cost < 1e-12:
            break
    
    # Check if we found a valid decomposition
    T_est = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
    if np.max(np.abs(T_est - T_target)) < 1e-9:
        return best_U, best_V, best_W, r
    
    # If not, try with more iterations or different initialization
    return None, None, None, r

def _try_small_rank_search(deadline):
    """
    Attempt to find a rank < 33 decomposition using sub-tensor decomposition.
    Split 2x3*3x6 into two 2x3*3x3 sub-problems.
    """
    n, m, p = 2, 3, 6
    
    # Try to find rank-9 decomposition of 2x3x3
    U_sub, V_sub, W_sub, r_sub = _find_2x3x3_rank9(deadline)
    
    if U_sub is not None:
        # Combine two sub-decompositions
        # First sub-problem: A (2x3) * B1 (3x3) -> C1 (2x3)
        # B1 is columns 0-2 of B, C1 is columns 0-2 of C
        # Second sub-problem: A (2x3) * B2 (3x3) -> C2 (2x3)
        # B2 is columns 3-5 of B, C2 is columns 3-5 of C
        
        total_rank = 2 * r_sub
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # First sub-problem (columns 0-2)
        for col in range(r_sub):
            for i in range(n * m):
                U[i][col] = U_sub[i, col]
            for s in range(m):
                for j in range(3):
                    V[s * p + j][col] = V_sub[s * 3 + j, col]
            for j in range(3):
                for i in range(n):
                    W[j * n + i][col] = W_sub[j * n + i, col]
        
        # Second sub-problem (columns 3-5)
        for col in range(r_sub):
            for i in range(n * m):
                U[i][r_sub + col] = U_sub[i, col]
            for s in range(m):
                for j in range(3):
                    V[s * p + (3 + j)][r_sub + col] = V_sub[s * 3 + j, col]
            for j in range(3):
                for i in range(n):
                    W[(3 + j) * n + i][r_sub + col] = W_sub[j * n + i, col]
        
        # Verify
        if _verify(U, V, W):
            return U, V, W, total_rank
    
    # Fallback to rank 33
    return _build_rank33(), 33

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # First, get the known rank-33 solution
    U, V, W = _build_rank33()
    best_rank = 33
    
    # Verify the solution
    if not _verify(U, V, W):
        # Should not happen, but fallback to naive
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for r in range(n):
            for s in range(m):
                for c in range(p):
                    U[r*m+s][col] = 1
                    V[s*p+c][col] = 1
                    W[c*n+r][col] = 1
                    col += 1
        best_rank = n*m*p
    
    # Try to improve with sub-tensor search
    try:
        U2, V2, W2, rank2 = _try_small_rank_search(deadline)
        if U2 is not None and rank2 < best_rank and _verify(U2, V2, W2):
            U, V, W = U2, V2, W2
            best_rank = rank2
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
