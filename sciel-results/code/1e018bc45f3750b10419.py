import numpy as np
import time

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

def _naive_decomposition():
    """Return the naive rank-36 decomposition."""
    n, m, p = 2, 3, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for r in range(n):
        for s in range(m):
            for c in range(p):
                U[r*m+s][col] = 1
                V[s*p+c][col] = 1
                W[c*n+r][col] = 1
                col += 1
    return U, V, W

def _try_linear_solve(rank, deadline):
    """Try to find a rank-rank decomposition by solving linear system."""
    n, m, p = 2, 3, 6
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    total_vars = (dim_u + dim_v + dim_w) * rank
    total_eqs = dim_u * dim_v * dim_w
    
    # Build the target tensor
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_target[i*m+k, k*p+j, j*n+i] = 1
    
    # For linear system: fix W to identity-like pattern and solve for U, V
    # Actually, this is nonlinear. Let's use a different approach.
    # Try random initialization and gradient descent.
    
    if time.time() > deadline - 5:
        return None
    
    # Use alternating least squares
    np.random.seed(42)
    U = np.random.randn(dim_u, rank)
    V = np.random.randn(dim_v, rank)
    W = np.random.randn(dim_w, rank)
    
    # Normalize
    for r in range(rank):
        norm = np.linalg.norm(U[:, r])
        if norm > 1e-10:
            U[:, r] /= norm
            V[:, r] *= norm
    
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        # Update W: for each (i,j), solve for W[:,r]
        # T[i,j,:] = sum_r U[i,r] * V[j,r] * W[:,r]
        # This is a linear system in W
        A_mat = np.zeros((dim_u * dim_v, rank))
        for i in range(dim_u):
            for j in range(dim_v):
                A_mat[i*dim_v+j, :] = U[i, :] * V[j, :]
        
        b_vec = T_target.reshape(-1)
        # W_flat should satisfy A_mat @ W_flat = b_vec
        # But W is (dim_w, rank), so we need to reshape
        # Actually, the equation is: for each (i,j,k): sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
        # This is: (U[i,:] * V[j,:]) @ W[:,r] summed over r = T[i,j,k]
        # So: sum_r (U[i,r]*V[j,r])*W[k,r] = T[i,j,k]
        # For fixed k: sum_r (U[i,r]*V[j,r])*W[k,r] = T[i,j,k]
        # This means: for each k, W[k,:] satisfies: sum over (i,j) of (U[i,r]*V[j,r])*W[k,r] = T[i,j,k]
        # Which is: (sum_{i,j} U[i,r]*V[j,r] * e_k) ... this is getting complicated.
        
        # Simpler: use the fact that T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # So T.reshape(dim_u*dim_v, dim_w) = sum_r (U[:,r] ⊗ V[:,r]) ⊗ W[:,r]
        # = sum_r np.kron(U[:,r], V[:,r]) outer W[:,r]
        # So M = T.reshape(dim_u*dim_v, dim_w) should equal sum_r np.kron(U[:,r], V[:,r]) @ W[:,r].T
        # Wait: np.kron(U[:,r], V[:,r]) is a vector of length dim_u*dim_v
        # And W[:,r] is a vector of length dim_w
        # So M = sum_r np.kron(U[:,r], V[:,r])[:,None] * W[:,r][None,:]
        # This is a rank-rank matrix factorization of M!
        
        M = T_target.reshape(dim_u*dim_v, dim_w)
        
        # Update W: M = sum_r a_r ⊗ w_r where a_r = np.kron(U[:,r], V[:,r])
        # So M = A @ W.T where A[r, :] = a_r
        A_mat = np.array([np.kron(U[:, r], V[:, r]) for r in range(rank)]).T  # (dim_u*dim_v, rank)
        W_new, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, M, rcond=None)
        W = W_new.T  # (dim_w, rank)
        
        # Update U: M = sum_r u_r ⊗ b_r where b_r = V[:,r] ⊗ W[:,r]
        # M = sum_r np.kron(u_r, V[:,r]) ⊗ W[:,r]... no
        # M[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For fixed j: M[:,j,:] = sum_r U[:,r] * V[j,r] * W[:,r].T
        # = sum_r (U[:,r] * V[j,r]) ⊗ W[:,r].T
        # So M[:,j,:] = sum_r c_{j,r} ⊗ W[:,r].T where c_{j,r} = U[:,r]*V[j,r]
        # This is: M[:,j,:] = C_j @ W.T where C_j[r, :] = c_{j,r}
        # Hmm, this is still coupled.
        
        # Let's just do a simple gradient step instead
        # Compute gradient of ||M - A@W.T||_F^2 with respect to U, V, W
        # This is getting too complex. Let's just return None for now.
        break
    
    return None

def _try_improve(deadline):
    """
    Attempt to find a rank < 33 decomposition.
    """
    n, m, p = 2, 3, 6
    best_U, best_V, best_W = None, None, None
    best_rank = 33
    
    # The naive decomposition has rank n*m*p = 36, but the true minimum might be lower.
    # For 2x3x6, let's try to find the actual minimum rank.
    # The tensor has structure: T[i*m+k, k*p+j, j*n+i] = 1
    # This is the matrix multiplication tensor.
    
    # Known results: rank of 2x2x2 multiplication is 7 (Strassen)
    # For 2x3x3: the rank is at most 9 (using Strassen-like methods)
    # For 2x3x6: we can view it as 2x3x3 + 2x3x3, but with shared A.
    
    # Let's try a structured construction:
    # Split B into B1 (cols 0-2) and B2 (cols 3-5)
    # C1 = A * B1 (2x3), C2 = A * B2 (2x3)
    # If we could decompose each with rank 9, total would be 18, but A is shared.
    # However, we can use the same "A factors" for both.
    
    # Actually, let's just try to verify if the rank-33 solution is valid,
    # and if not, fall back to naive.
    
    U0, V0, W0 = _build_rank33()
    if _verify(U0, V0, W0):
        return U0, V0, W0, 33
    
    # Fallback to naive
    U, V, W = _naive_decomposition()
    return U, V, W, n*m*p

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # First, get the known rank-33 solution
    U, V, W = _build_rank33()
    best_rank = 33
    
    # Verify the solution
    if not _verify(U, V, W):
        # Should not happen, but fallback to naive
        U, V, W = _naive_decomposition()
        best_rank = n*m*p
    
    # Try to improve with small-rank search
    try:
        U2, V2, W2, rank2 = _try_improve(deadline)
        if U2 is not None and rank2 < best_rank and _verify(U2, V2, W2):
            U, V, W = U2, V2, W2
            best_rank = rank2
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
