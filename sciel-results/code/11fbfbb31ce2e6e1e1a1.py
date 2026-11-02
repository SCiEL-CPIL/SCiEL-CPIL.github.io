"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
DEADLINE_S = 110.0


def _build_tensor3(n, m, p):
    """Build the 3-mode multiplication tensor T[i,j,k] = 1 for all (i,j,k).
    
    The tensor represents: C[k,i] = sum_j A[i,j] * B[j,k]
    As a 3-tensor with modes (i,j,k), T[i,j,k] = 1 for all valid indices.
    """
    T = np.ones((n, m, p), dtype=np.float64)
    return T


def _validate_decomposition(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k*n+i,r] == delta_{j,j'}
    
    Actually the correct condition for matrix mult tensor:
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should reconstruct the tensor.
    
    The multiplication tensor T3[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    so we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
    """
    # For efficiency, check a subset if full check is too slow
    # Full check: for all i, j, k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    
    # Vectorized check
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is equivalent to: for each k, the matrix M_k[i,j] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # should equal 1 for all i,j
    
    for k in range(p):
        # W[k*n+i, r] for i in range(n), r in range(rank)
        Wk = W[k*n:(k+1)*n, :]  # shape (n, rank)
        # For each i: sum_r U[i*m+j,r] * V[j*p+k,r] * Wk[i,r]
        # = (U[:, j*p+k, :] * Wk[i, :]).sum(axis=1) for each j... 
        # Let's do it differently:
        # M[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For fixed k, this is a bilinear form
        # U_slice: (n*m, rank), V_slice: (m*p, rank), W_slice: (n, rank)
        # M[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W_slice[i, r]
        
        # Reshape U to (n, m, rank)
        U_reshaped = U.reshape(n, m, rank)
        # V[j*p+k, r] for j in range(m), r in range(rank)
        V_slice = V[np.arange(m) * p + k, :]  # shape (m, rank)
        
        # M[i,j] = sum_r U_reshaped[i,j,r] * V_slice[j,r] * W_slice[i,r]
        # = sum_r (U_reshaped[i,j,r] * W_slice[i,r]) * V_slice[j,r]
        # For each i: (U_reshaped[i,:,:] * W_slice[i,:]) @ V_slice.T  -> gives (m,) vector
        # M[i,:] = (U_reshaped[i] * W_slice[i]) @ V_slice.T
        
        M = np.zeros((n, m))
        for i in range(n):
            M[i] = (U_reshaped[i] * W_slice[i]) @ V_slice.T
        
        if not np.allclose(M, np.ones((n, m)), atol=1e-6):
            return False
    return True


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _try_peel(n, m, p, R, T3, deadline):
    """Attempt to find a rank-R decomposition via iterative rank-1 peeling.
    
    Strategy: repeatedly find a rank-1 component that best fits the residual,
    subtract it, and continue until R components are placed or residual is zero.
    """
    residual = T3.copy()
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r in range(R):
        if time.time() > deadline:
            return None
        if np.allclose(residual, 0, atol=1e-10):
            # Pad remaining with zeros
            break
        
        # Find best rank-1 approximation of residual
        # Use truncated SVD on matricization
        # Matricize residual as (n*m, p) for mode-3 unfolding... 
        # Actually, for the tensor T[i,j,k], a rank-1 component is a[i]*b[j]*c[k]
        # We want to find a,b,c maximizing sum_{i,j,k} residual[i,j,k]*a[i]*b[j]*c[k]
        
        # Use alternating least squares or just try the trivial: since residual 
        # starts as all-ones, and we're peeling, let's use a simple approach.
        
        # For the all-ones tensor, the best rank-1 is a=1, b=1, c=1
        # But after peeling, residual changes.
        
        # Simple approach: use the fact that for matrix mult tensor, 
        # we can peel off terms corresponding to specific (i,j,k) triples.
        # But that gives schoolbook rank.
        
        # Better: try to find if a lower rank suffices by checking if the
        # tensor has a low-rank structure.
        
        # For now, use a heuristic: if residual is all ones, use a=1,b=1,c=1
        if np.allclose(residual, 1, atol=1e-10):
            a = np.ones(n)
            b = np.ones(m)
            c = np.ones(p)
            # Scale: we want sum residual*a*b*c = n*m*p, and a*b*c=1, so coefficient=1
            coeff = 1.0
        else:
            # Find best rank-1 via power method on the tensor
            # Initialize
            a = np.ones(n) / n
            b = np.ones(m) / m
            c = np.ones(p) / p
            
            for _ in range(50):
                if time.time() > deadline:
                    return None
                # Update a: a[i] = sum_{j,k} residual[i,j,k]*b[j]*c[k]
                bc = np.einsum('j,k->jk', b, c)
                a = np.einsum('ijk,jk->i', residual, bc)
                a /= (np.linalg.norm(a) + 1e-15)
                
                # Update b: b[j] = sum_{i,k} residual[i,j,k]*a[i]*c[k]
                ac = np.einsum('i,k->ik', a, c)
                b = np.einsum('ijk,ik->j', residual, ac)
                b /= (np.linalg.norm(b) + 1e-15)
                
                # Update c: c[k] = sum_{i,j} residual[i,j,k]*a[i]*b[j]
                ab = np.einsum('i,j->ij', a, b)
                c = np.einsum('ijk,ij->k', residual, ab)
                c /= (np.linalg.norm(c) + 1e-15)
            
            coeff = np.einsum('ijk,i,j,k->', residual, a, b, c)
            if abs(coeff) < 1e-15:
                # Try a different initialization
                return None
        
        # Place this rank-1 component
        U[np.arange(n) * m + np.arange(n)[:, None] * 0, r] = 0  # placeholder
        # Actually: U[i*m+j, r] = a[i] * (something)
        # The rank-1 component is: a[i]*b[j]*c[k]
        # So U[i*m+j, r] = a[i]*b[j]... no wait.
        # U has shape (n*m, rank), indexed by i*m+j
        # V has shape (m*p, rank), indexed by j*p+k  
        # W has shape (p*n, rank), indexed by k*n+i
        # The product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal a[i]*b[j]*c[k]*coeff
        # So we can set: U[i*m+j,r] = a[i]*b[j], V[j*p+k,r] = c[k], W[k*n+i,r] = coeff
        # Or distribute the factors differently.
        
        # Let's set: U[i*m+j,r] = a[i]*b[j], V[j*p+k,r] = 1, W[k*n+i,r] = coeff*c[k]
        # Then product = a[i]*b[j]*1*coeff*c[k] = coeff*a[i]*b[j]*c[k] ✓
        
        for i in range(n):
            for j in range(m):
                U[i * m + j, r] = a[i] * b[j]
        for j in range(m):
            for k in range(p):
                V[j * p + k, r] = 1.0
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = coeff * c[k]
        
        residual -= coeff * np.einsum('i,j,k->ijk', a, b, c)
    
    # Check if residual is zero
    if not np.allclose(residual, 0, atol=1e-6):
        return None
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + DEADLINE_S
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U_sb, V_sb, W_sb)
    
    # Build tensor
    T3 = _build_tensor3(n, m, p)
    
    # Compute column-space lower bound
    # Matricize T3 as (n*m, p*n) using the trace convention
    # T[i*m+j, k*n+i] = 1 for all i,j,k
    T_mat = np.zeros((n * m, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_mat[i * m + j, k * n + i] = 1.0
    
    R_star = int(np.linalg.matrix_rank(T_mat, tol=1e-9))
    
    # Try ranks from R_star to best_rank-1
    for R in range(max(R_star, 1), best_rank):
        if time.time() > deadline - 10:
            break
        result = _try_peel(n, m, p, R, T3, deadline)
        if result is not None:
            U, V, W = result
            if _validate_decomposition(n, m, p, U, V, W):
                best_rank = R
                best = (U, V, W)
                break  # Found minimal rank, stop
    
    U, V, W = best
    
    # Convert to lists of lists with integer/half-integer entries if possible
    U_list = [[round(float(U[i, r]), 10) for r in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[round(float(V[i, r]), 10) for r in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[round(float(W[i, r]), 10) for r in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}

