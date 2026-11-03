"""Find a low-rank exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
n, m, p = DIMENSIONS


def _build_tensor():
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid triples."""
    # T is a tensor of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _mat_mult_tensor_matrix(T, A, B):
    """Contract A (n x m) and B (m x p) through tensor T to get C (n x p).
    C[i,k] = sum_{j} T[i*m+j, j*p+k, k*n+i] * A[i,j] * B[j,k] = A[i,j]*B[j,k]"""
    # T is (n*m, m*p, p*n). We need C[i,k] = sum_j T[i*m+j, j*p+k, k*n+i] A[i,j] B[j,k]
    # Reshape T to (n, m, p, n) by reshaping
    # T[i*m+j, j*p+k, k*n+i] -> reshape: (n, m, p, n)
    T4 = T.reshape(n, m, p, n)
    # C[i,k] = sum_j T4[i,j,k,i] * A[i,j] * B[j,k]
    # But T4[i,j,k,i] = 1, so C[i,k] = sum_j A[i,j]*B[j,k]
    # In general, for a candidate decomposition we don't have T, we have U,V,W factors.
    # This function is just for reference/validation.
    C = np.zeros((n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i, k] += T4[i, j, k, i] * A[i, j] * B[j, k]
    return C


def _decompose_matrix_mult(T, r):
    """Given tensor T of shape (a, b, c) and target rank r, find U(a x r), V(b x r), W(c x r)
    such that T[i,j,k] = sum_l U[i,l]*V[j,l]*W[k,l].
    
    We use a direct approach: fix U and V, solve for W in least squares, then iterate.
    Actually, for this specific structured tensor, let's use the standard approach.
    
    Strategy: Parameterize U, V, W as free variables and minimize ||tensor - U*V*W||_F.
    Since r may be close to the true rank, we can use ALS (alternating least squares).
    """
    a, b, c = T.shape
    np.random.seed(42)
    
    # Initialize factors
    U = np.random.randn(a, r)
    V = np.random.randn(b, r)
    W = np.random.randn(c, r)
    
    # Normalize initial scale
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(300):
        # Update W: for each k, W[:,k] minimizes ||T[:,:,k] - U @ diag(W[:,k]) @ V.T||_F
        # T[:,:,k] = U @ (W[:,k][:,None] * V.T) 
        # Let X = T[:,:,k], then X = U * w_k where w_k = W[:,k]
        # In matrix form: X = U @ diag(W[:,k]) @ V.T? No.
        # T[i,j,k] = sum_l U[i,l] * V[j,l] * W[k,l]
        # For fixed U,V: T[:,:,k] (as matrix in i,j) = U @ diag(W[k,:]) @ V.T
        # So W[k,:] can be found by: for each k, solve min ||T[:,:,k] - U @ diag(w) @ V.T||_F^2
        # This is equivalent to: min_w sum_{i,j} (T[i,j,k] - (U*V.T) ⊙ ... )
        # Actually, T[:,:,k][i,j] = sum_l U[i,l]*V[j,l]*W[k,l]
        # = (U @ diag(W[k,:]) @ V.T)[i,j]
        # Let M = U @ V.T (a x b). Then T[:,:,k][i,j] = sum_l U[i,l]*V[j,l]*W[k,l]
        # This is not simply M ⊙ something unless we use the Khatri-Rao product.
        
        # Better: vectorize. For fixed U and V, T_flat (a*b*c) = kron(V.T, U) @ W_flat? 
        # Actually: T[i,j,k] = sum_l U[i,l] V[j,l] W[k,l]
        # Vectorize T as vector of length a*b*c.
        # T_flat = sum_l (U[:,l] ⊗ V[:,l] ⊗ W[:,l])
        # This is a linear system if we fix any two factors.
        
        # For W: fix U, V. Then for each k:
        # T[:,:,k] = U @ D_k @ V.T where D_k = diag(W[k,:])
        # Element-wise: T[i,j,k] = sum_l U[i,l]*V[j,l]*W[k,l]
        # Let P = U @ V.T (a x b). Then we want: T[:,:,k] approx P ⊙ (w_k broadcast)
        # But that's not right. T[i,j,k] = sum_l U[i,l]*V[j,l]*W[k,l]
        # = (U ⊙_l V) contracted with W[k,:]
        # Define K_ij_l = U[i,l]*V[j,l]. Then T[i,j,k] = sum_l K_ij_l * W[k,l]
        # So for fixed U,V, we have a linear system for W:
        # For each (i,j,k): T[i,j,k] = sum_l (U[i,l]*V[j,l]) * W[k,l]
        # Reshape: A_W is (a*b*c, r) where A_W[i*b*a + j*c + k, l] = U[i,l]*V[j,l]
        # Then T_flat = A_W @ W_flat? No, W_flat would be r*... 
        # Actually W has shape (c, r). Let's think of it as: for each l, the contribution to T is
        # (U[:,l]) ⊗ (V[:,l]) ⊗ (W[:,l])
        # So T_flat = sum_l u_l ⊗ v_l ⊗ w_l where u_l = U[:,l], v_l = V[:,l], w_l = W[:,l]
        # If we fix U and V, then T_flat = A @ w where A is (a*b*c, c*r) and w is W reshaped to (c*r,)? 
        # Let's be more careful. W has shape (c, r). Flatten W to w (c*r,).
        # A[row, col] where row = i*b*a + j*c + k, and for each (i,j,k), the coefficient of W[k', l] is
        # U[i,l]*V[j,l] if k'==k, 0 otherwise.
        # So A is sparse in some sense. Let's just do it directly.
        
        # Actually, let me just use a simpler ALS approach with NumPy operations.
        
        # Update W:
        # For each l, W[:,l] appears in T[:,:,l'] only via... no, W[k,l] couples all k for fixed l.
        # T[i,j,k] = sum_l' U[i,l']*V[j,l']*W[k,l']
        # For fixed U, V: treat as T[:,:,k] = U @ D_k @ V.T where D_k is diagonal.
        # Alternatively: reshape T to (a, b, c). For each k, T[:,:,k] = U @ W[k,:,None] * V.T? No.
        # U[i,l]*W[k,l]*V[j,l] summed over l.
        # (U @ diag(W[k,:]) @ V.T)[i,j] = sum_l U[i,l]*W[k,l]*V[j,l]. Yes!
        # So T[:,:,k] = U @ diag(W[k,:]) @ V.T
        # Let B_k = T[:,:,k], and let G = U.T @ B_k @ V.T? 
        # Actually: B_k = U @ diag(W[k,:]) @ V.T
        # Multiply on left by U.T and on right by V.T:
        # U.T @ B_k @ V.T = U.T @ U @ diag(W[k,:]) @ V.T @ V
        # Let C = U.T @ U (r x r), D = V.T @ V (r x r)
        # U.T @ B_k @ V.T = C @ diag(W[k,:]) @ D
        # Let E_k = U.T @ B_k @ V.T (r x r)
        # E_k = C @ diag(W[k,:]) @ D
        # If C and D are invertible: diag(W[k,:]) = C^{-1} @ E_k @ D^{-1}
        # But this only works if the diagonal part is isolated. Actually E_k has both diagonal and off-diagonal.
        # The diagonal of E_k should be C @ diag(W[k,:]) @ D, but off-diagonal should be zero if it were exact.
        # This approach gets messy. Let me just do a gradient-based approach or use the standard 
        # matrix multiplication tensor structure.
        
        # Given the time constraint, let me try a different, more direct approach.
        # For the matrix multiplication tensor T_{n,m,p}[i,j,k] = 1 if indices match, 0 otherwise,
        # we know the schoolbook rank is n*m*p = 36.
        # Can we do better? Let's try to find a rank-35 or lower decomposition using optimization.
        
        # Use scipy minimize on the flattened parameters of U, V, W.
        break  # Let me restructure below.
    
    return best_U, best_V, best_W, best_loss


def _try_rank(T, r, time_limit=10):
    """Try to find a rank-r decomposition of tensor T. Returns (U, V, W, loss) or None."""
    a, b, c = T.shape
    start_time = time.time()
    np.random.seed(int(hash(str(r)) % 10000))
    
    # Try multiple random initializations
    best_U, best_V, best_W = None, None, None
    best_loss = np.inf
    
    T_flat = T.flatten()
    
    for init in range(3):
        U = np.random.randn(a, r)
        V = np.random.randn(b, r)
        W = np.random.randn(c, r)
        
        # Normalize
        nrm = np.sqrt(np.linalg.norm(U)**2 + np.linalg.norm(V)**2 + np.linalg.norm(W)**2)
        U /= nrm
        V /= nrm
        W /= nrm
        
        def model(params):
            Uf = params[:a*r].reshape(a, r)
            Vf = params[a*r:a*r+b*r].reshape(b, r)
            Wf = params[a*r+b*r:a*r+b*r+c*r].reshape(c, r)
            # T[i,j,k] = sum_l Uf[i,l]*Vf[j,l]*Wf[k,l]
            # Reshape and compute
            # T_flat = sum_l u_l ⊗ v_l ⊗ w_l
            # = einsum('il,jl,kl->ijk', Uf, Vf, Wf).flatten()
            try:
                Tc = np.einsum('il,jl,kl->ijk', Uf, Vf, Wf).flatten()
            except Exception:
                return np.inf
            loss = np.sum((Tc - T_flat)**2)
            return loss
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        res = minimize(model, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 100000})
        
        if res.fun < best_loss:
            best_loss = res.fun
            Uf = res.x[:a*r].reshape(a, r)
            Vf = res.x[a*r:a*r+b*r].reshape(b, r)
            Wf = res.x[a*r+b*r:a*r+b*r+c*r].reshape(c, r)
            best_U, best_V, best_W = Uf.copy(), Vf.copy(), Wf.copy()
        
        if time.time() - start_time > time_limit:
            break
    
    if best_U is None:
        return None
    
    # Check if the decomposition is exact
    # Verify with the factor formula
    T_check = np.einsum('il,jl,kl->ijk', best_U, best_V, best_W)
    final_loss = np.max(np.abs(T_check - T))
    
    if final_loss < 1e-8:
        return best_U, best_V, best_W, 0.0
    else:
        return None


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying 2x3 by 3x6 matrices."""
    global n, m, p
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 140  # Keep 10s safety margin
    
    # Build the tensor
    T = _build_tensor()
    T = T.astype(np.float64)
    
    # Start with schoolbook (rank 36) as the guaranteed fallback
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
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank
    for r in range(20, best_rank):  # Try ranks from 20 up to 35
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(T, r, time_limit=min(20, max(5, deadline - 5 - (time.time() - start_time))))
        if result is not None:
            U_found, V_found, W_found, loss = result
            best_rank = r
            best_U, best_V, best_W = U_found, V_found, W_found
            break  # Found a valid decomposition, return it
    
    # Convert to lists as required
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
