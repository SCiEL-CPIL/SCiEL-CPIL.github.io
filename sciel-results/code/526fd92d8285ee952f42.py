"""Rank-minimal tensor decomposition for 2x3x9 matrix multiplication via numerical search."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

def _build_target_tensor():
    """Build the 3D target tensor T[i,j,k] = 1 if indices match, else 0.
    Shape: (n*m, m*p, p*n) where:
      U index a = i*m + j  (i in [0,n), j in [0,m))
      V index b = j*p + k  (j in [0,m), k in [0,p))
      W index c = k*n + i  (k in [0,p), i in [0,n))
    T[a,b,c] = 1 if (a%m == b//p) else 0
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _decompose_to_tensors(U, V, W):
    """Given U (nm x r), V (mp x r), W (pn x r), compute the tensor they represent.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    nm, mp, pn, r = U.shape[0], V.shape[0], W.shape[0], U.shape[1]
    # Efficient: T = np.einsum('ar,br,cr->abc', U, V, W)
    return np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

def _rank_error(U, V, W, T):
    """Compute the Frobenius norm of the error tensor T - decompose(U,V,W)."""
    T_decomp = _decompose_to_tensors(U, V, W)
    return np.linalg.norm(T - T_decomp)

def _try_rank(r, T, max_iter=300, tol=1e-10, seed=42):
    """
    Attempt to find a rank-r decomposition by iterative refinement.
    Returns (U, V, W) as float arrays or None if not found within max_iter.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rng = np.random.default_rng(seed)
    
    # Initialize with random values
    U = rng.standard_normal((nm, r)) * 0.5
    V = rng.standard_normal((mp, r)) * 0.5
    W = rng.standard_normal((pn, r)) * 0.5
    
    # Use a simple iterative method: alternating least squares style
    # For each factor, fix the other two and solve the linear system.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed V, W: T[a,b,c] over b,c gives U[a,:] * (V[:,b] outer W[:,c]) ... 
    # Actually, let's use a gradient-free approach or a simple projection.
    
    # Better: use the fact that for fixed V and W, the equation is linear in U.
    # T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
    # Let M_r = V[:,r] outer W[:,r], shape (mp, pn)
    # Then T[a,:,:] = sum_r U[a,r] * M_r
    # This is a linear system: T_flat[a, :] = U[a, :] @ M_stack
    # where M_stack has shape (mp*pn, r) with M_r flattened.
    
    # Precompute the flattened M matrices
    # M_stack[b*pn + c, r] = V[b, r] * W[c, r]
    # This can be built as: np.einsum('br,cr->bcr', V, W) then reshape to (mp*pn, r)
    
    # Let's implement alternating updates:
    # 1. Fix V, W. Solve for U.
    # 2. Fix U, W. Solve for V.
    # 3. Fix U, V. Solve for W.
    
    # Precompute the linear operators
    # For U: T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # Reshape T to (nm, mp*pn)
    T_2d = T.reshape(nm, mp * pn)
    
    # Build A_U: (mp*pn, r) where A_U[b*pn+c, r] = V[b,r]*W[c,r]
    # A_U = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
    
    # Similarly for V: T[a, b, c] = sum_r V[b, r] U[a, r] W[c, r]
    # T_2d_v = T.reshape(mp, nm*pn) ... but indices are mixed.
    # T[a, b, c] -> for fixed b: T[:, b, :] has shape (nm, pn)
    # T[:, b, :] = sum_r V[b, r] * (U[:, r] outer W[:, r])
    # So for each b, we have a linear system in V[b, :].
    
    # Let's do a simple gradient descent with small steps, or use np.linalg.lstsq for each factor.
    
    # Actually, let's use a simpler approach: random restarts with rank r, and check if we can get close.
    # Given the time constraint, let's try a few random initializations and see if we can get a low error.
    
    best_err = np.inf
    best_U, best_V, best_W = U, V, W
    
    for trial in range(10):  # 10 random trials
        U = rng.standard_normal((nm, r)) * 0.5
        V = rng.standard_normal((mp, r)) * 0.5
        W = rng.standard_normal((pn, r)) * 0.5
        
        # Alternating least squares
        for it in range(max_iter):
            # Update U
            A_U = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, r)
            # T_2d @ inv(A_U) ... but A_U is (mp*pn, r), T_2d is (nm, mp*pn)
            # We want U such that U @ A_U ≈ T_2d
            # This is U[a, :] @ A_U = T_2d[a, :]
            # So U[a, :] = T_2d[a, :] @ A_U^+ (pseudo-inverse)
            # A_U^+ has shape (r, mp*pn)
            try:
                A_U_pinv = np.linalg.pinv(A_U)
                U_new = T_2d @ A_U_pinv
                U = U_new
            except:
                pass
            
            # Update V
            # T[:, b, :] = sum_r V[b, r] * (U[:, r] outer W[:, r])
            # For each b: T[:, b, :].reshape(nm*pn) = V[b, :] @ A_V[b, :]
            # where A_V[b, r] = ... 
            # Let's build A_V: (mp, nm*pn, r) where A_V[b, a*pn+c, r] = U[a,r]*W[c,r]
            A_V = np.einsum('ar,cr->acr', U, W).reshape(mp, nm * pn, r)  # Wait, indices: a from U, c from W
            # Actually: A_V[b, a*pn+c, r] should be U[a, r] * W[c, r]
            # np.einsum('ar,cr->acr', U, W) gives shape (nm, pn, r)
            # We need to broadcast this for each b? No, A_V depends on b only through V.
            # The equation is: for each b, T[:, b, :].reshape(-1) = V[b, :] @ (U outer W).reshape(nm*pn, r)
            # Let M = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, r)
            M = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, r)
            # T_v: (mp, nm*pn) where T_v[b, a*pn+c] = T[a, b, c]
            T_v = np.transpose(T, (1, 0, 2)).reshape(mp, nm * pn)
            try:
                M_pinv = np.linalg.pinv(M)
                V_new = T_v @ M_pinv
                V = V_new
            except:
                pass
            
            # Update W
            # T[a, b, c] = sum_r W[c, r] * (U[a, r] outer V[b, r])
            # For each c: T[:, :, c].reshape(nm*mp) = W[c, :] @ M_W
            M = np.einsum('ar,br->abr', U, V).reshape(nm * mp, r)
            T_w = np.transpose(T, (0, 1, 2)).reshape(pn, nm * mp)  # T_w[c, a*mp+b] = T[a, b, c]
            try:
                M_pinv = np.linalg.pinv(M)
                W_new = T_w @ M_pinv
                W = W_new
            except:
                pass
            
            # Check convergence
            err = _rank_error(U, V, W, T)
            if err < tol:
                return U, V, W
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if best_err < tol:
        return best_U, best_V, best_W
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication.
    Uses numerical search to find a decomposition of rank < 54.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    T = _build_target_tensor()
    
    # Try ranks from 48 down to 54
    for r in range(48, 55):
        if time.time() > deadline:
            break
        result = _try_rank(r, T, max_iter=100, tol=1e-10, seed=r)
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition (rank 54)
    n, m, p = DIMENSIONS
    rank = n * m * p
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][col] = 1
                V_list[j * p + k][col] = 1
                W_list[k * n + i][col] = 1
                col += 1
    return {"U": U_list, "V": V_list, "W": W_list}
