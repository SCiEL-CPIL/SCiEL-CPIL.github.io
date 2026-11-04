"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 16
    dim_w = p * n  # 8
    
    # Build the multiplication tensor T of shape (8, 16, 8)
    # T[a, b, c] = 1 iff (i, j) with a = i*m+j, (j', k') with b = j'*p+k', (k'', i'') with c = k''*n+i''
    # and j == j' and i == i'' and k == k''
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Start with schoolbook rank 32
    def make_schoolbook():
        U = np.zeros((dim_u, 32))
        V = np.zeros((dim_v, 32))
        W = np.zeros((dim_w, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = 32
    
    def check_rank(U, V, W):
        """Check if U @ V^T @ W^T reproduces T (up to small tolerance)."""
        # The tensor T[a,b,c] should equal sum_c' U[a, c'] * V[b, c'] * W[c, c']
        # This is the tensor product U ⊗ V ⊗ W
        r = U.shape[1]
        # Compute sum over rank of outer products
        # T_recon[a, b, c] = sum_{k=0}^{r-1} U[a,k] * V[b,k] * W[c,k]
        # This is a 3D tensor contraction
        # Efficient way: for each k, add U[:,k] outer V[:,k] outer W[:,k]
        # But that's r iterations. Alternatively:
        # T_recon = np.einsum('ak,bk,ck->abc', U, V, W)
        T_recon = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=1e-9, rtol=1e-9)
    
    def try_rank(r, max_iter=200, tol=1e-8):
        """Try to find a rank-r decomposition using alternating minimization."""
        if r >= best_rank:
            return None
        if time.time() > deadline:
            return None
        
        # Initialize with random or schoolbook-based initialization
        # Use a smarter initialization: take schoolbook rank 32, randomly project to rank r
        U_s, V_s, W_s = make_schoolbook()
        
        # Random linear combination to get rank r init
        np.random.seed(42)
        
        for attempt in range(3):
            # Create random mixing matrices
            if attempt == 0:
                # Keep first r columns
                U_init = U_s[:, :r].copy()
                V_init = V_s[:, :r].copy()
                W_init = W_s[:, :r].copy()
            elif attempt == 1:
                # Random projection from rank 32 to rank r
                A = np.random.randn(32, r)
                B = np.random.randn(32, r)
                C = np.random.randn(32, r)
                # Scale to maintain some magnitude
                A /= np.linalg.norm(A) * np.sqrt(r) if np.linalg.norm(A) > 0 else 1
                B /= np.linalg.norm(B) * np.sqrt(r) if np.linalg.norm(B) > 0 else 1
                C /= np.linalg.norm(C) * np.sqrt(r) if np.linalg.norm(C) > 0 else 1
                U_init = U_s @ A
                V_init = V_s @ B
                W_init = W_s @ C
            else:
                # Random init with some structure
                U_init = np.random.randn(dim_u, r) * 0.5
                V_init = np.random.randn(dim_v, r) * 0.5
                W_init = np.random.randn(dim_w, r) * 0.5
                # Ensure some entries match the pattern
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            # At least one component should be nonzero
                            idx = (i * m + j + k) % r
                            U_init[a, idx] = 1.0
                            V_init[b, idx] = 1.0
                            W_init[c, idx] = 1.0
            
            # Alternating minimization
            U_curr, V_curr, W_curr = U_init.copy(), V_init.copy(), W_init.copy()
            
            for iteration in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Fix V, W: solve for U
                # For each a: T[a, :, :] = sum_k U[a,k] * V[:, k] outer W[:, k]
                # Let M_vw = V @ W.T  (shape dim_v x dim_w)
                # Then T[a, b, c] = sum_k U[a,k] * V[b,k] * W[c,k]
                # = sum_k U[a,k] * (V[b,k] * W[c,k])
                # For fixed V, W: define for each k: vec_k = V[:,k] * W[:,k] (elementwise, shape dim_v x dim_w)
                # T[a, :, :] = sum_k U[a,k] * vec_k
                # This is a linear system in U[a, :]
                
                # Build the design matrix: for each (b,c), the coefficient of U[a,k] is V[b,k]*W[c,k]
                # Flatten: design[k, idx] = V[b_idx, k] * W[c_idx, k] where idx = b*dim_w + c
                # Target: T[a, :, :].ravel()
                
                # Efficient: M = np.einsum('bk,ck->bc', V_curr, W_curr)  # dim_v x dim_w
                # Then for each a: T[a].ravel() = U[a] @ (something)
                # Actually: T[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
                # Let D = V_curr @ W_curr.T  -- no, that's not right.
                # T[a,b,c] = sum_k U[a,k] * (V_curr[:, k] * W_curr[:, k])[b, c]... no.
                # V_curr[b,k] * W_curr[c,k] summed over k with U[a,k].
                # For each a, this is a linear system:
                # T[a, b, c] = sum_{k=0}^{r-1} U[a,k] * V_curr[b,k] * W_curr[c,k]
                # Let M_bc = V_curr @ W_curr.T has shape (dim_v, dim_w), M_bc[b,c] = sum_k V[b,k]*W[c,k]
                # But we need per-k terms, not summed.
                # 
                # Better approach: form the matrix A_a where A_a has columns indexed by k and rows by (b,c):
                # A_a[k, b*dim_w+c] = V_curr[b,k] * W_curr[c,k]
                # Then T[a, :, :].ravel() = U[a, :] @ A_a.T... 
                # Actually: sum_k U[a,k] * A_a[k, idx] = (U[a] @ A_a.T)[idx]... no
                # sum_k U[a,k] * A_a[k, idx] = (U[a] @ A_a.T)[idx] only if A_a is r x (dim_v*dim_w)
                # Yes! So A has shape (r, dim_v * dim_w)
                # A[k, b*dim_w+c] = V_curr[b,k] * W_curr[c,k]
                
                # Build A efficiently
                # A = np.einsum('bk,ck->kbc', V_curr, W_curr).transpose(0, 1, 2).reshape(r, dim_v * dim_w)
                # = np.einsum('bk,ck->kbc', V_curr, W_curr).reshape(r, dim_v * dim_w)
                
                A_mat = np.einsum('bk,ck->kbc', V_curr, W_curr, optimize=True)
                A_mat = A_mat.reshape(r, dim_v * dim_w)
                
                # Solve: U[a, :] = (A_mat.T @ A_mat)^-1 A_mat.T @ T[a].ravel()
                # Use lstsq for stability
                try:
                    U_curr = np.zeros((dim_u, r))
                    for a in range(dim_u):
                        target = T[a, :, :].ravel()
                        U_curr[a, :], _, _, _ = np.linalg.lstsq(A_mat, target, rcond=None)
                except np.linalg.LinAlgError:
                    break
                
                if time.time() > deadline:
                    break
                
                # Fix U, W: solve for V
                # T[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
                # For each b: T[:, b, c] = sum_k V[b,k] * (U[:,k] * W[:,k])... 
                # T[a,b,c] = sum_k V[b,k] * U[a,k] * W[c,k]
                # A_mat2[k, a*dim_w+c] = U_curr[a,k] * W_curr[c,k]
                B_mat = np.einsum('ak,ck->kac', U_curr, W_curr, optimize=True)
                B_mat = B_mat.reshape(r, dim_u * dim_w)
                
                V_curr = np.zeros((dim_v, r))
                for b in range(dim_v):
                    target = T[:, b, :].ravel()
                    V_curr[b, :], _, _, _ = np.linalg.lstsq(B_mat, target, rcond=None)
                
                if time.time() > deadline:
                    break
                
                # Fix U, V: solve for W
                # T[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
                # For each c: T[a,b,c] = sum_k W[c,k] * U[a,k] * V[b,k]
                # A_mat3[k, a*dim_v+b] = U_curr[a,k] * V_curr[b,k]
                C_mat = np.einsum('ak,bk->kab', U_curr, V_curr, optimize=True)
                C_mat = C_mat.reshape(r, dim_u * dim_v)
                
                W_curr = np.zeros((dim_w, r))
                for c in range(dim_w):
                    target = T[:, :, c].ravel()
                    W_curr[c, :], _, _, _ = np.linalg.lstsq(C_mat, target, rcond=None)
                
                if time.time() > deadline:
                    break
                
                # Check convergence
                if iteration % 10 == 0:
                    T_recon = np.einsum('ak,bk,ck->abc', U_curr, V_curr, W_curr, optimize=True)
                    err = np.linalg.norm(T_recon - T)
                    if err < tol:
                        break
            
            # Final check
            if check_rank(U_curr, V_curr, W_curr):
                # Try to clean up: zero out very small entries
                U_curr[np.abs(U_curr) < 1e-10] = 0
                V_curr[np.abs(V_curr) < 1e-10] = 0
                W_curr[np.abs(W_curr) < 1e-10] = 0
                if check_rank(U_curr, V_curr, W_curr):
                    return U_curr, V_curr, W_curr
                else:
                    return U_curr, V_curr, W_curr
        
        return None
    
    # Try ranks from 31 down to some minimum
    # Known bounds: for 2x4*4x4, the rank is at most 32 (schoolbook)
    # Let's try 31, 30, 29, ...
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iter=300)
        if result is not None:
            U_new, V_new, W_new = result
            if check_rank(U_new, V_new, W_new):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = r
                # Can we go lower?
                continue
    
    # Convert to list format
    def to_list(arr):
        return arr.tolist()
    
    # Ensure integer or half-integer entries
    # Check if entries are close to integers or half-integers
    def snap_to_half(x):
        """Snap values close to n/2 to exact n/2."""
        x2 = 2 * x
        rounded = np.round(x2)
        if np.allclose(x2, rounded, atol=1e-8):
            return (rounded / 2.0)
        return x
    
    best_U = snap_to_half(best_U)
    best_V = snap_to_half(best_V)
    best_W = snap_to_half(best_W)
    
    # Final verification
    if not check_rank(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = make_schoolbook()
        best_rank = 32
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
