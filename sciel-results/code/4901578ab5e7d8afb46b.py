"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin

    # First, try to find a rank-23 decomposition for 3x3x3
    # Then extend to 3x3x6 by doubling rank to 46
    # If that fails, fall back to schoolbook rank 54
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Generate schoolbook decomposition
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
    best_decomp = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find a better decomposition using numerical optimization
    # We'll try to find a decomposition of rank r for the 3x3x3 problem,
    # then extend to 3x3x6 by doubling.
    
    # The 3x3x3 multiplication tensor
    # T[i, j, k] = 1 if i, j, k all in range(3)
    # But we need to be careful about the indexing convention.
    # For 3x3x3: A is 3x3, B is 3x3, C is 3x3
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # So T[i, j, k] = 1 for all i, j, k in range(3)
    # The decomposition: C = U @ diag(c) @ V @ W^T... 
    # Actually, the standard form is:
    # C_{ik} = sum_r (sum_j U_{i j, r} A_{i j}) * (sum_k V_{j k, r} B_{j k}) * W_{k i, r}
    # Wait, let me re-read the convention.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W has indices (k, i, r) where k is the column of B / row of C, i is row of A / col of C.
    # U has indices (i, j, r) where i is row of A, j is col of A / row of B.
    # V has indices (j, k, r) where j is row of B, k is col of B.
    
    # For 3x3x3:
    # U: shape (9, r), indexed by (i*3 + j)
    # V: shape (9, r), indexed by (j*3 + k)
    # W: shape (9, r), indexed by (k*3 + i)
    
    # Let's try to solve for rank r decompositions of 3x3x3.
    # The known minimum rank is 23.
    # We'll try to find it numerically.
    
    def verify_decomp_333(U, V, W, tol=1e-6):
        """Verify that U, V, W decompose the 3x3x3 multiplication tensor."""
        n, m, p = 3, 3, 3
        r = U.shape[1]
        # Compute the reconstructed tensor
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        # The target tensor is all ones
        return np.allclose(T, 1.0, atol=tol)
    
    def extend_333_to_336(U3, V3, W3):
        """Extend a 3x3x3 decomposition to 3x3x6.
        
        The 3x3x6 multiplication: A (3x3) * B (3x6) = C (3x6)
        We can view B as two 3x3 matrices side by side: B = [B1, B2]
        Then C = [A*B1, A*B2]
        So we can use the 3x3x3 decomposition twice, once for each half.
        
        For 3x3x6:
        U: shape (9, 2r)
        V: shape (18, 2r)
        W: shape (18, 2r)
        
        The first r columns use the 3x3x3 decomposition for B1.
        The last r columns use the 3x3x3 decomposition for B2.
        """
        r = U3.shape[1]
        rank = 2 * r
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        # First half: B1 is columns 0-2 of B (indices j*6 + k for k=0,1,2)
        U[:, :r] = U3
        # V for first half: V3[j*3 + k, r_idx] -> V_ext[j*6 + k, r_idx] for k=0,1,2
        V3 = V3
        for j in range(3):
            for k in range(3):
                V[j * 6 + k, :r] = V3[j * 3 + k, :]
        # W for first half: W3[k*3 + i, r_idx] -> W_ext[k*3 + i, r_idx] for k=0,1,2
        # W has shape (18, rank), indexed by (k*3 + i) where k=0..5, i=0..2
        for k in range(3):
            for i in range(3):
                W[k * 3 + i, :r] = W3[k * 3 + i, :]
        
        # Second half: B2 is columns 3-5 of B (indices j*6 + k for k=3,4,5)
        U[:, r:] = U3
        for j in range(3):
            for k in range(3):
                V[j * 6 + (k + 3), r:] = V3[j * 3 + k, :]
        for k in range(3):
            for i in range(3):
                W[(k + 3) * 3 + i, r:] = W3[k * 3 + i, :]
        
        return U, V, W
    
    def verify_decomp_336(U, V, W, tol=1e-6):
        """Verify that U, V, W decompose the 3x3x6 multiplication tensor."""
        n, m, p = 3, 3, 6
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=tol)
    
    # Try to find a rank-23 decomposition of 3x3x3 using optimization
    # This is a hard problem, so we'll try a few random restarts.
    
    import scipy.optimize as optimize
    
    def solve_333_rank(r, max_iter=100, timeout=20.0):
        """Try to find a rank-r decomposition of 3x3x3.
        
        Returns (U, V, W) or None if not found.
        """
        if time.time() > deadline:
            return None
        
        n, m, p = 3, 3, 3
        
        # Random initialization
        def random_init():
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            return U, V, W
        
        def loss_fn(params):
            U = params[:n * m * r].reshape(n * m, r)
            V = params[n * m * r : n * m * r + m * p * r].reshape(m * p, r)
            W = params[n * m * r + m * p * r :].reshape(p * n, r)
            T = np.zeros((n, m, p))
            for r_idx in range(r):
                u = U[:, r_idx].reshape(n, m)
                v = V[:, r_idx].reshape(m, p)
                w = W[:, r_idx].reshape(p, n)
                T += np.einsum('ij,jk,ki->ijk', u, v, w)
            err = T - 1.0
            return np.sum(err * err)
        
        # Try a few random restarts
        best_U, best_V, best_W = None, None, None
        best_loss = float('inf')
        
        for trial in range(5):
            if time.time() > deadline:
                break
            
            U, V, W = random_init()
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = optimize.minimize(loss_fn, params, method='L-BFGS-B', 
                                          maxiter=max_iter, options={'maxfun': 2000})
            except Exception:
                continue
            
            U_opt = result.x[:n * m * r].reshape(n * m, r)
            V_opt = result.x[n * m * r : n * m * r + m * p * r].reshape(m * p, r)
            W_opt = result.x[n * m * r + m * p * r :].reshape(p * n, r)
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_U, best_V, best_W = U_opt, V_opt, W_opt
            
            if best_loss < 1e-8:
                break
        
        if best_loss < 1e-6:
            # Round to nearest 0.5 to get integer/half-integer entries
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            if verify_decomp_333(U_rounded, V_rounded, W_rounded, tol=1e-6):
                return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try to find a rank-23 decomposition
    if time.time() < deadline - 5.0:
        result_23 = solve_333_rank(23, max_iter=50, timeout=15.0)
        if result_23 is not None:
            U3, V3, W3 = result_23
            U336, V336, W336 = extend_333_to_336(U3, V3, W3)
            if verify_decomp_336(U336, V336, W336, tol=1e-6):
                best_rank = 46
                best_decomp = (U336, V336, W336)
                # Convert to lists
                return {"U": U336.tolist(), "V": V336.tolist(), "W": W336.tolist()}
    
    # Fall back to schoolbook
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
