"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x5 matrix.
    Uses a randomized Newton-Raphson approach on the polynomial system defining
    decomposable tensors of rank r, trying ranks from a lower bound upwards.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 145s safety margin
    
    # The tensor T has entries T[i, j, k] = 1 if we multiply (i,j) of A by (j,k) of B
    # In the decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k}
    # where delta is 1 for all valid (i,j,k) tuples (since every entry of the product
    # involves the corresponding entry of A and B)
    
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 80 polynomial equations in 3*r variables (entries of U, V, W)
    # with the constraint that each term is a product of three variables (one from each factor)
    
    # The variety of decomposable tensors of rank r is dense for r >= some bound.
    # For 4x4x5, the schoolbook rank is 80. We need to find the minimum r.
    
    # Known results: for matrix multiplication, the rank is at most n*m*p (schoolbook).
    # For 4x4x5, let's try to find a decomposition with lower rank.
    
    # Strategy: Use randomized initialization and Newton's method on the residual
    # F(U,V,W) = tensor(U,V,W) - T, where tensor(U,V,W)[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    def _build_tensor(u, v, w, r):
        """Build the decomposed tensor from factors u (nm x r), v (mp x r), w (pn x r)."""
        # u: shape (n*m, r)
        # v: shape (m*p, r)
        # w: shape (p*n, r)
        # Result: tensor[i,j,k] = sum_c u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c]
        t = np.zeros((n, m, p))
        for c in range(r):
            uc = u[:, c]  # (n*m,)
            vc = v[:, c]  # (m*p,)
            wc = w[:, c]  # (p*n,)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
        return t
    
    def _residual(u, v, w, r):
        """Compute the residual tensor - target."""
        t = _build_tensor(u, v, w, r)
        return t - 1.0  # target is all ones
    
    def _jacobian_and_solve(u, v, w, r):
        """
        Compute the Newton step. The residual is F[i,j,k] = sum_c u[i*m+j,c]*v[j*p+k,c]*w[k*n+i,c] - 1
        dF/d(u[a,c]) = v[j*p+k,c] * w[k*n+i,c] where a = i*m+j
        dF/d(v[b,c]) = u[i*m+j,c] * w[k*n+i,c] where b = j*p+k
        dF/d(w[d,c]) = u[i*m+j,c] * v[j*p+k,c] where d = k*n+i
        
        This is a 3-way tensor product structure. The Jacobian is a 3-mode tensor.
        For Newton, we need to solve J * delta = -F.
        
        This is complex. Instead, use a simpler approach: coordinate descent or 
        randomized gradient descent on the sum of squares of residuals.
        """
        pass
    
    def _random_decomposition(r, seed):
        """Generate a random decomposition of rank r."""
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((n * m, r))
        v = rng.standard_normal((m * p, r))
        w = rng.standard_normal((p * n, r))
        return u, v, w
    
    def _verify(u, v, w, r, tol=1e-8):
        """Verify the decomposition is exact (within tolerance)."""
        t = _build_tensor(u, v, w, r)
        return np.allclose(t, 1.0, atol=tol, rtol=tol)
    
    def _polish(u, v, w, r, max_iter=200, tol=1e-10):
        """
        Polish the decomposition using a simple iterative method.
        Use alternating least squares style updates.
        """
        for iteration in range(max_iter):
            if time.time() - start_time > deadline:
                break
            
            # Update u: for each row i*m+j, we want sum_c u[i*m+j,c] * v[j*p+k,c] * w[k*n+i,c] = 1
            # This is linear in u given v,w.
            # For a fixed (i,j), the equation is: for all k, sum_c u[i*m+j,c] * (v[j*p+k,c] * w[k*n+i,c]) = 1
            # This is a linear system: A[i,j] * u[i*m+j,:] = ones(p)
            # where A[i,j][k,c] = v[j*p+k,c] * w[k*n+i,c]
            
            for i in range(n):
                for j in range(m):
                    # Build the linear system for u[i*m+j, :]
                    # A[k, c] = v[j*p+k, c] * w[k*n+i, c]
                    A = np.outer(v[j * p:(j + 1) * p, :], w[(i):(i + 1), :].T * 1)  # This is wrong
                    # Let me redo: A[k, c] = v[j*p+k, c] * w[k*n+i, c]
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = v[j * p + k, c] * w[k * n + i, c]
                    # Solve A * x = ones(p)
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        u[i * m + j, :] = x
                    except:
                        pass
            
            # Update v: for each row j*p+k, we want sum_c u[i*m+j,c] * v[j*p+k,c] * w[k*n+i,c] = 1
            # For fixed (j,k), the equation is: for all i, sum_c v[j*p+k,c] * (u[i*m+j,c] * w[k*n+i,c]) = 1
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = u[i * m + j, c] * w[k * n + i, c]
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        v[j * p + k, :] = x
                    except:
                        pass
            
            # Update w: for each row k*n+i, we want sum_c u[i*m+j,c] * v[j*p+k,c] * w[k*n+i,c] = 1
            # For fixed (k,i), the equation is: for all j, sum_c w[k*n+i,c] * (u[i*m+j,c] * v[j*p+k,c]) = 1
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = u[i * m + j, c] * v[j * p + k, c]
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        w[k * n + i, :] = x
                    except:
                        pass
            
            # Check convergence
            if _verify(u, v, w, r, tol=1e-8):
                break
        
        return u, v, w
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: the rank must be at least max(n*m, m*p, p*n) / something... 
    # Actually, a simple lower bound is that rank >= max(n, m, p) in some cases,
    # but for matrix multiplication it's known that rank >= n*m*p / (something).
    # Let's just try from a reasonable value.
    
    # For 4x4x5, let's try ranks from 40 up to 80 (schoolbook is 80)
    # Actually, let's be smarter. The minimum possible rank is at least the 
    # "quantum rank" or other lower bounds, but let's just try a range.
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, always have the schoolbook solution as fallback
    rank_sb = n * m * p
    u_sb = np.zeros((n * m, rank_sb))
    v_sb = np.zeros((m * p, rank_sb))
    w_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_sb[i * m + j, col] = 1
                v_sb[j * p + k, col] = 1
                w_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    for r in range(60, rank_sb + 1):  # Start from 60, go up to schoolbook
        if time.time() - start_time > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() - start_time > deadline:
                break
            
            seed = r * 1000 + attempt
            u, v, w = _random_decomposition(r, seed)
            
            # Polish
            u, v, w = _polish(u, v, w, r, max_iter=100)
            
            if _verify(u, v, w, r, tol=1e-8):
                best_rank = r
                best_result = (u, v, w, r)
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        return {"U": u_sb.tolist(), "V": v_sb.tolist(), "W": w_sb.tolist()}
    else:
        u, v, w, r = best_result
        return {"U": u.tolist(), "V": v.tolist(), "W": w.tolist()}
# EVOLVE-BLOCK-END
