import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Build the tensor from the decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The (i,j,k) entry should be 1
                    val = 0.0
                    for t in range(r):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check off-diagonal entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Check if (a,b,c) is a valid (i,j,k) triple
                    i = a // m
                    j = a % m
                    jj = b // p
                    k = b % p
                    kk = c // n
                    ii = c % n
                    if i == ii and j == jj and k == kk:
                        continue  # This is a diagonal entry, already checked
                    val = 0.0
                    for t in range(r):
                        val += U[a, t] * V[b, t] * W[c, t]
                    if abs(val) > 1e-9:
                        return False
        return True

    def round_to_half_int(arr):
        """Round array entries to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def try_rank_r(r, max_iters=50, tol=1e-8):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() > deadline:
            return None
        
        num_U = n * m   # 6
        num_V = m * p   # 15
        num_W = p * n   # 10
        
        # Precompute valid (i,j,k) -> (a,b,c) mapping
        valid = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid[(a, b, c)] = 1
        
        # All (a,b,c) triples
        all_triples = [(a, b, c) for a in range(num_U) for b in range(num_V) for c in range(num_W)]
        
        def compute_residual(U, V, W):
            """Compute the residual (max absolute error)."""
            max_err = 0.0
            for (a, b, c) in all_triples:
                val = 0.0
                for t in range(r):
                    val += U[a, t] * V[b, t] * W[c, t]
                if (a, b, c) in valid:
                    err = abs(val - 1.0)
                else:
                    err = abs(val)
                if err > max_err:
                    max_err = err
            return max_err

        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        for restart in range(max_iters):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            np.random.seed(restart * 1000 + 42)
            U = np.random.randn(num_U, r) * 0.1
            V = np.random.randn(num_V, r) * 0.1
            W = np.random.randn(num_W, r) * 0.1
            
            # Normalize columns to avoid trivial scaling issues
            for t in range(r):
                norm = np.sqrt(np.sum(U[:, t]**2) * np.sum(V[:, t]**2) * np.sum(W[:, t]**2))
                if norm > 1e-12:
                    scale = 1.0 / (norm ** (1.0/3.0))
                    U[:, t] *= scale
                    V[:, t] *= scale
                    W[:, t] *= scale
            
            err = compute_residual(U, V, W)
            
            # Iterative refinement: alternate solving for each factor
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for U given V, W
                # For each (a,b,c) in valid: sum_t U[a,t] * V[b,t] * W[c,t] = 1
                # For each (a,b,c) not in valid: sum_t U[a,t] * V[b,t] * W[c,t] = 0
                # This is linear in U for fixed V, W
                # Build linear system for each column a of U
                # Actually, let's use a simpler approach: gradient descent or alternating least squares
                
                # For each a in range(num_U):
                #   We need to find U[a, :] such that for all (b,c):
                #     sum_t U[a,t] * V[b,t] * W[c,t] = target(a,b,c)
                # where target is 1 if (a,b,c) valid, else 0
                # This is a linear system: M_a * U[a,:] = rhs_a
                # where M_a[b*r+c, t] = V[b,t] * W[c,t]
                
                for a in range(num_U):
                    # Build linear system
                    M = np.zeros((num_V * num_W, r))
                    rhs = np.zeros(num_V * num_W)
                    idx = 0
                    for b in range(num_V):
                        for c in range(num_W):
                            for t in range(r):
                                M[idx, t] = V[b, t] * W[c, t]
                            if (a, b, c) in valid:
                                rhs[idx] = 1.0
                            else:
                                rhs[idx] = 0.0
                            idx += 1
                    
                    # Solve least squares
                    try:
                        U[a, :], residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
                
                # Similarly for V
                for b in range(num_V):
                    M = np.zeros((num_U * num_W, r))
                    rhs = np.zeros(num_U * num_W)
                    idx = 0
                    for a in range(num_U):
                        for c in range(num_W):
                            for t in range(r):
                                M[idx, t] = U[a, t] * W[c, t]
                            if (a, b, c) in valid:
                                rhs[idx] = 1.0
                            else:
                                rhs[idx] = 0.0
                            idx += 1
                    try:
                        V[b, :], residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
                
                # Similarly for W
                for c in range(num_W):
                    M = np.zeros((num_U * num_V, r))
                    rhs = np.zeros(num_U * num_V)
                    idx = 0
                    for a in range(num_U):
                        for b in range(num_V):
                            for t in range(r):
                                M[idx, t] = U[a, t] * V[b, t]
                            if (a, b, c) in valid:
                                rhs[idx] = 1.0
                            else:
                                rhs[idx] = 0.0
                            idx += 1
                    try:
                        W[c, :], residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
                
                err = compute_residual(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < tol:
                    break
            
            if best_err < tol:
                break
        
        if best_err < tol and best_U is not None:
            # Round to half-integers and verify
            U_r = round_to_half_int(best_U)
            V_r = round_to_half_int(best_V)
            W_r = round_to_half_int(best_W)
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Also try without rounding
            if verify(best_U.tolist(), best_V.tolist(), best_W.tolist()):
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        return None

    # Try rank 28 first (known to be achievable for 2x3x5)
    for r in [28, 29]:
        if time.time() > deadline:
            break
        result = try_rank_r(r, max_iters=30, tol=1e-8)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
