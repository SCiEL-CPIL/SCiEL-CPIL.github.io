"""Rewritten program for 3x5x7 matrix multiplication tensor rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # For each output element (i, k), the contribution should be:
    # sum over j of A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])
    # Wait, that's not right. Let me reconsider.
    
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should equal 1 for all i,j,k (the multiplication tensor).
    
    # Actually, let me re-examine. The standard tensor for matrix multiplication:
    # C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor M satisfies: M[i,j,k] = 1 for all valid i,j,k
    # A rank-1 component is: u[i*m+j] * v[j*p+k] * w[k*n+i]
    # Sum over c: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _round_to_halfint(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _try_rank(r, n, m, p, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute loss: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    loss += (val - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-8:
            break
        
        # Gradient descent (simplified)
        # This is expensive, so let's do a few steps
        lr = 0.01
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = val - 1.0
                    for c in range(r):
                        # Gradient w.r.t. U[i*m+j, c]
                        grad_U = err * V[j*p+k, c] * W[k*n+i, c]
                        # Gradient w.r.t. V[j*p+k, c]
                        grad_V = err * U[i*m+j, c] * W[k*n+i, c]
                        # Gradient w.r.t. W[k*n+i, c]
                        grad_W = err * U[i*m+j, c] * V[j*p+k, c]
                        U[i*m+j, c] -= lr * grad_U
                        V[j*p+k, c] -= lr * grad_V
                        W[k*n+i, c] -= lr * grad_W
    
    # Round to half-integers
    U_rounded = _round_to_halfint(best_U)
    V_rounded = _round_to_halfint(best_V)
    W_rounded = _round_to_halfint(best_W)
    
    # Verify the rounded version
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    # Try without rounding (check if it's already close to half-integers)
    if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better ranks
    # Known lower bound: max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # But that's not achievable in general. Let's try from a reasonable upper bound down.
    
    # For 3x5x7, let's try ranks from 80 down to 40
    for r in range(80, 34, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                    break
            
            result = _try_rank(r, n, m, p, max_iters=2000, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break  # Found a valid decomposition at this rank, try lower
        
        if time.time() > deadline - 5:
            break
    
    return best_result
