"""Matrix multiplication 2x3x7 tensor decomposition - rank minimal search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_polynomial_construction(n, m, p):
    """
    Try a polynomial-based construction.
    For matmul: C[i,k] = sum_j A[i,j]*B[j,k]
    View as: multiply polynomials in different variables, evaluate at points.
    
    Strategy: Use the fact that we can represent the multiplication tensor
    using evaluations of products of polynomials.
    """
    # For n=2, m=3, p=7, try to find a structured rank decomposition
    # One approach: use the "baby-step giant-step" style or Strassen-like block methods
    
    # Simple structured attempt: exploit the rectangular structure
    # We need rank R such that we can write T as sum of R rank-1 tensors
    
    # Try a construction based on separating the indices
    # U depends on (i,j), V on (j,k), W on (k,i)
    
    # For small dimensions, try direct algebraic construction
    # The key insight: for fixed j, we have a bilinear map from (i) x (k) to (i,k)
    # This is like outer product structure
    
    # Let's try: for each j, use rank-1 components for the (i,k) part
    # But we need to share components across j values
    
    # Alternative: use the fact that n=2 is small
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # This is 2*7 = 14 outputs, each is a dot product of length 3
    # Schoolbook: 14*3 = 42 rank-1 terms
    
    # Can we do better? For each k, we need to compute two dot products.
    # If we precompute certain combinations of A and B, we might reduce rank.
    
    # Try: introduce auxiliary variables that are sums of products
    # This is essentially finding a low-rank decomposition of the 3-mode tensor
    
    return None  # Placeholder - will use optimization instead

def _optimize_decomposition(n, m, p, target_rank, seed=42, max_iters=2000):
    """
    Use gradient-free optimization (alternating least squares with regularization)
    to find a rank-target_rank decomposition.
    """
    np.random.seed(seed)
    
    R = target_rank
    # Initialize
    U = np.random.randn(n*m, R) * 0.1
    V = np.random.randn(m*p, R) * 0.1
    W = np.random.randn(p*n, R) * 0.1
    
    # Reshape
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    ones_p = np.ones(p)
    ones_n = np.ones(n)
    ones_m = np.ones(m)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Solve for U: for each (i,j), find u such that sum_k V[j,k,:] * W[k,i,:] * u ≈ ones(p)
        U3_new = np.zeros_like(U3)
        for i in range(n):
            for j in range(m):
                # M[k,r] = V[j,k,r] * W[k,i,r]
                M = V3[j, :, :] * W3[:, i, :]  # (p, R)
                # Solve M @ u = ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=1e-12)
                    U3_new[i, j, :] = u
                except:
                    U3_new[i, j, :] = U3[i, j, :]
        
        # Solve for V: for each (j,k), find v such that sum_i U[i,j,:] * W[k,i,:] * v ≈ ones(n)
        V3_new = np.zeros_like(V3)
        for j in range(m):
            for k in range(p):
                M = U3[:, j, :] * W3[k, :, :]  # (n, R)
                try:
                    v, _, _, _ = np.linalg.lstsq(M, ones_n, rcond=1e-12)
                    V3_new[j, k, :] = v
                except:
                    V3_new[j, k, :] = V3[j, k, :]
        
        # Solve for W: for each (k,i), find w such that sum_j U[i,j,:] * V[j,k,:] * w ≈ ones(m)
        W3_new = np.zeros_like(W3)
        for k in range(p):
            for i in range(n):
                M = U3[i, :, :] * V3[:, k, :]  # (m, R)
                try:
                    w, _, _, _ = np.linalg.lstsq(M, ones_m, rcond=1e-12)
                    W3_new[k, i, :] = w
                except:
                    W3_new[k, i, :] = W3[k, i, :]
        
        U3, V3, W3 = U3_new, V3_new, W3_new
        U, V, W = U3.reshape(n*m, R), V3.reshape(m*p, R), W3.reshape(p*n, R)
        
        # Compute error
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    # Try to snap to rational values
    if best_err < 1e-8:
        for scale in [1, 2, 4, 8, 16, 32]:
            Us = np.round(best_U * scale) / scale
            Vs = np.round(best_V * scale) / scale
            Ws = np.round(best_W * scale) / scale
            if _verify(Us.tolist(), Vs.tolist(), Ws.tolist(), n, m, p):
                return Us.tolist(), Vs.tolist(), Ws.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0.0] * rank_sb for _ in range(n * m)]
    V_sb = [[0.0] * rank_sb for _ in range(m * p)]
    W_sb = [[0.0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1.0
                V_sb[j*p+k][col] = 1.0
                W_sb[k*n+i][col] = 1.0
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try ranks from low to high
    # Lower bound: max(2*7, 3*7, 2*3) = 21
    # Known: for 2x3x7, rank might be achievable below 42
    target_ranks = list(range(21, 43))
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds, more for lower ranks
        num_seeds = 8 if target_rank < 30 else 4
        
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            result = _optimize_decomposition(n, m, p, target_rank, seed=seed, max_iters=1500)
            if result is not None:
                U_res, V_res, W_res = result
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U_res, "V": V_res, "W": W_res}
                    # Found a good one, try to go lower
                    break
        
        # If we found rank <= 25, we're doing well
        if best_rank <= 25:
            break
    
    return best_result
# EVOLVE-BLOCK-END
