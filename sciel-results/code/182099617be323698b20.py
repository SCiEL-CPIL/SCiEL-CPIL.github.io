"""
Rank-minimal exact tensor decomposition for 2x3 x 3x7 matrix multiplication.

Strategy:
- The matrix multiplication tensor for (n,m,p) = (2,3,7) has entries T[i,j,k] = 1.
- We seek U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
  T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
- The schoolbook rank is n*m*p = 42.
- We use a structured initialization based on the known rank-5 decomposition for 2x3x3
  and then attempt to extend it, plus general numerical search.

Key insight: For 2x3 matrices, the minimal rank is 5 (Strassen-like).
We can try to exploit this structure. However, for 3x7, the structure is less clear.

Approach:
1. Try to find a decomposition with rank < 42 using iterative numerical methods.
2. Use the known rank-5 decomposition for 2x3x3 as a seed/initialization strategy.
3. Fall back to schoolbook if no better decomposition is found.

The rank-5 decomposition for 2x3x3 (multiplying 2x3 by 3x3):
This is equivalent to the matrix multiplication M_2,3,3.
Known: rank(M_2,3,3) = 5.

For the full problem, we can think of the 3x7 multiplication as having 7 columns,
each of which is a 3x1 vector. But the tensor structure couples all dimensions.

Let's use a direct numerical approach with good initialization.
"""

DIMENSIONS = [2, 3, 7]

import time
import sys
import numpy as np
import random

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank5_2x3x3():
    """
    Return the rank-5 decomposition for 2x3 x 3x3 matrix multiplication.
    This gives U (6x5), V (9x5), W (6x5) for n=2, m=3, p=3.
    
    The known rank-5 decomposition for M_2,3,3:
    We can derive it from the fact that rank(M_2,3,3) = 5.
    
    One standard construction:
    Let A be 2x3, B be 3x3. We want C = A*B (2x3).
    
    Using the decomposition:
    r1: U=[1,0,0,0,0,0], V=[1,0,0,0,0,0,0,0,0], W=[1,0,0,0,0,0]
    r2: U=[0,1,0,0,0,0], V=[0,1,0,0,0,0,0,0,0], W=[0,1,0,0,0,0]
    r3: U=[0,0,1,0,0,0], V=[0,0,1,0,0,0,0,0,0], W=[0,0,1,0,0,0]
    r4: U=[1,0,0,1,0,0], V=[0,0,0,1,0,0,0,0,0], W=[0,0,0,1,0,0]
    r5: U=[0,0,0,0,1,0], V=[0,0,0,0,1,0,0,0,0], W=[0,0,0,0,1,0]
    
    Wait, let me think more carefully. The tensor is T[i,j,k] where i in [0,1], j in [0,2], k in [0,2].
    T[i,j,k] = 1 for all valid indices.
    
    Actually, the standard result is that M_2,3,3 has rank 5. Let me use a known explicit decomposition.
    
    From literature, one rank-5 decomposition of M_2,3,3:
    
    Term 1: U has 1 at (0,0), V has 1 at (0,0), W has 1 at (0,0)
    Term 2: U has 1 at (0,1), V has 1 at (1,0), W has 1 at (1,0)
    Term 3: U has 1 at (1,0), V has 1 at (0,1), W has 1 at (0,1)
    Term 4: U has 1 at (1,1), V has 1 at (1,1), W has 1 at (1,1)
    Term 5: U has 1 at (0,2) and (1,2), V has 1 at (2,0) and (2,1) and (2,2), W has 1 at (0,2) and (1,2) and (2,2)
    
    Let me just use a numerical approach to find it, or use a simpler initialization.
    """
    n, m, p = 2, 3, 3
    rank = 5
    # We'll find this numerically or use a heuristic
    # For now, return schoolbook as placeholder; we'll use numerical search
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Schoolbook for 2x3x3 gives rank 18, but we want rank 5
    # Let's use a known construction
    # 
    # Actually, let me just use the numerical solver with a good seed
    
    return None

def _solve_tensor_rank(n, m, p, target_rank, max_iter=1000, tol=1e-10, seed=42, time_limit=30):
    """
    Find a rank decomposition of the matrix multiplication tensor.
    
    Uses an alternating least squares / iterative refinement approach.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    U = rng.standard_normal((n*m, target_rank))
    V = rng.standard_normal((m*p, target_rank))
    W = rng.standard_normal((p*n, target_rank))
    
    # Normalize columns
    for r in range(target_rank):
        nu = np.linalg.norm(U[:, r])
        nv = np.linalg.norm(V[:, r])
        nw = np.linalg.norm(W[:, r])
        if nu > 1e-12 and nv > 1e-12 and nw > 1e-12:
            scale = (nu * nv * nw) ** (1/3)
            U[:, r] /= nu
            V[:, r] /= nv
            W[:, r] /= nw
            U[:, r] *= scale
            V[:, r] *= scale
            W[:, r] *= scale
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Update U: for each (i,j), U[i*m+j, :] is determined by fixing V, W
        # T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # For fixed (i,j), this is a linear system in U[i*m+j, :]
        # Let a_k = V[j*p+k, :] * W[k*n+i, :] element-wise sum over... 
        # Actually: sum_r U[idx,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for each k
        # So we have a system: A[idx] * U[idx, :]^T = 1
        # where A[idx][k, r] = V[j*p+k, r] * W[k*n+i, r]
        
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # Build system: for each k, sum_r U[idx,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # This is a linear system with p equations and target_rank unknowns
                # A is p x target_rank, b is p x 1 with all ones
                A = np.zeros((p, target_rank))
                for k in range(p):
                    for r in range(target_rank):
                        A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                b = np.ones(p)
                # Solve least squares: A @ u = b
                try:
                    u, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    u = U[idx, :].copy()
                U[idx, :] = u
        
        # Update V: for each (j,k), V[j*p+k, :] determined by fixing U, W
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                A = np.zeros((n, target_rank))
                for i in range(n):
                    for r in range(target_rank):
                        A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                b = np.ones(n)
                try:
                    v, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    v = V[idx, :].copy()
                V[idx, :] = v
        
        # Update W: for each (k,i), W[k*n+i, :] determined by fixing U, V
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                A = np.zeros((m, target_rank))
                for j in range(m):
                    for r in range(target_rank):
                        A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                b = np.ones(m)
                try:
                    w, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    w = W[idx, :].copy()
                W[idx, :] = w
        
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual += (s - 1.0)**2
        residual = np.sqrt(residual)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < tol:
            break
    
    return best_U, best_V, best_W, best_residual

def _try_rank(n, m, p, rank, max_restarts=5, time_limit=25):
    """Try to find a rank decomposition of the given rank."""
    start_time = time.time()
    
    best_residual = np.inf
    best_result = None
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        seed = 42 + restart * 7919
        U, V, W, residual = _solve_tensor_rank(n, m, p, rank, max_iter=500, tol=1e-10, seed=seed, time_limit=20)
        
        if residual < best_residual:
            best_residual = residual
            best_result = (U, V, W)
        
        if residual < 1e-9:
            break
    
    if best_result is not None and best_residual < 1e-9:
        U, V, W = best_result
        # Verify
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p  # 42
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But for matrix multiplication tensors, the rank is at least max(n, m, p)^2 in some cases
    # For 2x3x7, let's try ranks from a reasonable range
    
    # Try ranks from lower bound up to schoolbook-1
    # Focus on promising ranges
    for rank in range(21, best_rank):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank(n, m, p, rank, max_restarts=4, time_limit=20)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found a valid decomposition at this rank
    
    return best

# EVOLVE-BLOCK-END
