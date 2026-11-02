"""Rewrite for matrix multiplication 3x3x5 tensor decomposition.

Searches for rank-minimal exact tensor decomposition.
Uses structured initialization + local search to find lower-rank solutions.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]


def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a list of (u,v,w) indices with value 1."""
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                entries.append((u, v, w))
    return entries


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that sum_r U[u,r]*V[v,r]*W[w,r] equals target tensor."""
    rank = len(U[0]) if U else 0
    target = _build_target_tensor(n, m, p)
    target_set = set(target)
    
    # Check all target entries equal 1
    for (u, v, w) in target:
        val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
        if abs(val - 1.0) > tol:
            return False
    
    # Check non-target entries equal 0
    all_u = range(n * m)
    all_v = range(m * p)
    all_w = range(p * n)
    for u in all_u:
        for v in all_v:
            for w in all_w:
                if (u, v, w) not in target_set:
                    val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                    if abs(val) > tol:
                        return False
    return True


def _schoolbook(n, m, p):
    """Return schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random rank decomposition."""
    if seed is not None:
        random.seed(seed)
    U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}


def _als_optimize(U, V, W, n, m, p, max_iters=100, tol=1e-10, time_limit=5):
    """Optimize factors using Alternating Least Squares."""
    start_time = time.time()
    rank = len(U[0])
    target = _build_target_tensor(n, m, p)
    target_set = set(target)
    
    # Precompute index mappings
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        for u in range(n * m):
            # Build A_u and t_u
            A = np.zeros((m * p * p * n, rank))
            t = np.zeros(m * p * p * n)
            idx = 0
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) in target_set:
                        t[idx] = 1.0
                    # A[idx, r] = V[v, r] * W[w, r]
                    for r in range(rank):
                        A[idx, r] = V[v][r] * W[w][r]
                    idx += 1
            # Solve least squares
            if rank > 0:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A, t, rcond=None)
                for r in range(rank):
                    U[u][r] = sol[r]
        
        # Update V
        for v in range(m * p):
            A = np.zeros((n * m * p * n, rank))
            t = np.zeros(n * m * p * n)
            idx = 0
            for u in range(n * m):
                for w in range(p * n):
                    if (u, v, w) in target_set:
                        t[idx] = 1.0
                    for r in range(rank):
                        A[idx, r] = U[u][r] * W[w][r]
                    idx += 1
            if rank > 0:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A, t, rcond=None)
                for r in range(rank):
                    V[v][r] = sol[r]
        
        # Update W
        for w in range(p * n):
            A = np.zeros((n * m * m * p, rank))
            t = np.zeros(n * m * m * p)
            idx = 0
            for u in range(n * m):
                for v in range(m * p):
                    if (u, v, w) in target_set:
                        t[idx] = 1.0
                    for r in range(rank):
                        A[idx, r] = U[u][r] * V[v][r]
                    idx += 1
            if rank > 0:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A, t, rcond=None)
                for r in range(rank):
                    W[w][r] = sol[r]
        
        # Check convergence
        residual = 0.0
        for (u, v, w) in target:
            val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
            residual += (val - 1.0)**2
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) not in target_set:
                        val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                        residual += val**2
        
        if residual < tol:
            break
    
    return U, V, W, residual

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p  # 45
    
    # Try to find lower-rank decompositions using ALS
    # We try ranks from a lower bound up to 44
    # Lower bound is max(n*m, m*p, p*n) = 15
    # Let's try from 20 to 44 to be safe with time
    
    random.seed(42)
    np.random.seed(42)
    
    # We will try to optimize from a random start for each rank
    # If we find a valid solution, we update best_result
    
    for rank in range(25, best_rank):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random small values
            U = [[random.uniform(-0.1, 0.1) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.uniform(-0.1, 0.1) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.uniform(-0.1, 0.1) for _ in range(rank)] for _ in range(p * n)]
            
            # Optimize
            U_opt, V_opt, W_opt, residual = _als_optimize(U, V, W, n, m, p, max_iters=50, time_limit=5)
            
            if residual < 1e-6:
                # Verify exactly
                if _verify(U_opt, V_opt, W_opt, n, m, p):
                    best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = rank
                    break
        
        if best_rank < n * m * p:
            break
    
    return best_result
