"""
Program to find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
The schoolbook rank is 3*3*7 = 63.
We search for lower rank decompositions using a targeted approach.
"""
import time
import numpy as np
import random

_START_TIME = time.time()
TIME_LIMIT = 130.0  # Safety margin before 150s kill

def _check_time():
    return (time.time() - _START_TIME) < TIME_LIMIT

def construct():
    n, m, p = 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def compute_residual(U, V, W):
        """Compute the residual between the tensor and the decomposition."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T - T_approx)
    
    # Try to find rank 62 solution by attempting to merge two terms
    # For each pair of schoolbook terms, check if their sum can be represented as a single rank-1 tensor
    # This is unlikely to work directly, so we try a randomized local search
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Attempt randomized search for rank 62
    if _check_time():
        rank_try = 62
        max_attempts = 20
        for attempt in range(max_attempts):
            if not _check_time():
                break
            
            # Initialize with schoolbook terms, then perturb
            U = U_sb[:, :rank_try].copy()
            V = V_sb[:, :rank_try].copy()
            W = W_sb[:, :rank_try].copy()
            
            # Add small random perturbations
            U += np.random.randn(nm, rank_try) * 0.01
            V += np.random.randn(mp, rank_try) * 0.01
            W += np.random.randn(pn, rank_try) * 0.01
            
            # Local search
            for it in range(2000):
                if not _check_time():
                    break
                which = random.choice([0, 1, 2])
                if which == 0:
                    idx = (random.randint(0, nm-1), random.randint(0, rank_try-1))
                    old = U[idx]
                    U[idx] += random.uniform(-0.1, 0.1)
                elif which == 1:
                    idx = (random.randint(0, mp-1), random.randint(0, rank_try-1))
                    old = V[idx]
                    V[idx] += random.uniform(-0.1, 0.1)
                else:
                    idx = (random.randint(0, pn-1), random.randint(0, rank_try-1))
                    old = W[idx]
                    W[idx] += random.uniform(-0.1, 0.1)
                
                res = compute_residual(U, V, W)
                if res < 1e-10:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank_try
                    break
            
            if best_rank == rank_try:
                break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
