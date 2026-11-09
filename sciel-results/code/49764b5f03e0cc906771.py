"""
Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.

Strategy: Slice-and-sum decomposition exploiting linear dependence of tensor slices
along the middle dimension. We partition the middle dimension (size 4) and attempt
to find lower-rank decompositions for the resulting sub-problems.

The schoolbook rank is 3*4*6 = 72. We attempt to find a lower rank by:
1. Partitioning the middle dimension into smaller parts
2. Using known low-rank decompositions for sub-problems where available
3. Combining the decompositions

The tensor condition is:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all valid (i,j,k)
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    return False
    return True

def _try_rank_r(n, m, p, r, time_budget=5.0):
    """
    Try to find a rank-r decomposition using random search with structure.
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    U_shape = (n*m, r)
    V_shape = (m*p, r)
    W_shape = (p*n, r)
    
    # For small rank reductions, try structured random search
    # Use the fact that many entries can be 0, 1, or -1
    
    for attempt in range(50):
        if time.time() - start > time_budget:
            break
        
        # Initialize with random small integers
        U = np.zeros(U_shape, dtype=object)
        V = np.zeros(V_shape, dtype=object)
        W = np.zeros(W_shape, dtype=object)
        
        # For each rank component, assign a pattern
        for r_idx in range(r):
            # Randomly choose which (i,j) pairs this component activates in U
            active_u = set()
            for _ in range(random.randint(1, max(1, n*m//3))):
                idx = random.randint(0, n*m - 1)
                active_u.add(idx)
            
            active_v = set()
            for _ in range(random.randint(1, max(1, m*p//3))):
                idx = random.randint(0, m*p - 1)
                active_v.add(idx)
            
            active_w = set()
            for _ in range(random.randint(1, max(1, p*n//3))):
                idx = random.randint(0, p*n - 1)
                active_w.add(idx)
            
            for idx in active_u:
                U[idx, r_idx] = random.choice([-1, 1])
            for idx in active_v:
                V[idx, r_idx] = random.choice([-1, 1])
            for idx in active_w:
                W[idx, r_idx] = random.choice([-1, 1])
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W

def _slice_sum_decomposition(n, m, p, time_budget=30.0):
    """
    Try slice-and-sum decomposition by partitioning the middle dimension.
    
    Partition m = m1 + m2, then:
    C = A1*B1 + A2*B2
    
    Where A1 is n x m1, B1 is m1 x p, A2 is n x m2, B2 is m2 x p.
    
    The rank is rank(n,m1,p) + rank(n,m2,p).
    """
    start = time.time()
    n, m, p = 3, 4, 6
    best_rank = n * m * p  # Schoolbook
    best_decomp = _schoolbook_decomposition(n, m, p)
    
    # Try different partitions of the middle dimension
    for m1 in range(1, m):
        m2 = m - m1
        if time.time() - start > time_budget:
            break
        
        # For each sub-problem, use schoolbook (or try to find better)
        # Sub-problem 1: n x m1 * m1 x p
        rank1 = n * m1 * p
        # Sub-problem 2: n x m2 * m2 x p
        rank2 = n * m2 * p
        
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            # Build combined decomposition
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Sub-problem 1: A1 (n x m1) * B1 (m1 x p) -> C (n x p)
            # U indices: i*m + j for j in [0, m1)
            # V indices: j*p + k for j in [0, m1)
            # W indices: k*n + i
            col = 0
            for i in range(n):
                for j in range(m1):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            
            # Sub-problem 2: A2 (n x m2) * B2 (m2 x p) -> C (n x p)
            # A2 corresponds to columns m1..m-1 of A
            # B2 corresponds to rows m1..m-1 of B
            for i in range(n):
                for j in range(m1, m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            
            best_rank = total_rank
            best_decomp = (U, V, W)
    
    return best_decomp[0], best_decomp[1], best_decomp[2]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Try slice-and-sum decomposition
    if time.time() < deadline - 60:
        U_ss, V_ss, W_ss = _slice_sum_decomposition(n, m, p, time_budget=30.0)
        if _verify_decomposition(U_ss, V_ss, W_ss, n, m, p):
            rank_ss = len(U_ss[0]) if U_ss else 0
            if rank_ss < best_rank:
                U, V, W = U_ss, V_ss, W_ss
                best_rank = rank_ss
    
    # Try random search for rank best_rank - 1
    if time.time() < deadline - 20 and best_rank > 1:
        result = _try_rank_r(n, m, p, best_rank - 1, time_budget=15.0)
        if result is not None:
            U, V, W = result
            best_rank -= 1
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        best_rank = schoolbook_rank
    
    return {"U": U, "V": V, "W": W}
