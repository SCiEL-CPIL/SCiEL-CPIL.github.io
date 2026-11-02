"""
Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix 
by a 2 by 8 matrix.

The output factor follows the official trace convention W[k*n+i,r].
Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), 
(m*p, rank), (p*n, rank) with integer/half-integer entries that exactly 
decompose the matrix multiplication tensor; score is the rank.
"""

import time

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Build a set of valid (a,b,c) indices
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
                
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    n, m, p = N, M, P
    
    # Strassen terms for M_{2,2,2}
    # Indices for 2x2 matrix: 00, 01, 10, 11 -> 0, 1, 2, 3
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    num_blocks = p // 2 # 4 blocks
    
    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U: indices i*m + j. m=2.
            # u is length 4, corresponding to (0,0), (0,1), (1,0), (1,1)
            for i in range(4):
                U[i][col] = u[i]
                
            # Fill V: indices j*p + k.
            # Block block_idx uses k = 2*block_idx and 2*block_idx+1.
            # v is length 4, corresponding to:
            # 0: j=0, k_local=0 -> j=0, k_global=2*block_idx
            # 1: j=0, k_local=1 -> j=0, k_global=2*block_idx+1
            # 2: j=1, k_local=0 -> j=1, k_global=2*block_idx
            # 3: j=1, k_local=1 -> j=1, k_global=2*block_idx+1
            for l in range(4):
                if l == 0:
                    b_idx = 0 * p + (2 * block_idx)
                elif l == 1:
                    b_idx = 0 * p + (2 * block_idx + 1)
                elif l == 2:
                    b_idx = 1 * p + (2 * block_idx)
                elif l == 3:
                    b_idx = 1 * p + (2 * block_idx + 1)
                V[b_idx][col] = v[l]
                
            # Fill W: indices k*n + i.
            # Block block_idx uses k = 2*block_idx and 2*block_idx+1.
            # w is length 4, corresponding to:
            # 0: i=0, k_local=0 -> i=0, k_global=2*block_idx
            # 1: i=0, k_local=1 -> i=0, k_global=2*block_idx+1
            # 2: i=1, k_local=0 -> i=1, k_global=2*block_idx
            # 3: i=1, k_local=1 -> i=1, k_global=2*block_idx+1
            for l in range(4):
                if l == 0:
                    c_idx = (2 * block_idx) * n + 0
                elif l == 1:
                    c_idx = (2 * block_idx + 1) * n + 0
                elif l == 2:
                    c_idx = (2 * block_idx) * n + 1
                elif l == 3:
                    c_idx = (2 * block_idx + 1) * n + 1
                W[c_idx][col] = w[l]
                    
    return U, V, W

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

import numpy as np
from scipy.optimize import minimize
import time

def _try_rank_27_search(time_limit=10.0):
    """
    Attempt to find a rank-27 decomposition using numerical optimization.
    Returns (U, V, W) if successful, else None.
    """
    n, m, p = N, M, P
    nm, mp, pn = n*m, m*p, p*n
    rank = 27
    
    # Target tensor T[a, b, c] = 1 if (a,b,c) is valid else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    def error(params):
        U = params[:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = params[nm*rank+mp*rank:].reshape(pn, rank)
        
        # Compute decomposition tensor D[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        D = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Frobenius norm of difference
        return np.linalg.norm(T - D)
    
    start_time = time.time()
    best_error = float('inf')
    
    # Try a few random initializations
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U0 = np.random.randn(nm, rank) * 0.1
        V0 = np.random.randn(mp, rank) * 0.1
        W0 = np.random.randn(pn, rank) * 0.1
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Optimize
        try:
            res = minimize(error, params0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-6})
            if res.fun < best_error:
                best_error = res.fun
                
                if best_error < 1e-4:
                    U_opt = res.x[:nm*rank].reshape(nm, rank)
                    V_opt = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_opt = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Round to nearest 0.5
                    U_half = np.round(U_opt * 2) / 2
                    V_half = np.round(V_opt * 2) / 2
                    W_half = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    if verify(U_half.tolist(), V_half.tolist(), W_half.tolist()):
                        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        except Exception:
            continue
            
    return None

def construct():
    """Entry point: return the best decomposition found."""
    # Attempt to find a lower rank (27) decomposition
    result = _try_rank_27_search(time_limit=10.0)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    # Fallback to rank 28 Strassen decomposition
    U, V, W = strassen_decomposition_228()
    return {"U": U, "V": V, "W": W}
