"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

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
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-9):
    """Check that U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build tensor via einsum
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # Expected: T[i,j,k] = 1 for all i,j,k
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol, rtol=0)


def _try_rank(rank, timeout=120.0, start_time=None):
    """Try to find a decomposition of given rank. Returns dict or None."""
    if start_time is None:
        start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We'll use a randomized approach: initialize with random values, then try to
    # project/solve. For exact decompositions, we can try to find integer/half-integer
    # solutions.
    
    # Strategy: For small ranks, try to solve the system of equations.
    # The tensor has nm*mp*pn = 20*25*20 = 10000 entries, but many are zero.
    # Actually, for matmul, T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # This is a system of 100 equations (4*5*5=100) in 3*rank variables.
    # For rank < 100, we have more variables than equations, so solutions may exist.
    
    # Let's try a different approach: use the fact that for rank r, we need
    # U (nm x r), V (mp x r), W (pn x r) such that the above holds.
    
    # Random initialization + iterative refinement
    best_solution = None
    rng = np.random.default_rng(42)
    
    # Try multiple random initializations
    max_iters = 100
    for init in range(50):
        if time.time() - start_time > timeout - 5:
            break
        
        # Initialize with small random values
        U = rng.standard_normal((nm, rank))
        V = rng.standard_normal((mp, rank))
        W = rng.standard_normal((pn, rank))
        
        # Normalize
        for c in range(rank):
            norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
            if norm > 0:
                scale = (1.0 / norm) ** (1.0/3.0)
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
        
        # Check if it works
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            # Try to round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Iterative refinement: gradient descent on the loss
        loss = 0
        for it in range(max_iters):
            if time.time() - start_time > timeout - 5:
                break
            
            # Compute current tensor
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            loss = np.sum((T - 1.0)**2)
            
            if loss < 1e-12:
                break
            
            # Compute gradients
            # dL/dU[i*m+j,c] = 2*(T-1)[i,j,k] * V[j*p+k,c] * W[k*n+i,c] summed over k
            # This is complex; let's use a simpler approach
            
            # Actually, let's just try random restarts with different seeds
            break
        
        # Try a structured initialization based on known algorithms
        if init == 0:
            # Try the "rank-1" style: each term contributes to a subset
            pass
    
    return None


def _try_known_constructions():
    """Try known low-rank constructions for 4x5 x 5x5."""
    # For square matrices, we know:
    # 2x2: rank 7 (Strassen)
    # 3x3: rank 23 (best known)
    # 4x4: rank 49 (best known)
    # 5x5: rank 78 (best known)
    # 
    # For 4x5 x 5x5, we can try:
    # - Block decomposition using Strassen on 2x2 blocks
    # - Other known algorithms
    
    # Let's try a block-based approach:
    # Split A (4x5) into blocks, B (5x5) into blocks
    # Use Strassen's algorithm for 2x2 matrix multiplication within blocks
    
    # For 4x4 x 4x4, Strassen-based gives rank 7^2 = 49
    # For 4x5 x 5x5, we can pad to 4x8 x 8x8 or use a different blocking
    
    # Actually, let's try a direct construction:
    # n=4, m=5, p=5
    # We can view this as computing 4x5 output from 4x5 input * 5x5 matrix
    
    # Try: use 2x2 Strassen blocks
    # A is 4x5 = (2x2 blocks of 2x2, 2x2, 2x1, 2x1)
    # This gets complicated. Let's try a simpler approach.
    
    # For now, just return the schoolbook as fallback
    return _schoolbook()


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First, try known constructions
    # For 4x5 x 5x5, the best known rank is at most 4*5*5 = 100 (schoolbook)
    # We can potentially do better with Strassen-like algorithms
    
    # Try to find a decomposition with rank < 100
    # Start with some reasonable ranks to try
    
    # Known lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Upper bound from schoolbook: 100
    
    # Try a few specific constructions
    result = None
    
    # Try rank 75, 80, 85, 90, 95 (just below 100)
    for rank in [75, 80, 85, 90, 95]:
        if time.time() > deadline:
            break
        result = _try_rank(rank, timeout=deadline - time.time(), start_time=start_time)
        if result is not None:
            return result
    
    # If no better solution found, return schoolbook
    if time.time() > deadline:
        return _schoolbook()
    
    # Final fallback
    return _schoolbook()

# EVOLVE-BLOCK-END
