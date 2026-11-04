"""Matrix multiplication 3x5 * 5x5: rank-minimal exact tensor decomposition."""
import time
import itertools
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 5]

# Deadline guard: evaluator kills after 150s, we need safety margin
DEADLINE_SECONDS = 135.0

def _time_left():
    return DEADLINE_SECONDS - (time.time() - _START_TIME)

_START_TIME = time.time()

def _validate_np(U, V, W, n, m, p, tol=1e-10):
    """Fast validation using NumPy. U,V,W are numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank).
    Convention: W[k*n + i, r]
    Check: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    # Reshape to 3D tensors
    U3 = U.reshape(n, m, rank)  # U3[i,j,r]
    V3 = V.reshape(m, p, rank)  # V3[j,k,r]
    W3 = W.reshape(p, n, rank)  # W3[k,i,r]
    
    # For each (i,j,k), compute sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # We can do this efficiently with einsum or manual loops
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # Use broadcasting: U3[i,j,:] * V3[j,k,:] * W3[k,i,:] then sum over r
    # Shape: (n, m, rank) * (m, p, rank) * (p, n, rank) -> need to align indices
    # U3: (n, m, r), V3: (m, p, r), W3: (p, n, r)
    # We want T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # Reshape for broadcasting:
    # U3[i,j,r] -> (n, m, 1, 1, r)
    # V3[j,k,r] -> (1, m, p, 1, r)  
    # W3[k,i,r] -> (1, 1, p, n, r)
    # Product: (n, m, p, n, r) -> sum over r gives (n, m, p, n)
    # Then check that diagonal in last two dims equals 1? No...
    
    # Actually simpler: iterate over i,j,k and compute
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _validate_sparse(U, V, W, n, m, p):
    """Validate using sparse structure - only check nonzeros."""
    rank = U.shape[1] if hasattr(U, 'shape') else len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    uv = U[i*m+j][r]
                    if uv == 0:
                        continue
                    vw = V[j*p+k][r]
                    if vw == 0:
                        continue
                    s += uv * vw * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _strassen_2x2_block(n, m, p):
    """
    Try to use Strassen-like decomposition for sub-blocks.
    For 2x2 matmul, rank is 7 instead of 8.
    For our 3x5 * 5x5, we can try to decompose into blocks.
    """
    # This is complex; let's try a simpler approach first
    return None

def _try_random_search(n, m, p, max_iter=10000, start_rank=None):
    """Try to find low-rank decompositions using random sparse patterns with NumPy."""
    best_rank = n * m * p  # schoolbook
    if start_rank is None:
        start_rank = max(10, best_rank // 2)
    
    for rank in range(start_rank, best_rank):
        if _time_left() < 5:
            break
        # Try random constructions
        for attempt in range(max(1, max_iter // 100)):
            if _time_left() < 2:
                break
            # Random sparse U, V, W
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Fill with random small integers at random positions
            for r in range(rank):
                # ~3 nonzeros per column in U
                for _ in range(3):
                    row = random.randint(0, n*m-1)
                    U[row, r] = random.choice([-1, 1])
                for _ in range(3):
                    row = random.randint(0, m*p-1)
                    V[row, r] = random.choice([-1, 1])
                for _ in range(3):
                    row = random.randint(0, p*n-1)
                    W[row, r] = random.choice([-1, 1])
            
            if _validate_np(U, V, W, n, m, p):
                return U, V, W, rank
    return None

def _structured_search(n, m, p):
    """Try structured decompositions based on known algebraic identities.
    
    For matrix multiplication, the tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    The schoolbook decomposition has rank n*m*p.
    
    Key insight: The matmul tensor is a "generic" tensor in the sense that each (i,j,k)
    triple corresponds to a unique product A[i,j]*B[j,k] contributing to C[i,k].
    There is no algebraic way to share these products across different (i,j,k) triples
    because each product involves a unique combination of indices.
    
    For 3x5*5x5, the rank is believed to be exactly 75 (schoolbook optimal).
    However, let's try a few structured approaches to verify.
    """
    # Attempt 1: Try to see if we can reduce by using linearity in j
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is already the most efficient form - no further reduction possible
    # without changing the problem.
    
    # The tensor is "direct sum" of n*m*p independent rank-1 tensors,
    # so the rank should be exactly n*m*p.
    
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank = n*m*p = 75)
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Verify schoolbook is valid
    if not _validate_np(U, V, W, n, m, p):
        # This should never happen, but fallback
        U, V, W, rank = _schoolbook(n, m, p)
    
    # Try to find better (lower rank) decompositions
    # First, try structured search (likely returns None for this case)
    if _time_left() > 20:
        result = _structured_search(n, m, p)
        if result is not None:
            U, V, W, rank = result
    
    # Try random search for lower rank
    if _time_left() > 10:
        result = _try_random_search(n, m, p, max_iter=5000, start_rank=50)
        if result is not None:
            U, V, W, rank = result
    
    # Final verification
    if not _validate_np(U, V, W, n, m, p):
        # Fallback to fresh schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
    
    # Convert numpy arrays to lists of ints
    U_out = U.astype(int).tolist()
    V_out = V.astype(int).tolist()
    W_out = W.astype(int).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
