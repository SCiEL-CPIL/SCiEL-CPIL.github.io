"""Rewritten program: search for a lower-rank exact decomposition of the 2x3 * 3x10 matmul tensor."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60

# Deadline guard: keep a safety margin before the 120s wall-clock limit.
DEADLINE = 110.0  # seconds
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return DEADLINE - _elapsed()

def _verify(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i, j, k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k]).
    """
    rank = U.shape[1]
    # Compute the reconstructed tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(N, M, rank)
    V3 = V.reshape(M, P, rank)
    W3 = W.reshape(P, N, rank)
    
    # Use einsum: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # The matmul tensor is all ones (since A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1)
    expected = np.ones((N, M, P))
    return np.allclose(T, expected, atol=tol)

def _schoolbook():
    """Return the trivial rank-60 schoolbook decomposition."""
    U = np.zeros((N * M, RANK_SCHOOLBOOK))
    V = np.zeros((M * P, RANK_SCHOOLBOOK))
    W = np.zeros((P * N, RANK_SCHOOLBOOK))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W

def _try_random_search(max_rank, n_iter=5000, seed=42):
    """Random search for a decomposition of given rank using integer/half-integer entries."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    for it in range(n_iter):
        if _time_left() < 5.0:
            break
        
        # Random integer/half-integer entries in range [-2, 2]
        # Values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
        vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        
        U = np_rng.choice(vals, size=(N * M, max_rank))
        V = np_rng.choice(vals, size=(M * P, max_rank))
        W = np_rng.choice(vals, size=(P * N, max_rank))
        
        if _verify(U, V, W, tol=1e-8):
            return U, V, W
    return None

def _try_structured_search(max_rank):
    """Try structured low-rank decompositions based on known matmul tensor structure."""
    # For 2x3 * 3x10, the tensor has rank at most 60 (schoolbook).
    # Known lower bounds: rank >= max(n*m, m*p, p*n) / something... 
    # Actually, for matrix multiplication, rank >= n*m*p / max(n,m,p) is not tight.
    # Let's try rank values from low to high.
    
    for rank in range(15, max_rank + 1):
        if _time_left() < 10.0:
            break
        result = _try_random_search(rank, n_iter=2000, seed=rank * 1000 + 42)
        if result is not None:
            return result
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
    global _start_time
    _start_time = time.time()
    
    # First, get the schoolbook solution as a fallback
    U_fb, V_fb, W_fb = _schoolbook()
    
    # Try to find a lower-rank decomposition
    # Known: the tensor rank of matmul is at most n*m*p, and at least max(n*m, m*p, p*n)
    # For 2x3, 3x10: n*m=6, m*p=30, p*n=20. So rank >= 30? Not necessarily.
    # Actually the lower bound is more subtle. Let's try searching.
    
    best = (U_fb, V_fb, W_fb)
    best_rank = RANK_SCHOOLBOOK
    
    # Try structured search for lower ranks
    # Start from a reasonable lower bound
    lower_bound = max(N * M, M * P, P * N)  # 30
    # Actually, let's try from a smaller value since some decompositions can be lower
    # Try ranks from 20 up to 59
    for rank in range(20, best_rank):
        if _time_left() < 15.0:
            break
        result = _try_random_search(rank, n_iter=3000, seed=rank * 7919 + 13)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, tol=1e-8):
                best = (U, V, W)
                best_rank = rank
                break  # Found a valid decomposition, return it
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
