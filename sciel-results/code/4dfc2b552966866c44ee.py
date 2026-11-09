"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

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
    return {"U": U, "V": W, "W": W}

def _verify_fast(U, V, W):
    """
    Fast verification using einsum.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Convention: W index is k*n + i.
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    We want T[i,j,k] = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != r or len(V[0]) != r or len(W[0]) != r:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to 3D
    U3 = U_np.reshape(n, m, r)  # U[i, j, c]
    V3 = V_np.reshape(m, p, r)  # V[j, k, c]
    W3 = W_np.reshape(p, n, r)  # W[k, i, c]
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Check if all entries are 1
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _verify(U, V, W):
    """
    Verify if the decomposition exactly multiplies matrices.
    """
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != r or len(V[0]) != r or len(W[0]) != r:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U_np[i * m + j]
                v = V_np[j * p + k]
                w = W_np[k * n + i]
                s = np.dot(u, v * w)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _try_random_rank(target_rank, timeout=5):
    """
    Attempt to find a random rank decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    for _ in range(2000):
        if time.time() - start_time > timeout:
            break
        U = np.random.randint(-2, 3, size=(n * m, target_rank))
        V = np.random.randint(-2, 3, size=(m * p, target_rank))
        W = np.random.randint(-2, 3, size=(p * n, target_rank))
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def _structured_decomposition():
    """
    Try to find a structured low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    
    for rank in range(25, 50):
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c in range(rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            
            U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i, c] = 1
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        for trial in range(200):
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                
                U[i * m + j, c] = random.choice([-1, 1])
                V[j * p + k, c] = random.choice([-1, 1])
                W[k * n + i, c] = random.choice([-1, 1])
            
            if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _systematic_search(target_rank, timeout=10):
    """
    More systematic search using a grid of possible values.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    values = [-2, -1, 0, 1, 2]
    
    for _ in range(10000):
        if time.time() - start_time > timeout:
            break
        
        U = np.random.choice(values, size=(n * m, target_rank))
        V = np.random.choice(values, size=(m * p, target_rank))
        W = np.random.choice(values, size=(p * n, target_rank))
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_half_integer_search(target_rank, timeout=5):
    """
    Try search with half-integer values.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    values = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    for _ in range(5000):
        if time.time() - start_time > timeout:
            break
        
        U = np.random.choice(values, size=(n * m, target_rank))
        V = np.random.choice(values, size=(m * p, target_rank))
        W = np.random.choice(values, size=(p * n, target_rank))
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_sparse_search(target_rank, timeout=10):
    """
    Try sparse search where most entries are 0.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    for _ in range(5000):
        if time.time() - start_time > timeout:
            break
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Randomly place non-zero entries
        for c in range(target_rank):
            # Choose random positions
            i_u = random.randint(0, n * m - 1)
            i_v = random.randint(0, m * p - 1)
            i_w = random.randint(0, p * n - 1)
            
            val = random.choice([-2, -1, 1, 2])
            U[i_u, c] = val
            V[i_v, c] = val
            W[i_w, c] = val
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    deadline = time.time() + 140  # 10s margin for 150s limit
    
    # 1. Try schoolbook (guaranteed)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # 2. Try structured decomposition
    if time.time() < deadline - 40:
        res = _structured_decomposition()
        if res:
            return res
    
    # 3. Try systematic search for lower ranks
    for r in range(74, 25, -1):
        if time.time() > deadline - 20:
            break
        
        remaining_time = deadline - time.time() - 15
        if remaining_time < 5:
            break
        timeout = min(8, remaining_time / 8)
        
        res = _systematic_search(r, timeout=timeout)
        if res:
            return res
        
        # Also try random search
        timeout2 = min(4, remaining_time / 8)
        res = _try_random_rank(r, timeout=timeout2)
        if res:
            return res
        
        # Try half-integer search
        if time.time() < deadline - 18:
            res = _try_half_integer_search(r, timeout=min(3, (deadline - time.time() - 18)))
            if res:
                return res
        
        # Try sparse search
        if time.time() < deadline - 16:
            res = _try_sparse_search(r, timeout=min(3, (deadline - time.time() - 16)))
            if res:
                return res
            
    return best
# EVOLVE-BLOCK-END
