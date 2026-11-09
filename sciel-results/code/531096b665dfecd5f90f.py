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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """
    Verify if the decomposition exactly multiplies matrices.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Convention: W index is k*n + i.
    T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
    We want T[i, j, k] = 1 for all valid i, j, k.
    """
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    # Check shapes
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
    
    for _ in range(500):
        if time.time() - start_time > timeout:
            break
        # Use smaller range for better chance
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        W = np.random.randint(-1, 2, size=(p * n, target_rank))
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def _try_structured_search(target_rank, timeout=10):
    """
    Try structured search using the fact that the target tensor is all-ones.
    We can try to find a decomposition where the vectors have structure.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try using the property that sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    # for all i, j, k.
    
    # One approach: use the fact that we can write 1 as a sum of products.
    # For example, if we use rank = n*m + m*p + p*n - something, we might find cancellations.
    
    # Let's try a more systematic approach: start with schoolbook and try to merge terms.
    # Or, try random search with better initialization.
    
    for attempt in range(200):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with random values in {-1, 0, 1}
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Fill with random values
        for _ in range(target_rank):
            # Randomly choose which entries to set
            u_idx = random.randint(0, n * m - 1)
            v_idx = random.randint(0, m * p - 1)
            w_idx = random.randint(0, p * n - 1)
            
            u_val = random.choice([-1, 1])
            v_val = random.choice([-1, 1])
            w_val = random.choice([-1, 1])
            
            U[u_idx, _] = u_val
            V[v_idx, _] = v_val
            W[w_idx, _] = w_val
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    deadline = time.time() + 110  # 10s margin for 120s limit
    
    # 1. Try schoolbook (guaranteed)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # 2. Try to find lower rank
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25.
    
    for r in range(74, 24, -1):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        timeout = min(5, remaining / 2)
        
        # First try random search
        res = _try_random_rank(r, timeout=timeout)
        if res:
            return res
        
        # Then try structured search
        remaining = deadline - time.time()
        if remaining > 5:
            res = _try_structured_search(r, timeout=min(5, remaining / 2))
            if res:
                return res
        
        if time.time() > deadline:
            break
            
    return best
# EVOLVE-BLOCK-END
