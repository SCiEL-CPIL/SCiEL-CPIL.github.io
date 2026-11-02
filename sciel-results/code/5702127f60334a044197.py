"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try ranks from 30 to 55, looking for a lower-rank decomposition
    for rank in range(30, 56):
        if time.time() > deadline - 15:
            break
        res = _search_rank(n, m, p, rank, deadline)
        if res is not None:
            return res
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _search_rank(n, m, p, rank, deadline):
    """Search for a rank-`rank` decomposition using coordinate descent."""
    max_restarts = 30
    for _ in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with half-integer entries
        U = np.random.choice([-1, 0, 1], size=(n*m, rank)) * 0.5
        V = np.random.choice([-1, 0, 1], size=(m*p, rank)) * 0.5
        W = np.random.choice([-1, 0, 1], size=(p*n, rank)) * 0.5
        
        # Coordinate descent
        for it in range(300):
            if time.time() > deadline - 10:
                break
            
            if _verify(U, V, W, n, m, p, rank):
                return _to_lists(U, V, W, n, m, p, rank)
            
            # Perturb a random entry
            choice = random.random()
            if choice < 1/3:
                r = random.randint(0, n*m - 1)
                c = random.randint(0, rank - 1)
                old = U[r, c]
                for delta in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]:
                    U[r, c] = old + delta
                    if _verify(U, V, W, n, m, p, rank):
                        return _to_lists(U, V, W, n, m, p, rank)
                U[r, c] = old
            elif choice < 2/3:
                r = random.randint(0, m*p - 1)
                c = random.randint(0, rank - 1)
                old = V[r, c]
                for delta in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]:
                    V[r, c] = old + delta
                    if _verify(U, V, W, n, m, p, rank):
                        return _to_lists(U, V, W, n, m, p, rank)
                V[r, c] = old
            else:
                r = random.randint(0, p*n - 1)
                c = random.randint(0, rank - 1)
                old = W[r, c]
                for delta in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]:
                    W[r, c] = old + delta
                    if _verify(U, V, W, n, m, p, rank):
                        return _to_lists(U, V, W, n, m, p, rank)
                W[r, c] = old
    
    return None

def _verify(U, V, W, n, m, p, rank):
    """Check if U, V, W form a valid decomposition."""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=1e-10)

def _to_lists(U, V, W, n, m, p, rank):
    """Convert numpy arrays to lists of lists."""
    U_l = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
    V_l = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
    W_l = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
    return {"U": U_l, "V": V_l, "W": W_l}
# EVOLVE-BLOCK-END
