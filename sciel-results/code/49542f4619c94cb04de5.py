import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 138  # seconds, leave 12s margin for 150s limit

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic."""
    rank = len(U[0]) if U else 0
    
    def to_int2(x):
        v = int(round(x * 2))
        if abs(x * 2 - v) > 1e-9:
            return None
        return v
    
    U2 = [[to_int2(x) for x in row] for row in U]
    V2 = [[to_int2(x) for x in row] for row in V]
    W2 = [[to_int2(x) for x in row] for row in W]
    
    for mat in [U2, V2, W2]:
        for row in mat:
            if any(x is None for x in row):
                return False
    
    U3 = np.array(U2).reshape(n, m, rank)
    V3 = np.array(V2).reshape(m, p, rank)
    W3 = np.array(W2).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                if s != 4:
                    return False
    return True

def _build_schoolbook(n, m, p):
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

def _greedy_exact_search(n, m, p, rank, seed, max_steps=2000):
    """
    Greedy exact search for a low-rank decomposition.
    Uses integer values in {-1, 0, 1}.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize with random integers
    U = rng.integers(-1, 2, size=(n * m, rank))
    V = rng.integers(-1, 2, size=(m * p, rank))
    W = rng.integers(-1, 2, size=(p * n, rank))
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Calculate initial error
    def calc_error(U3, V3, W3):
        recon = np.zeros((n, m, p), dtype=np.int32)
        for c in range(rank):
            recon += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        return np.sum((recon - 4) ** 2)
    
    current_error = calc_error(U3, V3, W3)
    
    for step in range(max_steps):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        # Randomly pick a matrix and an index to modify
        mat_choice = rng.integers(0, 3)
        if mat_choice == 0:
            idx = rng.integers(0, n * m)
            col = rng.integers(0, rank)
            old_val = U[idx, col]
            new_val = rng.integers(-1, 2)
            if old_val == new_val: continue
            U[idx, col] = new_val
            U3[idx // m, idx % m, col] = new_val
        elif mat_choice == 1:
            idx = rng.integers(0, m * p)
            col = rng.integers(0, rank)
            old_val = V[idx, col]
            new_val = rng.integers(-1, 2)
            if old_val == new_val: continue
            V[idx, col] = new_val
            V3[idx // p, idx % p, col] = new_val
        else:
            idx = rng.integers(0, p * n)
            col = rng.integers(0, rank)
            old_val = W[idx, col]
            new_val = rng.integers(-1, 2)
            if old_val == new_val: continue
            W[idx, col] = new_val
            W3[idx // n, idx % n, col] = new_val
            
        new_error = calc_error(U3, V3, W3)
        
        if new_error < current_error:
            current_error = new_error
        else:
            # Revert
            if mat_choice == 0:
                U[idx, col] = old_val
                U3[idx // m, idx % m, col] = old_val
            elif mat_choice == 1:
                V[idx, col] = old_val
                V3[idx // p, idx % p, col] = old_val
            else:
                W[idx, col] = old_val
                W3[idx // n, idx % n, col] = old_val
        
        if current_error == 0:
            # Verify with exact arithmetic
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = 2, 4, 7
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 29 to 40
    # We start with lower ranks to find better solutions
    for rank in range(29, 41):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Try a few seeds for each rank
        for seed in range(10):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            res = _greedy_exact_search(n, m, p, rank, seed, max_steps=1000)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
        
        # If we found a solution with this rank, we return it
        # But we want the minimal rank, so we continue if we can
        # However, greedy search might not find the minimal rank easily.
        # If we find a valid solution, it's an improvement over schoolbook (56).
        # We can return the first valid one found to save time, or keep searching for lower rank.
        # Since we iterate from low to high, the first valid one is the lowest rank found by this method.
        # But it might fail for rank 29 and succeed for 30.
        
    return best
