"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import itertools
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _build_rank(r, entries_U, entries_V, entries_W):
    """Build U, V, W matrices from list of tuples."""
    n, m, p = DIMENSIONS
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    for c in range(r):
        ui, uj, uval = entries_U[c]
        vi, vj, vval = entries_V[c]
        wi, wj, wval = entries_W[c]
        U[ui][c] = uval
        V[vi][c] = vval
        W[wi][c] = wval
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 if i,j,k are valid indices (which they always are for the multiplication)
    # Actually, the tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1 if this term contributes to C[i,k]
    # In the decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _random_rank(r, trial_seed=None):
    """Try to find a rank-r decomposition by random search."""
    n, m, p = DIMENSIONS
    if trial_seed is not None:
        random.seed(trial_seed)
    
    # Random initialization
    U = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(r)] for _ in range(n * m)]
    V = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(r)] for _ in range(m * p)]
    W = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(r)] for _ in range(p * n)]
    
    # Simple local search: try to fix entries
    for it in range(50):
        # Pick a random (i,j,k) that's wrong
        bad = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                    if abs(s - 1) > 1e-9:
                        bad.append((i, j, k))
        if not bad:
            return U, V, W
        i, j, k = random.choice(bad)
        # Try to fix by adjusting one entry
        c = random.randrange(r)
        choices = [0, 1, -1, 0.5, -0.5, 2, -2]
        old = U[i*m+j][c]
        for new_val in choices:
            U[i*m+j][c] = new_val
            if _verify(U, V, W):
                return U, V, W
        U[i*m+j][c] = old
        old = V[j*p+k][c]
        for new_val in choices:
            V[j*p+k][c] = new_val
            if _verify(U, V, W):
                return U, V, W
        V[j*p+k][c] = old
        old = W[k*n+i][c]
        for new_val in choices:
            W[k*n+i][c] = new_val
            if _verify(U, V, W):
                return U, V, W
        W[k*n+i][c] = old
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s budget
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 48
    
    # Try to find better decompositions
    # Known results: for 3x4 * 4x4, the tensor rank is at most 48 (schoolbook)
    # Let's try ranks below 48
    
    # Start with schoolbook as fallback
    best_result = None
    best_rank = rank_schoolbook
    
    # Build schoolbook
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_result = {"U": U, "V": V, "W": W}
    
    # Try lower ranks
    # The theoretical lower bound for 3x4*4x4 is at least max(n*p, n*m, m*p) = max(12, 12, 16) = 16
    # But practical lower bounds are higher. Let's try from rank 47 down to some limit.
    
    # For each rank, try a few random initializations
    for r in range(rank_schoolbook - 1, max(15, rank_schoolbook - 20), -1):
        if time.time() > deadline:
            break
        
        # Try a few random seeds
        for seed in range(3):
            if time.time() > deadline:
                break
            
            result = _random_rank(r, trial_seed=seed + r * 100)
            if result is not None:
                U_r, V_r, W_r = result
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = r
                # If we found a good rank, we can stop early if it's quite low
                if r <= 30:
                    return best_result
                break
    
    return best_result
# EVOLVE-BLOCK-END
