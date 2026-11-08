"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 5]
_DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = 1 for all i,j,k
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(n, m, p, rank, max_iters=20000, seed=42):
    """Try to find a rank-`rank` decomposition using random integer/half-integer entries."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Initialize with random small integers
    for r in range(rank):
        U[:, r] = rng.randint(-2, 3, size=n*m)
        V[:, r] = rng.randint(-2, 3, size=m*p)
        W[:, r] = rng.randint(-2, 3, size=p*n)
    
    best_err = float('inf')
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        # Compute current error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += abs(s - 1.0)
        
        if err < 1e-9:
            return U, V, W
        
        if err < best_err:
            best_err = err
            if best_err < 0.01:
                # Try local refinement
                for r in range(rank):
                    for idx in range(n*m):
                        old = U[idx, r]
                        for delta in [-1, -0.5, 0.5, 1]:
                            new_val = old + delta
                            U[idx, r] = new_val
                            # Quick check if this helps
                            new_err = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        s = 0.0
                                        for rr in range(rank):
                                            s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                        new_err += abs(s - 1.0)
                            if new_err < err:
                                err = new_err
                            else:
                                U[idx, r] = old
        # Random perturbation
        r = rng.randint(rank)
        choice = rng.randint(3)
        if choice == 0:
            idx = rng.randint(n*m)
            U[idx, r] += rng.choice([-1, -0.5, 0.5, 1])
        elif choice == 1:
            idx = rng.randint(m*p)
            V[idx, r] += rng.choice([-1, -0.5, 0.5, 1])
        else:
            idx = rng.randint(p*n)
            W[idx, r] += rng.choice([-1, -0.5, 0.5, 1])
    
    return None

def _search_decomposition(n, m, p):
    """Search for a low-rank decomposition, starting from lower ranks."""
    # Known lower bound: for matrix multiplication, rank >= max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But actually the lower bound is more subtle. Let's try ranks from 20 upward.
    start_rank = 20
    for rank in range(start_rank, 61):
        if time.time() > _DEADLINE:
            break
        for seed in range(10):
            if time.time() > _DEADLINE:
                break
            result = _try_rank(n, m, p, rank, max_iters=5000, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists
                U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
                return U_list, V_list, W_list
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a better decomposition
    result = _search_decomposition(n, m, p)
    
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
