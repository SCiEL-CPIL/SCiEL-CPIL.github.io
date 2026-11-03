"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, num_trials=50, trial_range=5):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    rank = U.shape[1]
    
    for _ in range(num_trials):
        A = np.random.randint(-trial_range, trial_range + 1, size=(n, m), dtype=np.int64)
        B = np.random.randint(-trial_range, trial_range + 1, size=(m, p), dtype=np.int64)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=np.int64)
        for r in range(rank):
            for i in range(n):
                a_part = np.dot(U[i*m:(i+1)*m, r], A[i, :])
                for k in range(p):
                    v_vals = V[np.array([j*p + k for j in range(m)]), r]
                    b_part = np.dot(v_vals, B[:, k])
                    C_computed[i, k] += W[k*n + i, r] * a_part * b_part
        
        if not np.array_equal(C_computed, C_expected):
            return False
    return True

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try to find a decomposition with rank < 45 using randomized search
    # with structured initialization
    
    best_result = None
    
    # Try ranks from 35 to 44 (skip 30-34 as they're unlikely)
    for rank in range(35, 45):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_attempts = 30
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Start from schoolbook and perturb
            U = np.zeros((n*m, rank), dtype=np.int64)
            V = np.zeros((m*p, rank), dtype=np.int64)
            W = np.zeros((p*n, rank), dtype=np.int64)
            
            # Initialize with schoolbook-like structure
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U[i*m + j, col] = 1
                            V[j*p + k, col] = 1
                            W[k*n + i, col] = 1
                        col += 1
            
            # If rank < 45, we need to merge some terms
            # Try random linear combinations of the schoolbook terms
            if rank < n*m*p:
                # Randomly combine terms
                for col in range(rank):
                    # Pick 2-3 source terms to combine
                    sources = random.sample(range(n*m*p), min(3, n*m*p))
                    for src in sources:
                        si = src // (m*p)
                        sj = (src // p) % m
                        sk = src % p
                        coeff = random.choice([-1, 1, 1, 1])
                        U[si*m + sj, col] += coeff
                        V[sj*p + sk, col] += coeff
                        W[sk*n + si, col] += coeff
            
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, num_trials=20, trial_range=3):
                if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, num_trials=50, trial_range=10):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook
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
