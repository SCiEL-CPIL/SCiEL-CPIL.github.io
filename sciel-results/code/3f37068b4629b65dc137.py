"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j, j',k, i',k'] = delta(j,j')*delta(i,i')*delta(k,k')"""
    # Flatten indices: a = i*m+j (0..nm-1), b = j'*p+k (0..mp-1), c = k'*n+i (0..pn-1)
    # T[a,b,c] = 1 if j==j' and i==i' and k==k'
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T"""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Compute tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    # Compare with true tensor
    T_true = _build_tensor(n, m, p).astype(np.float64)
    return np.allclose(T, T_true, atol=tol)

def _try_random_decomposition(n, m, p, target_rank, num_trials=100, seed=None):
    """Try to find a rank-target_rank decomposition using random initialization and verification."""
    if seed is not None:
        np.random.seed(seed)
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    nm, mp, pn = n*m, m*p, p*n
    for trial in range(num_trials):
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        # Check if this random decomposition matches T
        T_rec = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(target_rank):
            T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        if np.allclose(T_rec, T, atol=1e-6):
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = nm * mp // m  # = n*m*p = 52
    
    # Start with schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank decompositions
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Try ranks from schoolbook-1 down to some minimum
    for target_rank in range(schoolbook_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        result = _try_random_decomposition(n, m, p, target_rank, num_trials=50)
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Convert to list of lists
            best_U = U_arr.tolist()
            best_V = V_arr.tolist()
            best_W = W_arr.tolist()
            best_rank = target_rank
            break  # Found a valid decomposition at this rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
