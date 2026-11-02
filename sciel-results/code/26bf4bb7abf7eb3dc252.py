"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # First, try to find rank 16 via search
    rank16_result = _search_rank(n, m, p, 16, deadline)
    if rank16_result is not None:
        return rank16_result
    
    # Fallback: rank 17 explicit construction
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_rank(n, m, p, target_rank, deadline):
    """Attempt to find a rank-target_rank decomposition via random search."""
    # Build the multiplication tensor T[i, j, k] = 1 if (i,j,k) is a valid triple
    # T has shape (n*m, m*p, p*n) with T[i,j,k] = 1 iff i,j,k form a valid matmul term
    # i = row_a * m + col_a, j = row_b * p + col_b, k = col_c * n + row_c
    # Valid if col_a == row_b and row_a == row_c
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Try random integer/half-integer matrices
    rng = np.random.RandomState(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        U = np.zeros((n*m, target_rank), dtype=np.float64)
        V = np.zeros((m*p, target_rank), dtype=np.float64)
        W = np.zeros((p*n, target_rank), dtype=np.float64)
        
        # Initialize with random small integers
        for r in range(target_rank):
            U[:, r] = rng.choice([-2, -1, 0, 1, 2], size=n*m)
            V[:, r] = rng.choice([-2, -1, 0, 1, 2], size=m*p)
            W[:, r] = rng.choice([-2, -1, 0, 1, 2], size=p*n)
        
        # Check if decomposition is exact
        if _check_decomposition(U, V, W, T, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try local search: perturb one entry
        for _ in range(50):
            if time.time() > deadline - 5:
                    break
            
            r = rng.randint(target_rank)
            mat_choice = rng.randint(3)
            if mat_choice == 0:
                idx = rng.randint(n*m)
                U[idx, r] += rng.choice([-1, 1])
            elif mat_choice == 1:
                idx = rng.randint(m*p)
                V[idx, r] += rng.choice([-1, 1])
            else:
                idx = rng.randint(p*n)
                W[idx, r] += rng.choice([-1, 1])
            
            if _check_decomposition(U, V, W, T, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _check_decomposition(U, V, W, T, n, m, p):
    """Check if U,V,W exactly decompose T."""
    # Reconstruct: T_recon[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Efficient: for each r, outer product
    T_recon = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    return np.allclose(T_recon, T, atol=1e-10)
# EVOLVE-BLOCK-END
