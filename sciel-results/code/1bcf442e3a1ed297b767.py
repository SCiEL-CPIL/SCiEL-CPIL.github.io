"""Matrix multiplication 4x5x5 tensor decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook solution
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try to find lower rank decomposition over F_7
    # For matrix multiplication tensor, rank is likely 100 for 4x5x5
    # but we'll try a few ranks with structured search
    
    np.random.seed(42)
    
    # Try ranks 99 down to 95 with random search over F_7
    for rank in range(99, 94, -1):
        if time.time() > deadline:
            break
        
        for trial in range(30):
            if time.time() > deadline:
                break
            
            U = np.random.randint(0, 7, size=(n*m, rank))
            V = np.random.randint(0, 7, size=(m*p, rank))
            W = np.random.randint(0, 7, size=(p*n, rank))
            
            # Check over F_7
            T_recon = np.zeros((n*m, m*p, p*n), dtype=int)
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            T_recon = T_recon % 7
            
            if np.array_equal(T_recon, T):
                best_rank = rank
                best_U = U.astype(float)
                best_V = V.astype(float)
                best_W = W.astype(float)
                break
        else:
            continue
        break
    
    # Verify best solution over reals
    if best_rank < rank_schoolbook:
        T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(best_rank):
            T_recon += np.einsum('a,b,c->abc', best_U[:,r], best_V[:,r], best_W[:,r])
        if not np.allclose(T_recon, T, atol=1e-10):
            best_U = U_school
            best_V = V_school
            best_W = W_school
            best_rank = rank_schoolbook
    
    def to_output(mat):
        rounded = np.round(mat * 2) / 2
        if np.allclose(rounded, mat, atol=1e-10):
            return rounded.tolist()
        return mat.tolist()
    
    return {
        "U": to_output(best_U),
        "V": to_output(best_V),
        "W": to_output(best_W)
    }
