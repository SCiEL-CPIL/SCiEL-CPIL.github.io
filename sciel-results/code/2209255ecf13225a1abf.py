"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_random_rank(r, n, m, p, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition via random search."""
    rng = np.random.RandomState(seed)
    for it in range(max_iters):
        # Random sparse initialization with small integer entries
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        for c in range(r):
            # Each column has a few non-zero entries
            nnz_u = rng.randint(1, 4)
            nnz_v = rng.randint(1, 4)
            nnz_w = rng.randint(1, 4)
            
            rows_u = rng.choice(n*m, size=nnz_u, replace=False)
            rows_v = rng.choice(m*p, size=nnz_v, replace=False)
            rows_w = rng.choice(p*n, size=nnz_w, replace=False)
            
            U[rows_u, c] = rng.choice([-1, 1], size=nnz_u)
            V[rows_v, c] = rng.choice([-1, 1], size=nnz_v)
            W[rows_w, c] = rng.choice([-1, 1], size=nnz_w)
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s safety margin
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bound is at least max(n*m, m*p, n*p) = 25
    # Try ranks from a reasonable estimate downward
    # For 3x5x5, let's try from 40 down to 25
    for r in range(40, 24, -1):
        if time.time() > deadline - 30:  # Stop if less than 30s remaining
            break
        result = _try_random_rank(r, n, m, p, max_iters=500, seed=r*1000+42)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break  # Found a decomposition at this rank, try lower
    
    # Convert to lists of lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Ensure entries are integer or half-integer
    # Convert to proper format
    U_out = []
    for row in U_list:
        U_out.append([x if x == int(x) else x for x in row])
    V_out = []
    for row in V_list:
        V_out.append([x if x == int(x) else x for x in row])
    W_out = []
    for row in W_list:
        W_out.append([x if x == int(x) else x for x in row])
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
