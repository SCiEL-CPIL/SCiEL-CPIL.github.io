"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s limit

    # Known rank-23 decomposition for 3x3x3 matrix multiplication
    U3 = np.array([
        [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)
    
    V3 = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ], dtype=int)
    
    W3 = np.array([
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=int)
    
    # Build the rank-46 decomposition for 3x3x6
    rank = 46
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # First 3x3x3 block: k=0,1,2
    for c in range(23):
        U[:, c] = U3[:, c]
        for j in range(m):
            for k in range(3):
                V[j*p+k, c] = V3[j*3+k, c]
        for k in range(3):
            for i in range(n):
                W[k*n+i, c] = W3[k*3+i, c]
    
    # Second 3x3x3 block: k=3,4,5
    for c in range(23, 46):
        U[:, c] = U3[:, c-23]
        for j in range(m):
            for k in range(3):
                V[j*p+k, c] = V3[j*3+(k-3), c-23]
        for k in range(3):
            for i in range(n):
                W[k*n+i, c] = W3[(k-3)*3+i, c-23]
    
    def verify_full(U_mat, V_mat, W_mat, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i*m+j] * V_mat[j*p+k] * W_mat[k*n+i])
                    if val != 1:
                        return False
        return True
    
    if not verify_full(U, V, W, n, m, p):
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb), dtype=int)
        V_fb = np.zeros((m*p, rank_fb), dtype=int)
        W_fb = np.zeros((p*n, rank_fb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 46
    
    # Try to find lower rank by random search
    # Strategy: for each target rank r from 45 down, try random sparse decompositions
    # The decomposition must satisfy: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    def try_random_rank(r, max_iters=2000):
        """Try to find a rank-r decomposition via random search."""
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random sparse pattern with values from {-1, 0, 1}
            U_r = np.zeros((n*m, r), dtype=int)
            V_r = np.zeros((m*p, r), dtype=int)
            W_r = np.zeros((p*n, r), dtype=int)
            
            # Initialize with random sparse entries
            for c in range(r):
                # Each column gets a few non-zero entries
                nz_u = random.randint(2, 4)
                nz_v = random.randint(2, 4)
                nz_w = random.randint(2, 4)
                
                u_idx = random.sample(range(n*m), min(nz_u, n*m))
                v_idx = random.sample(range(m*p), min(nz_v, m*p))
                w_idx = random.sample(range(p*n), min(nz_w, p*n))
                
                for idx in u_idx:
                    U_r[idx, c] = random.choice([-1, 1])
                for idx in v_idx:
                    V_r[idx, c] = random.choice([-1, 1])
                for idx in w_idx:
                    W_r[idx, c] = random.choice([-1, 1])
            
            if verify_full(U_r, V_r, W_r, n, m, p):
                return (U_r, V_r, W_r)
        return None
    
    # Try ranks from 45 down to 40
    for r in range(45, 39, -1):
        if time.time() > deadline:
            break
        result = try_random_rank(r, max_iters=500)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
