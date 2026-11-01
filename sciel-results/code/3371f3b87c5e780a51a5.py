import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    start_time = time.time()
    timeout = 120.0
    
    # Rank 2p = 24 decomposition for 2x2 x 2x12 matrix multiplication
    # Each term r = j*p + k corresponds to the product A[i,j] * B[j,k] -> C[i,k]
    # for all i simultaneously (since U[i*m+j, r] = 1 for both i=0,1)
    
    rank = 2 * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 24 x 24
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # U[i*m + j, r] = 1 for all i
            for i in range(n):
                U[i * m + j][r] = 1
            
            # V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            
            # W[k*n + i, r] = 1 for all i
            for i in range(n):
                W[k * n + i][r] = 1
    
    # Verify decomposition correctness
    def verify(U, V, W, tol=1e-10):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check non-zero entries: T(i,j,k) = 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c_idx = k * n + i
                    a_idx = i * m + j
                    b_idx = j * p + k
                    val = np.dot(U[a_idx, :], np.multiply(V[b_idx, :], W[c_idx, :]))
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    jj = b // p
                    k = b % p
                    kk = c // n
                    ii = c % n
                    if (i == ii and j == jj and k == kk):
                        continue
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(val) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to Strassen-based rank 42 if verification fails
        num_blocks = p // 2
        rank42 = num_blocks * 7
        U42 = [[0] * rank42 for _ in range(n * m)]
        V42 = [[0] * rank42 for _ in range(m * p)]
        W42 = [[0] * rank42 for _ in range(p * n)]
        
        terms = [
            {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
            {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1, (1,1): -1}},
            {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(1,0): 1, (1,1): 1}},
            {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}},
            {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (1,0): 1}},
            {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
            {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
        ]
        
        for b in range(num_blocks):
            for t_idx in range(7):
                r = b * 7 + t_idx
                term = terms[t_idx]
                for (i, j), val in term['U'].items():
                    U42[i * m + j][r] = val
                for (j, k_loc), val in term['V'].items():
                    k_glob = b * 2 + k_loc
                    V42[j * p + k_glob][r] = val
                for (k_loc, i), val in term['W'].items():
                    k_glob = b * 2 + k_loc
                    W42[k_glob * n + i][r] = val
        
        return {"U": U42, "V": V42, "W": W42}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
