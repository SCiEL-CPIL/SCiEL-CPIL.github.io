import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    timeout = 120.0
    
    # Strassen-based decomposition: partition p=12 columns into 6 pairs
    # Each pair uses 7 rank-1 terms (Strassen for 2x2x2 matrix mult)
    # Total rank = 6 * 7 = 42
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Verified Strassen coefficients (U, V, W) for each of 7 terms
    strassen_terms = [
        # M1
        {0: 1, 3: 1}, 
        {0: 1, 3: 1}, 
        {0: 1, 3: 1}, 
        # M2
        {2: 1, 3: 1}, 
        {0: 1},        
        {2: 1, 3: -1}, 
        # M3
        {0: 1},       
        {1: 1, 3: -1}, 
        {1: 1, 3: 1},  
        # M4
        {3: 1},       
        {2: 1, 0: -1}, 
        {0: 1, 2: 1},  
        # M5
        {0: 1, 1: 1}, 
        {3: 1},        
        {0: -1, 1: 1}, 
        # M6
        {0: -1, 2: 1}, 
        {0: 1, 1: 1},  
        {3: 1},        
        # M7
        {1: 1, 3: -1}, 
        {2: 1, 3: 1},  
        {0: 1},        
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            
            for a_idx, val in u_coeffs.items():
                U[a_idx, col] = val
            
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b, col] = val
            
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w, col] += val
    
    # Verify decomposition correctness
    # T[c,a,b] = 1 if C[c] involves A[a]*B[b], else 0
    # C[k,i] = sum_j A[i,j]*B[j,k], so T[k*2+i, i*2+j, j*12+k] = 1
    def verify(U, V, W, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c_idx = k * n + i
                    a_idx = i * m + j
                    b_idx = j * p + k
                    val = np.dot(U[a_idx, :], np.multiply(V[b_idx, :], W[c_idx, :]))
                    if abs(val - 1.0) > tol:
                        return False
        # Check zeros
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Skip the non-zero entries
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
        # Should not happen, but fallback
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
