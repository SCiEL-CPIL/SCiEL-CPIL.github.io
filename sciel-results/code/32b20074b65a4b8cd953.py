"""Matrix multiplication 2x4x4 tensor decomposition using Strassen blocks."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    result = _strassen_rank28(n, m, p)
    
    # Quick verification
    if time.time() < deadline - 5:
        if not _verify(result, n, m, p):
            # Fallback: return schoolbook rank-32
            result = _schoolbook_rank32(n, m, p)
    
    return result

def _strassen_rank28(n, m, p):
    rank = 4 * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for b in range(4):
        j_off = 2 * (b // 2)
        k_off = 2 * (b % 2)
        
        for s in range(7):
            col = b * 7 + s
            u_dict, v_dict, w_dict = terms[s]
            
            for x_idx, coeff in u_dict.items():
                i = x_idx // 2
                j_local = x_idx % 2
                j = j_off + j_local
                a_idx = i * m + j
                U[a_idx][col] += coeff
            
            for y_idx, coeff in v_dict.items():
                j_local = y_idx // 2
                k_local = y_idx % 2
                j = j_off + j_local
                k = k_off + k_local
                b_idx = j * p + k
                V[b_idx][col] += coeff
            
            for z_idx, coeff in w_dict.items():
                i = z_idx // 2
                k_local = z_idx % 2
                k = k_off + k_local
                c_idx = k * n + i
                W[c_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_rank32(n, m, p):
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][r] = 1
                V[b_idx][r] = 1
                W[c_idx][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify(result, n, m, p):
    """Verify the decomposition is correct."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    nA = n * m
    nB = m * p
    nC = p * n
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (nA, rank) or V.shape != (nB, rank) or W.shape != (nC, rank):
        return False
    
    # Compute tensor from decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Expected: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    #           T[a,b,c] = 0 otherwise
    
    # For efficiency, only check the non-zero expected entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                # Compute sum_r U[a_idx,r] * V[b_idx,r] * W[c_idx,r]
                val = 0.0
                for r in range(rank):
                    val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that no other (a,b,c) has non-zero value
    # This is expensive, so we do a spot check
    # For each (a,b), check all c
    for a in range(nA):
        for b in range(nB):
            for c in range(nC):
                # Check if this is a valid multiplication entry
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                i2 = c % n
                k2 = c // n
                
                is_valid = (j == j2 and i == i2 and k == k2)
                
                if not is_valid:
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val) > 1e-9:
                        return False
    
    return True
# EVOLVE-BLOCK-END
