"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    We decompose the problem into 6 independent 2x2x2 matrix multiplications
    (pairs of columns). Each 2x2x2 multiplication uses Strassen's algorithm (rank 7).
    
    Total rank = 6 * 7 = 42.
    
    The decomposition satisfies:
    For all i in [0,2), j in [0,2), k in [0,12):
      sum_c U[i*2+j, c] * V[j*12+k, c] * W[k*2+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    # Initialize factor matrices
    # U shape: (n*m, rank) = (4, 42)
    # V shape: (m*p, rank) = (24, 42)
    # W shape: (p*n, rank) = (24, 42)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen's 7-term decomposition for 2x2 matrices.
    # Let A be 2x2, B be 2x2, C = A*B.
    # Indices: A[i,j], B[j,k], C[i,k] with i,j,k in {0,1}.
    #
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We apply this to each pair of columns (k0, k1) in B.
    # U indices: U[i*m + j, c]
    # V indices: V[j*p + k, c]
    # W indices: W[k*n + i, c]
    
    # Pre-compute the Strassen coefficients for reuse
    # Each term s (0..6) defines:
    # u_coeffs: list of (idx_u, coeff)
    # v_coeffs: list of (idx_v, coeff)
    # w_coeffs: list of (idx_w, coeff)
    
    # Pattern for one 2x2x2 block (local indices k_loc 0,1)
    # Global k = 2*t + k_loc
    
    strassen_pattern = []
    
    # s=0: M1
    strassen_pattern.append({
        'U': [(0, 1), (3, 1)],
        'V': [(0, 0, 1), (1, 1, 1)],
        'W': [(0, 0, 1), (1, 1, 1)]
    })
    
    # s=1: M2
    strassen_pattern.append({
        'U': [(2, 1), (3, 1)],
        'V': [(0, 0, 1)],
        'W': [(0, 1, 1), (1, 1, -1)]
    })
    
    # s=2: M3
    strassen_pattern.append({
        'U': [(0, 1)],
        'V': [(0, 1, 1), (1, 1, -1)],
        'W': [(1, 0, 1), (1, 1, 1)]
    })
    
    # s=3: M4
    strassen_pattern.append({
        'U': [(3, 1)],
        'V': [(1, 0, 1), (0, 0, -1)],
        'W': [(0, 0, 1), (0, 1, 1)]
    })
    
    # s=4: M5
    strassen_pattern.append({
        'U': [(0, 1), (1, 1)],
        'V': [(1, 1, 1)],
        'W': [(0, 0, -1), (1, 0, 1)]
    })
    
    # s=5: M6
    strassen_pattern.append({
        'U': [(2, 1), (0, -1)],
        'V': [(0, 0, 1), (0, 1, 1)],
        'W': [(1, 1, 1)]
    })
    
    # s=6: M7
    strassen_pattern.append({
        'U': [(1, 1), (3, -1)],
        'V': [(1, 0, 1), (1, 1, 1)],
        'W': [(0, 0, 1)]
    })
    
    for t in range(num_pairs):
        base_c = t * 7
        k0 = 2 * t
        
        for s in range(7):
            c = base_c + s
            pat = strassen_pattern[s]
            
            # Fill U
            for u_idx, coeff in pat['U']:
                U[u_idx, c] += coeff
            
            # Fill V
            for j, k_loc, coeff in pat['V']:
                global_k = k0 + k_loc
                v_idx = j * p + global_k
                V[v_idx, c] += coeff
            
            # Fill W
            for k_loc, i, coeff in pat['W']:
                global_k = k0 + k_loc
                w_idx = global_k * n + i
                W[w_idx, c] += coeff
                
        if time.time() > deadline:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
