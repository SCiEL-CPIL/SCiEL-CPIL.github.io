"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Decomposes into 6 independent 2x2x2 multiplications (column pairs),
    each using Strassen's rank-7 algorithm. Total rank = 6 * 7 = 42.
    
    Returns factor matrices U (4x42), V (24x42), W (24x42) such that:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2    # 6
    rank = num_pairs * 7  # 42
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0.0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]   # 24 x 42
    
    # Strassen pattern: for each term s, define contributions to U, V, W
    # U entries: (u_index, coeff)
    # V entries: (j, local_k, coeff)
    # W entries: (local_k, i, coeff)
    
    strassen_pattern = [
        # s=0: M1
        {'U': [(0, 1), (3, 1)],
         'V': [(0, 0, 1), (1, 1, 1)],
         'W': [(0, 0, 1), (1, 1, 1)]},
        # s=1: M2
        {'U': [(2, 1), (3, 1)],
         'V': [(0, 0, 1)],
         'W': [(0, 1, 1), (1, 1, -1)]},
        # s=2: M3
        {'U': [(0, 1)],
         'V': [(0, 1, 1), (1, 1, -1)],
         'W': [(1, 0, 1), (1, 1, 1)]},
        # s=3: M4
        {'U': [(3, 1)],
         'V': [(1, 0, 1), (0, 0, -1)],
         'W': [(0, 0, 1), (0, 1, 1)]},
        # s=4: M5
        {'U': [(0, 1), (1, 1)],
         'V': [(1, 1, 1)],
         'W': [(0, 0, -1), (1, 0, 1)]},
        # s=5: M6
        {'U': [(2, 1), (0, -1)],
         'V': [(0, 0, 1), (0, 1, 1)],
         'W': [(1, 1, 1)]},
        # s=6: M7
        {'U': [(1, 1), (3, -1)],
         'V': [(1, 0, 1), (1, 1, 1)],
         'W': [(0, 0, 1)]},
    ]
    
    for t in range(num_pairs):
        base_c = t * 7
        k0 = 2 * t
        
        for s in range(7):
            c = base_c + s
            pat = strassen_pattern[s]
            
            for u_idx, coeff in pat['U']:
                U[u_idx][c] += coeff
            
            for j, k_loc, coeff in pat['V']:
                global_k = k0 + k_loc
                v_idx = j * p + global_k
                V[v_idx][c] += coeff
            
            for k_loc, i, coeff in pat['W']:
                global_k = k0 + k_loc
                w_idx = global_k * n + i
                W[w_idx][c] += coeff
        
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
