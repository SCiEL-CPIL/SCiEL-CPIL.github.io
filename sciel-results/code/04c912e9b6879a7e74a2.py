"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen's algorithm applied to each pair of columns.
    For 2x2 x 2xp with p even, group columns into p/2 pairs.
    Each pair uses Strassen's rank-7 decomposition for 2x2x2.
    Total rank = (p/2) * 7 = 6 * 7 = 42 for p=12.
    
    The decomposition satisfies:
    For all i in [0,2), j in [0,2), k in [0,12):
      sum_c U[i*2+j, c] * V[j*12+k, c] * W[k*2+i, c] = 1
    """
    t_start = time.time()
    deadline = t_start + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Pre-defined Strassen components for 2x2x2
    # Each term s has:
    # u_indices: list of (idx, val) for U
    # v_indices: list of (idx_local, val) for V (local column 0 or 1)
    # w_indices: list of (idx_local, i, val) for W (local column 0 or 1, row i)
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00(0), A11(3)
    # V: B00(0,0), B11(1,1) -> j=0,k=0 and j=1,k=1
    # W: C00(0,0), C11(1,1) -> k=0,i=0 and k=1,i=1
    terms = [
        # s=0: M1
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 0, 1), (1, 1, 1)], # (j, k_local, val)
            'w': [(0, 0, 1), (1, 1, 1)]  # (k_local, i, val)
        },
        # s=1: M2 = (A10+A11)*B00
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 0, 1)],
            'w': [(0, 1, 1), (1, 1, -1)] # C10 += M2, C11 -= M2
        },
        # s=2: M3 = A00*(B01-B11)
        {
            'u': [(0, 1)],
            'v': [(0, 1, 1), (1, 1, -1)],
            'w': [(1, 0, 1), (1, 1, 1)] # C01 += M3, C11 += M3
        },
        # s=3: M4 = A11*(B10-B00)
        {
            'u': [(3, 1)],
            'v': [(1, 0, 1), (0, 0, -1)],
            'w': [(0, 0, 1), (0, 1, 1)] # C00 += M4, C10 += M4
        },
        # s=4: M5 = (A00+A01)*B11
        {
            'u': [(0, 1), (1, 1)],
            'v': [(1, 1, 1)],
            'w': [(0, 0, -1), (1, 0, 1)] # C00 -= M5, C01 += M5
        },
        # s=5: M6 = (A10-A00)*(B00+B01)
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 0, 1), (0, 1, 1)],
            'w': [(1, 1, 1)] # C11 += M6
        },
        # s=6: M7 = (A01-A11)*(B10+B11)
        {
            'u': [(1, 1), (3, -1)],
            'v': [(1, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1)] # C00 += M7
        }
    ]
    
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base = t * 7
        k0 = 2 * t      # first column in pair
        k1 = 2 * t + 1  # second column in pair
        
        for s in range(7):
            c = base + s
            term = terms[s]
            
            # Update U
            for u_idx, val in term['u']:
                U[u_idx, c] += val
                
            # Update V
            for j, k_local, val in term['v']:
                global_k = k0 + k_local
                V[j * p + global_k, c] += val
                
            # Update W
            for k_local, i, val in term['w']:
                global_k = k0 + k_local
                W[global_k * n + i, c] += val

    # Convert to lists as expected by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
