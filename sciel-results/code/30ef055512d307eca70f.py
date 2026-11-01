import time
import numpy as np

def construct():
    """
    Rank-42 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses 6 blocks of 2 columns. Each block uses Strassen (7 terms).
    """
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 2, 2, 12
    num_pairs = p // 2
    rank_per_pair = 7
    rank = num_pairs * rank_per_pair
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for a 2x2 * 2x2 block
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices for block starting at col c0: 
    #   0,0 -> 0*12 + c0
    #   0,1 -> 0*12 + c1
    #   1,0 -> 1*12 + c0
    #   1,1 -> 1*12 + c1
    # C indices for block starting at col c0:
    #   0,0 -> c0*2 + 0
    #   1,0 -> c0*2 + 1
    #   0,1 -> c1*2 + 0
    #   1,1 -> c1*2 + 1
    
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base = t * rank_per_pair
        c0 = 2 * t
        c1 = 2 * t + 1
        
        # V indices
        v_00 = 0 * p + c0
        v_01 = 0 * p + c1
        v_10 = 1 * p + c0
        v_11 = 1 * p + c1
        
        # W indices
        w_00 = c0 * n + 0
        w_10 = c0 * n + 1
        w_01 = c1 * n + 0
        w_11 = c1 * n + 1
        
        # Term 1: (A00 + A11) * (B00 + B11)
        U[0, base+0] += 1
        U[3, base+0] += 1
        V[v_00, base+0] += 1
        V[v_11, base+0] += 1
        W[w_00, base+0] += 1
        W[w_10, base+0] += 1
        
        # Term 2: (A10 + A11) * B00
        U[2, base+1] += 1
        U[3, base+1] += 1
        V[v_00, base+1] += 1
        W[w_10, base+1] += 1
        W[w_11, base+1] += -1
        
        # Term 3: A00 * (B01 - B11)
        U[0, base+2] += 1
        V[v_01, base+2] += 1
        V[v_11, base+2] += -1
        W[w_01, base+2] += 1
        W[w_11, base+2] += 1
        
        # Term 4: A11 * (B10 - B00)
        U[3, base+3] += 1
        V[v_10, base+3] += 1
        V[v_00, base+3] += -1
        W[w_00, base+3] += 1
        W[w_10, base+3] += 1
        
        # Term 5: (A00 + A01) * B11
        U[0, base+4] += 1
        U[1, base+4] += 1
        V[v_11, base+4] += 1
        W[w_00, base+4] += -1
        W[w_01, base+4] += 1
        
        # Term 6: (A10 - A00) * (B00 + B01)
        U[2, base+5] += 1
        U[0, base+5] += -1
        V[v_00, base+5] += 1
        V[v_01, base+5] += 1
        W[w_11, base+5] += 1
        
        # Term 7: (A01 - A11) * (B10 + B11)
        U[1, base+6] += 1
        U[3, base+6] += -1
        V[v_10, base+6] += 1
        V[v_11, base+6] += 1
        W[w_00, base+6] += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
