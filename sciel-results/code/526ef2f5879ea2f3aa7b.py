"""Matrix multiplication 3,4,8 rank-88 decomposition with deadline guard."""
DIMENSIONS = [3, 4, 8]
import time

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8 block
    # 4 * 8 = 32 terms
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Check deadline
    if time.time() - start_time > 140:
        return {"U": U, "V": V, "W": W}
    
    # Part 2: i=1,2 using Strassen 2x2x2 blocks
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    # 2 * 4 * 7 = 56 terms
    
    # Strassen local factors
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Precompute non-zero indices for each strassen term to speed up loops
    # Format: list of (local_index, value)
    U_nz = []
    V_nz = []
    W_nz = []
    for t in range(7):
        U_nz.append([(l, strassen_U[t][l]) for l in range(4) if strassen_U[t][l] != 0])
        V_nz.append([(l, strassen_V[t][l]) for l in range(4) if strassen_V[t][l] != 0])
        W_nz.append([(l, strassen_W[t][l]) for l in range(4) if strassen_W[t][l] != 0])
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            for t in range(7):
                # U: map local A indices
                # l=0 -> i'=0, j'=0 -> gi=1, gj=j_start
                # l=1 -> i'=0, j'=1 -> gi=1, gj=j_start+1
                # l=2 -> i'=1, j'=0 -> gi=2, gj=j_start
                # l=3 -> i'=1, j'=1 -> gi=2, gj=j_start+1
                for l, val in U_nz[t]:
                    li = l // 2
                    lj = l % 2
                    gi = 1 + li
                    gj = j_start + lj
                    U[gi * m + gj][col] += val
                    
                # V: map local B indices
                # l=0 -> j'=0, k'=0 -> gj=j_start, gk=k_start
                # l=1 -> j'=0, k'=1 -> gj=j_start, gk=k_start+1
                # l=2 -> j'=1, k'=0 -> gj=j_start+1, gk=k_start
                # l=3 -> j'=1, k'=1 -> gj=j_start+1, gk=k_start+1
                for l, val in V_nz[t]:
                    lj = l // 2
                    lk = l % 2
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk][col] += val
                    
                # W: map local C indices
                # l=0 -> i'=0, k'=0 -> gi=1, gk=k_start
                # l=1 -> i'=0, k'=1 -> gi=1, gk=k_start+1
                # l=2 -> i'=1, k'=0 -> gi=2, gk=k_start
                # l=3 -> i'=1, k'=1 -> gi=2, gk=k_start+1
                for l, val in W_nz[t]:
                    li = l // 2
                    lk = l % 2
                    gi = 1 + li
                    gk = k_start + lk
                    W[gk * n + gi][col] += val
                    
                col += 1
                
                # Check deadline periodically
                if (col % 4) == 0 and time.time() - start_time > 140:
                    return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}
