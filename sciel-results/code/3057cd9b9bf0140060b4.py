import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Bottom part: i=2
    # A[2, j] * B[j, k]
    for j in range(m):
        for k in range(p):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U[u_idx][r] = 1
            V[v_idx][r] = 1
            W[w_idx][r] = 1
            r += 1
            
    # Top part: i=0,1
    # j=4 (single column)
    for i in range(2):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx][r] = 1
            V[v_idx][r] = 1
            W[w_idx][r] = 1
            r += 1
            
    # j=0,1 and j=2,3 blocks
    # Strassen 2x2x2 terms (local)
    # m1: (A00+A11)(B00+B11) -> C00+, C11+
    # m2: (A10+A11)B00      -> C10+, C11+
    # m3: A00(B01-B11)       -> C01+, C11-
    # m4: A11(B10-B00)       -> C00-, C01+
    # m5: (A00+A01)B11       -> C00+, C11+
    # m6: (A10-A00)(B00+B01) -> C10+
    # m7: (A01-A11)(B10+B11) -> C00+, C01+
    
    strassen_terms = [
        # (u_local, v_local, w_local)
        # u: [A00, A01, A10, A11]
        # v: [B00, B01, B10, B11]
        # w: [C00, C01, C10, C11]
        # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # m1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        # m3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
        # m4
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 1, 0, 0]),
        # m5
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1]),
        # m6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
        # m7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0]),
    ]
    
    for block_j in [0, 2]:
        for block_k in [0, 2, 4]:
            for u_loc, v_loc, w_loc in strassen_terms:
                # Apply to A block at i=0,1 and j=block_j, block_j+1
                # Apply to B block at j=block_j, block_j+1 and k=block_k, block_k+1
                # Apply to C block at i=0,1 and k=block_k, block_k+1
                
                # U
                for li, val in enumerate(u_loc):
                    if val != 0:
                        i_loc, j_loc = divmod(li, 2)
                        g_i = i_loc
                        g_j = block_j + j_loc
                        u_idx = g_i * m + g_j
                        U[u_idx][r] += val
                        
                # V
                for lj, val in enumerate(v_loc):
                    if val != 0:
                        j_loc, k_loc = divmod(lj, 2)
                        g_j = block_j + j_loc
                        g_k = block_k + k_loc
                        v_idx = g_j * p + g_k
                        V[v_idx][r] += val
                        
                # W
                for lk, val in enumerate(w_loc):
                    if val != 0:
                        i_loc, k_loc = divmod(lk, 2)
                        g_i = i_loc
                        g_k = block_k + k_loc
                        w_idx = g_k * n + g_i
                        W[w_idx][r] += val
                        
                r += 1

    return {"U": U, "V": V, "W": W}
