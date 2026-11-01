"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # Local 2x2 Strassen tables
    # U_loc[2*i + j, s]
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (A00)
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0 (A10)
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A11)
    ]
    
    # V_loc[2*j + k, s]
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0 (B00)
        [0, 0, 1, 0, 0, 1, 0],  # j=0, k=1 (B01)
        [0, 0, 0, 1, 0, 0, 1],  # j=1, k=0 (B10)
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1 (B11)
    ]
    
    # W_loc[2*k + i, s]
    W_loc = [
        [1, 0, 0, 0, -1, 0, 1], # k=0, i=0 (C00)
        [0, 0, 1, 0, 1, 0, 0],  # k=0, i=1 (C01)
        [0, 1, 0, 1, 0, 0, 0],  # k=1, i=0 (C10)
        [1, 0, 0, 0, 0, 1, 1]   # k=1, i=1 (C11)
    ]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Generate 49 terms
    for s1 in range(7):
        for s2 in range(7):
            r = 7 * s1 + s2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for i1 in range(2):
                            for j1 in range(2):
                                for k1 in range(2):
                                    # Local indices
                                    ui = 2 * i + j
                                    uj = 2 * i1 + j1
                                    vi = 2 * j + k
                                    vj = 2 * j1 + k1
                                    wi = 2 * k + i
                                    wj = 2 * k1 + i1
                                    
                                    u_val = U_loc[ui][s1] * U_loc[uj][s2]
                                    v_val = V_loc[vi][s1] * V_loc[vj][s2]
                                    w_val = W_loc[wi][s1] * W_loc[wj][s2]
                                    
                                    if u_val != 0 and v_val != 0 and w_val != 0:
                                        # Global indices
                                        I = 2 * i + i1
                                        J = 2 * j + j1
                                        K = 2 * k + k1
                                        
                                        u_idx = 4 * I + J
                                        v_idx = 4 * J + K
                                        w_idx = 4 * K + I
                                        
                                        U[u_idx][r] += u_val
                                        V[v_idx][r] += v_val
                                        W[w_idx][r] += w_val
                                        
                                    if time.time() > deadline:
                                        # Return partial result if timeout is imminent, though this should be fast
                                        return {"U": U, "V": V, "W": W}
                                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
