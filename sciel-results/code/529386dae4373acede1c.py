"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 56 using Strassen on 8 blocks of 2x2x2
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Local indices i', j', k' in {0, 1}
    # r in 0..6
    # U_local[r][i'][j']
    # V_local[r][j'][k']
    # W_local[r][k'][i']
    
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    # M2: (A10+A11)B00 -> C10+1, C11-1
    # M3: A00(B01-B11) -> C01+1, C11+1
    # M4: A11(B10-B00) -> C00+1, C10+1
    # M5: (A00+A10)B11 -> C00-1, C01+1
    # M6: (A01-A00)(B00+B10) -> C11+1
    # M7: (A10-A11)(B01+B11) -> C00+1
    
    U_loc = [
        [[1, 0], [0, 1]], # M1: A00, A11
        [[0, 0], [1, 1]], # M2: A10, A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 0], [1, 0]], # M5: A00, A10
        [[-1, 1], [0, 0]],# M6: -A00, A01
        [[0, 0], [1, -1]] # M7: A10, -A11
    ]
    
    V_loc = [
        [[1, 0], [0, 1]], # M1: B00, B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [-1, 0]],# M3: B01, -B11 (Wait: B01 is j=0,k=1; B11 is j=1,k=1)
        # Correction on V indices: V[j][k]
        # M3: A00 (B01 - B11). B01 is j=0, k=1. B11 is j=1, k=1.
        # So V[0][1] = 1, V[1][1] = -1.
        # My V_loc[2] above was [[0,1],[-1,0]] which is V[0][1]=1, V[1][0]=-1. Incorrect.
        # Let's re-list V_loc carefully.
        
        # M1: B00+B11 -> V[0][0]=1, V[1][1]=1.
        # M2: B00 -> V[0][0]=1.
        # M3: B01-B11 -> V[0][1]=1, V[1][1]=-1.
        # M4: B10-B00 -> V[1][0]=1, V[0][0]=-1.
        # M5: B11 -> V[1][1]=1.
        # M6: B00+B10 -> V[0][0]=1, V[1][0]=1.
        # M7: B01+B11 -> V[0][1]=1, V[1][1]=1.
        
        # Re-defining V_loc
        # [[V00, V01], [V10, V11]]
        
        # M1
        # [[1, 0], [0, 1]]
        # M2
        # [[1, 0], [0, 0]]
        # M3
        # [[0, 1], [0, -1]]
        # M4
        # [[-1, 0], [1, 0]]
        # M5
        # [[0, 0], [0, 1]]
        # M6
        # [[1, 0], [1, 0]]
        # M7
        # [[0, 1], [0, 1]]
    ]
    
    V_loc = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [1, 0]],
        [[0, 1], [0, 1]]
    ]
    
    W_loc = [
        # W[k][i]
        # M1: C00+1, C11+1 -> W[0][0]=1, W[1][1]=1
        [[1, 0], [0, 1]],
        # M2: C10+1, C11-1 -> W[0][1]=1, W[1][1]=-1
        [[0, 1], [0, -1]],
        # M3: C01+1, C11+1 -> W[1][0]=1, W[1][1]=1
        [[0, 0], [1, 1]],
        # M4: C00+1, C10+1 -> W[0][0]=1, W[0][1]=1
        [[1, 1], [0, 0]],
        # M5: C00-1, C01+1 -> W[0][0]=-1, W[1][0]=1
        [[-1, 0], [1, 0]],
        # M6: C11+1 -> W[1][1]=1
        [[0, 0], [0, 1]],
        # M7: C00+1 -> W[0][0]=1
        [[1, 0], [0, 0]]
    ]
    
    # Fill global matrices
    for alpha in range(2): # J blocks
        for beta in range(4): # K blocks
            base_r = (alpha * 4 + beta) * 7
            for r in range(7):
                global_r = base_r + r
                for i_prime in range(2):
                    for j_prime in range(2):
                        i = i_prime
                        j = 2 * alpha + j_prime
                        u_idx = i * m + j
                        val_u = U_loc[r][i_prime][j_prime]
                        if val_u != 0:
                            U[u_idx][global_r] += val_u
                            
                        for k_prime in range(2):
                            k = 2 * beta + k_prime
                            v_idx = j * p + k
                            val_v = V_loc[r][j_prime][k_prime]
                            if val_v != 0:
                                V[v_idx][global_r] += val_v
                                
                            w_idx = k * n + i
                            val_w = W_loc[r][k_prime][i_prime]
                            if val_w != 0:
                                W[w_idx][global_r] += val_w
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
