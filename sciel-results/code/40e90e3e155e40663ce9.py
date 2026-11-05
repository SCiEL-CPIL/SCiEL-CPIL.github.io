import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block sizes
    I = [2, 2, 1]
    J = [2, 2, 1]
    K = [2, 2, 1]
    
    # Calculate offsets
    I_off = [0] * 3
    J_off = [0] * 3
    K_off = [0] * 3
    for i in range(1, 3):
        I_off[i] = I_off[i-1] + I[i-1]
    for j in range(1, 3):
        J_off[j] = J_off[j-1] + J[j-1]
    for k in range(1, 3):
        K_off[k] = K_off[k-1] + K[k-1]
        
    # Precompute Strassen local decomposition for 2x2x2
    # U_strassen: 4x7
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    # V_strassen: 4x7
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    # W_strassen: 4x7 (indexed by k, i)
    # C00 (k=0, i=0): 1, 0, 0, 1, -1, 0, 1
    # C10 (k=0, i=1): 0, 1, 0, 1, 0, 0, 0
    # C01 (k=1, i=0): 0, 0, 1, 0, 1, 0, 0
    # C11 (k=1, i=1): 1, -1, 1, 0, 0, 1, 0
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Total rank calculation
    total_rank = 0
    block_ranks = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = I[i] * J[j] * K[k]
                total_rank += block_ranks[(i, j, k)]
                
    # Initialize global matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                rank = block_ranks[(i, j, k)]
                
                if i < 2 and j < 2 and k < 2:
                    # Use Strassen
                    # Local U: 4x7
                    # Local V: 4x7
                    # Local W: 4x7
                    for r in range(rank):
                        # U
                        for li in range(2):
                            for lj in range(2):
                                val = U_strassen[li * 2 + lj][r]
                                if val != 0:
                                    gi = I_off[i] + li
                                    gj = J_off[j] + lj
                                    U[gi * m + gj][current_rank + r] = val
                        # V
                        for lj in range(2):
                            for lk in range(2):
                                val = V_strassen[lj * 2 + lk][r]
                                if val != 0:
                                    gj = J_off[j] + lj
                                    gk = K_off[k] + lk
                                    V[gj * p + gk][current_rank + r] = val
                        # W
                        for lk in range(2):
                            for li in range(2):
                                val = W_strassen[lk * 2 + li][r]
                                if val != 0:
                                    gk = K_off[k] + lk
                                    gi = I_off[i] + li
                                    W[gk * n + gi][current_rank + r] = val
                else:
                    # Use Schoolbook
                    for li in range(I[i]):
                        for lj in range(J[j]):
                            for lk in range(K[k]):
                                # U
                                gi = I_off[i] + li
                                gj = J_off[j] + lj
                                U[gi * m + gj][current_rank] = 1
                                # V
                                V[gj * p + (K_off[k] + lk)][current_rank] = 1
                                # W
                                W[(K_off[k] + lk) * n + gi][current_rank] = 1
                                current_rank += 1
                                continue_outer = True
                                # We need to increment current_rank for each schoolbook term
                                # But the loop structure above adds 1 to current_rank per term
                                # So we need to be careful.
                                # Actually, the loop adds 1 to current_rank for each (li, lj, lk).
                                # This is correct for schoolbook.
                                # But for Strassen, we added 'rank' columns at once.
                                # So for Strassen, we should increment current_rank by rank after the inner loops.
                                # Let's refactor the increment logic.
                current_rank += 0 # Placeholder, logic below
                
                # Correcting increment logic:
                # For Strassen, we used columns current_rank to current_rank+6.
                # So we must increment current_rank by 7.
                # For Schoolbook, we incremented current_rank by 1 for each term.
                # This is inconsistent. Let's fix.
                
    # Re-implementing with correct rank tracking
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                rank = block_ranks[(i, j, k)]
                
                if i < 2 and j < 2 and k < 2:
                    # Strassen
                    for r in range(rank):
                        # U
                        for li in range(2):
                            for lj in range(2):
                                val = U_strassen[li * 2 + lj][r]
                                if val != 0:
                                    gi = I_off[i] + li
                                    gj = J_off[j] + lj
                                    U[gi * m + gj][current_rank + r] = val
                        # V
                        for lj in range(2):
                            for lk in range(2):
                                val = V_strassen[lj * 2 + lk][r]
                                if val != 0:
                                    gj = J_off[j] + lj
                                    gk = K_off[k] + lk
                                    V[gj * p + gk][current_rank + r] = val
                        # W
                        for lk in range(2):
                            for li in range(2):
                                val = W_strassen[lk * 2 + li][r]
                                if val != 0:
                                    gk = K_off[k] + lk
                                    gi = I_off[i] + li
                                    W[gk * n + gi][current_rank + r] = val
                    current_rank += rank
                else:
                    # Schoolbook
                    for li in range(I[i]):
                        for lj in range(J[j]):
                            for lk in range(K[k]):
                                gi = I_off[i] + li
                                gj = J_off[j] + lj
                                gk = K_off[k] + lk
                                
                                U[gi * m + gj][current_rank] = 1
                                V[gj * p + gk][current_rank] = 1
                                W[gk * n + gi][current_rank] = 1
                                
                                current_rank += 1
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
