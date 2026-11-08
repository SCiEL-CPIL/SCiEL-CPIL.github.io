import numpy as np
import time
from itertools import product

START_TIME = time.time()
TIME_LIMIT = 140  # seconds, safety margin from 150s

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the rank-50 baseline
    rank_baseline = 50
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        U_base[a_idx][col] += 1; U_base[d_idx][col] += 1
        V_base[e_idx][col] += 1; V_base[h_idx][col] += 1
        W_base[c00_idx][col] += 1; W_base[c11_idx][col] += 1
        col += 1
        
        U_base[c_idx][col] += 1; U_base[d_idx][col] += 1
        V_base[e_idx][col] += 1
        W_base[c10_idx][col] += 1; W_base[c11_idx][col] -= 1
        col += 1
        
        U_base[a_idx][col] += 1
        V_base[f_idx][col] += 1; V_base[h_idx][col] -= 1
        W_base[c01_idx][col] += 1; W_base[c11_idx][col] += 1
        col += 1
        
        U_base[d_idx][col] += 1
        V_base[g_idx][col] += 1; V_base[e_idx][col] -= 1
        W_base[c00_idx][col] += 1; W_base[c10_idx][col] += 1
        col += 1
        
        U_base[a_idx][col] += 1; U_base[b_idx][col] += 1
        V_base[h_idx][col] += 1
        W_base[c00_idx][col] -= 1; W_base[c01_idx][col] += 1
        col += 1
        
        U_base[c_idx][col] += 1; U_base[a_idx][col] -= 1
        V_base[e_idx][col] += 1; V_base[f_idx][col] += 1
        W_base[c11_idx][col] += 1
        col += 1
        
        U_base[b_idx][col] += 1; U_base[d_idx][col] -= 1
        V_base[g_idx][col] += 1; V_base[h_idx][col] += 1
        W_base[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U_base[a_idx][col] += 1; V_base[e_idx][col] += 1; W_base[c00_idx][col] += 1
        col += 1
        U_base[b_idx][col] += 1; V_base[g_idx][col] += 1; W_base[c00_idx][col] += 1
        col += 1
        U_base[c_idx][col] += 1; V_base[e_idx][col] += 1; W_base[c10_idx][col] += 1
        col += 1
        U_base[d_idx][col] += 1; V_base[g_idx][col] += 1; W_base[c10_idx][col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    assert col == rank_baseline
    
    # Check for linear dependencies to reduce rank
    U_np = np.array(U_base, dtype=float)
    V_np = np.array(V_base, dtype=float)
    W_np = np.array(W_base, dtype=float)
    
    # The tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We want to see if any column r is redundant.
    # A column r is redundant if T can be written without it.
    # This is equivalent to checking if the rank of the tensor is < 50.
    # We can try to remove columns one by one and check if the error increases.
    # However, exact integer solutions are hard to find via optimization.
    
    # Return the baseline
    return {"U": U_base, "V": V_base, "W": W_base}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
