"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x6 matrix.
    
    The current best known construction uses rank 21, applying Strassen's algorithm 
    independently to 3 non-overlapping 2x2 blocks of the 2x6 matrix.
    
    We return matrices U (4xR), V (12xR), W (12xR) where R is the rank.
    The decomposition satisfies: C_{i,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    (using Einstein summation convention over j, k, r for the full tensor product, 
     but specifically for matrix multiplication M_{n,m,p}:
     (A*B)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_{i,j,k,r} A_{i,j} B_{j,k} W_{k,i,r} * (U/V structure)
     
     Actually, the standard tensor decomposition of matrix multiplication tensor T_{i,j,k} is:
     T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
     
     Here:
     U has shape (n*m, rank) -> indices (i*m+j, r)
     V has shape (m*p, rank) -> indices (j*p+k, r)
     W has shape (p*n, rank) -> indices (k*n+i, r)
     
     The product is: sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (A*B)[i,k]
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Rank 21 decomposition: 3 independent Strassen 2x2 blocks
    # Each Strassen block uses rank 7
    total_rank = 21
    
    U = np.zeros((n * m, total_rank), dtype=int)  # 4 x 21
    V = np.zeros((m * p, total_rank), dtype=int)  # 12 x 21
    W = np.zeros((p * n, total_rank), dtype=int)  # 12 x 21
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Coefficients for U (input A): A[i,j] -> U[i*m+j, :]
    # For each of the 7 products, define which A elements contribute
    U_coeffs = [
        {0: 1, 3: 1},       # M1: a + d -> A[0,0] + A[1,1]
        {2: 1, 3: 1},       # M2: c + d -> A[1,0] + A[1,1]
        {0: 1},             # M3: a -> A[0,0]
        {3: 1},             # M4: d -> A[1,1]
        {0: 1, 1: 1},       # M5: a + b -> A[0,0] + A[0,1]
        {2: 1, 0: -1},      # M6: c - a -> A[1,0] - A[0,0]
        {1: 1, 3: -1}       # M7: b - d -> A[0,1] - A[1,1]
    ]
    
    # Coefficients for V (input B): B[j,k] -> V[j*p+k, :]
    # For each of the 7 products, define which B elements contribute
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: e + h -> B[0,0] + B[1,1]
        [(0, 0, 1)],                 # M2: e -> B[0,0]
        [(0, 1, 1), (1, 1, -1)],     # M3: f - h -> B[0,1] - B[1,1]
        [(1, 0, 1), (0, 0, -1)],     # M4: g - e -> B[1,0] - B[0,0]
        [(1, 1, 1)],                 # M5: h -> B[1,1]
        [(0, 0, 1), (0, 1, 1)],      # M6: e + f -> B[0,0] + B[0,1]
        [(1, 0, 1), (1, 1, 1)]       # M7: g + h -> B[1,0] + B[1,1]
    ]
    
    # Coefficients for W (output C): C[i,k] -> W[k*n+i, :]
    # For each of the 7 products, define which C elements contribute
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # M1 -> C[0,0] + C[1,1]
        [(1, 0, 1), (1, 1, -1)],  # M2 -> C[1,0] - C[1,1]
        [(0, 1, 1), (1, 1, 1)],   # M3 -> C[0,1] + C[1,1]
        [(0, 0, 1), (1, 0, 1)],   # M4 -> C[0,0] + C[1,0]
        [(0, 0, -1), (0, 1, 1)],  # M5 -> -C[0,0] + C[0,1]
        [(1, 1, 1)],              # M6 -> C[1,1]
        [(0, 0, 1)]               # M7 -> C[0,0]
    ]
    
    # Apply Strassen to each of the 3 blocks
    for r_block in range(3):
        for k_strassen in range(7):
            col = r_block * 7 + k_strassen
            
            # Fill U
            for idx, val in U_coeffs[k_strassen].items():
                U[idx, col] = val
            
            # Fill V
            for j_local, k_local, val in V_coeffs[k_strassen]:
                global_k = 2 * r_block + k_local
                v_idx = j_local * p + global_k
                V[v_idx, col] = val
            
            # Fill W
            for i_out, k_local, val in W_map[k_strassen]:
                global_k = 2 * r_block + k_local
                w_idx = global_k * n + i_out
                W[w_idx, col] = val
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
