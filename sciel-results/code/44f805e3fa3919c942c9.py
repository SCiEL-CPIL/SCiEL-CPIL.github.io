"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Use 2x2 block decomposition with Strassen (7-term) for each 2x2x2 sub-problem
    # A (4x4) -> 2x2 blocks A[i_b, j_b] for i_b, j_b in {0,1}
    # B (4x8) -> 2x2 blocks B[j_b, k_b] for j_b in {0,1}, k_b in {0,1,2,3}
    # C (4x8) -> 2x2 blocks C[i_b, k_b] for i_b in {0,1}, k_b in {0,1,2,3}
    #
    # C[i_b, k_b] = A[i_b, 0] * B[0, k_b] + A[i_b, 1] * B[1, k_b]
    # Each 2x2 * 2x2 product uses 7 Strassen terms
    # Total: 2 (i_b) * 2 (j_b) * 4 (k_b) * 7 = 112
    
    rank = 112
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 32 x 112
    W = [[0] * rank for _ in range(p * n)]  # 32 x 112
    
    col = 0
    
    # Strassen 7-term decomposition for 2x2 * 2x2
    # Local indices: A[il,jl], B[jl,kl], C[il,kl] for il,jl,kl in {0,1}
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00 -> C10 += M2
    # M3 = A00*(B01-B11) -> C01 += M3
    # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11 -> C01 += M5, C11 -= M5
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7, C11 -= M7
    
    # Define Strassen terms as (u_coeffs, v_coeffs, w_coeffs)
    # Each is a list of (local_idx, coeff) for the respective matrix
    strassen_terms = [
        # M1: (A00+A11) * (B00+B11) -> C00 +1, C11 +1
        ([(0,1),(5,1)], [(0,1),(9,1)], [(0,1),(5,1)]),
        # M2: (A10+A11) * B00 -> C10 +1
        ([(4,1),(5,1)], [(0,1)], [(4,1)]),
        # M3: A00 * (B01-B11) -> C01 +1
        ([(0,1)], [(1,1),(9,-1)], [(1,1)]),
        # M4: A11 * (B10-B00) -> C00 +1, C10 +1
        ([(5,1)], [(8,1),(0,-1)], [(0,1),(4,1)]),
        # M5: (A00+A01) * B11 -> C01 +1, C11 -1
        ([(0,1),(1,1)], [(9,1)], [(1,1),(5,-1)]),
        # M6: (A10-A00) * (B00+B01) -> C11 +1
        ([(4,1),(0,-1)], [(0,1),(1,1)], [(5,1)]),
        # M7: (A01-A11) * (B10+B11) -> C00 +1, C11 -1
        ([(1,1),(5,-1)], [(8,1),(9,1)], [(0,1),(5,-1)]),
    ]
    
    # Map local 2x2 indices to global indices
    # Local A[il,jl] in block (i_b,j_b) -> global A[(i_b*2+il), (j_b*2+jl)] -> idx (i_b*2+il)*4 + (j_b*2+jl)
    # Local B[jl,kl] in block (j_b,k_b) -> global B[(j_b*2+jl), (k_b*2+kl)] -> idx (j_b*2+jl)*8 + (k_b*2+kl)
    # Local C[il,kl] in block (i_b,k_b) -> global C[(i_b*2+il), (k_b*2+kl)] -> idx (k_b*2+kl)*4 + (i_b*2+il)
    
    for i_b in range(2):
        for j_b in range(2):
            for k_b in range(4):
                for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                    for (local_idx, coeff) in u_coeffs:
                        il, jl = local_idx // 2, local_idx % 2  # Wait, need to map properly
                        # Actually, local A indices: 0=(0,0), 1=(0,1), 4=(1,0), 5=(1,1)
                        # So local_idx 0 -> il=0,jl=0; 1 -> il=0,jl=1; 4 -> il=1,jl=0; 5 -> il=1,jl=1
                        # This mapping is: il = local_idx // 2, jl = local_idx % 2 for the 2x2 layout
                        # But 0,1,4,5 don't follow il*2+jl pattern... 
                        # Let me use: local A index = il*2 + jl for il,jl in {0,1}
                        # So 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                        # But I used 0,1,4,5 above which is wrong!
                        pass
    
    # Let me redo with correct local indexing: A[il,jl] has local index il*2+jl
    # So: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_terms = [
        # M1: (A00+A11) * (B00+B11) -> C00 +1, C11 +1
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2: (A10+A11) * B00 -> C10 +1
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        # M3: A00 * (B01-B11) -> C01 +1
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        # M4: A11 * (B10-B00) -> C00 +1, C10 +1
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # M5: (A00+A01) * B11 -> C01 +1, C11 -1
        ([(0,1),(1,1)], [(3,1)], [(1,1),(3,-1)]),
        # M6: (A10-A00) * (B00+B01) -> C11 +1
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7: (A01-A11) * (B10+B11) -> C00 +1, C11 -1
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,-1)]),
    ]
    
    col = 0
    for i_b in range(2):
        for j_b in range(2):
            for k_b in range(4):
                for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                    for (local_idx, coeff) in u_coeffs:
                        il, jl = local_idx // 2, local_idx % 2
                        u_idx = (i_b * 2 + il) * m + (j_b * 2 + jl)
                        U[u_idx][col] += coeff
                    
                    for (local_idx, coeff) in v_coeffs:
                        jl, kl = local_idx // 2, local_idx % 2
                        v_idx = (j_b * 2 + jl) * p + (k_b * 2 + kl)
                        V[v_idx][col] += coeff
                    
                    for (local_idx, coeff) in w_coeffs:
                        il, kl = local_idx // 2, local_idx % 2
                        w_idx = (k_b * 2 + kl) * n + (i_b * 2 + il)
                        W[w_idx][col] += coeff
                    
                    col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-9:
        # Fallback to full schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
