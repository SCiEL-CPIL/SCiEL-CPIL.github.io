import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Laderman's algorithm for 4x4x4
    # U, V, W are 16x21
    # I will define the coefficients for the 21 terms.
    
    # The 21 multiplications in Laderman's algorithm
    # Reference: C. Laderman, "A non-recursive matrix multiplication algorithm", J. Algorithms 1976.
    # Or standard implementations.
    
    # Indices for 4x4 matrix flattened: i*4 + j
    # A is 4x4, B is 4x4, C is 4x4.
    
    U_lad = np.zeros((16, 21))
    V_lad = np.zeros((16, 21))
    W_lad = np.zeros((16, 21))
    
    # Helper to set entries
    def set_entry(mat, row, col, val):
        mat[row][col] += val

    # M1
    set_entry(U_lad, 0, 0, 1); set_entry(U_lad, 5, 0, 1) # A00, A11
    set_entry(V_lad, 0, 0, 1); set_entry(V_lad, 5, 0, 1) # B00, B11
    set_entry(W_lad, 0, 0, 1) # C00
    
    # M2
    set_entry(U_lad, 0, 1, 1); set_entry(U_lad, 5, 1, 1)
    set_entry(V_lad, 1, 1, 1); set_entry(V_lad, 4, 1, 1) # B01, B10
    set_entry(W_lad, 1, 1, 1); set_entry(W_lad, 4, 1, 1) # C01, C10
    
    # M3
    set_entry(U_lad, 0, 2, 1); set_entry(U_lad, 5, 2, -1)
    set_entry(V_lad, 5, 2, 1) # B11
    set_entry(W_lad, 1, 2, 1); set_entry(W_lad, 4, 2, -1)
    
    # M4
    set_entry(U_lad, 0, 3, 1) # A00
    set_entry(V_lad, 4, 3, 1); set_entry(V_lad, 5, 3, -1) # B10, B11
    set_entry(W_lad, 4, 3, 1); set_entry(W_lad, 5, 3, 1) # C10, C11
    
    # M5
    set_entry(U_lad, 5, 4, 1) # A11
    set_entry(V_lad, 0, 4, 1); set_entry(V_lad, 4, 4, -1) # B00, B10
    set_entry(W_lad, 0, 4, 1); set_entry(W_lad, 4, 4, 1) # C00, C10
    
    # M6
    set_entry(U_lad, 1, 5, 1); set_entry(U_lad, 2, 5, 1) # A01, A02
    set_entry(V_lad, 0, 5, 1); set_entry(V_lad, 1, 5, 1) # B00, B01
    set_entry(W_lad, 0, 5, 1) # C00
    # ... This is getting tedious and error-prone without the exact table.
    
    # Instead, I will use a known correct set of coefficients or generate them?
    # I cannot generate them at runtime efficiently for 120s limit if it's complex.
    # I will use the Strassen-based rank 56 solution but try to optimize the SVD reduction?
    # No, the SVD reduction in the current code is flawed (it just takes first r columns).
    # The current code returns rank 56 effectively because the verification fails for reduced rank or it just returns the 56 rank one.
    # Actually, if I return rank 42, I need the correct factors.
    
    # Let's use the property that T_2,4,8 is a sub-tensor of T_4,4,8?
    # No, T_4,4,8 rank is 49+49=98.
    
    # I will stick to the Strassen 56 solution but fix the reduction logic?
    # No, 56 is the rank.
    
    # I will try to implement the rank 48 solution if I can find it.
    # Or just return the Strassen 56 solution cleanly.
    
    # Wait, the current score is 1.1429 (rank 56).
    # If I can get rank 48, score is 1.33.
    # If I can get rank 42, score is 1.52.
    
    # I will use the Laderman factors from a reliable source (hardcoded).
    
    # Laderman 21 factors for 4x4x4:
    # I will use a simplified version: Strassen for 4x4x4 is 49.
    # Laderman is 21.
    
    # I will provide the Laderman factors.
    
    U_lad = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # ...
    ]
    
    # This is too risky to hardcode from memory.
    
    # I will return the Strassen 56 solution.
    
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
