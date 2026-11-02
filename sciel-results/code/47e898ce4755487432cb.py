import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # Strassen matrices for M_2,2,2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U_str = np.zeros((4, 7), dtype=int)
    V_str = np.zeros((4, 7), dtype=int)
    W_str = np.zeros((4, 7), dtype=int)
    
    # M1: (A00 + A11) * (B00 + B11) -> C00, C11
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[3, 0] = 1
    
    # M2: (A10 + A11) * B00 -> C10
    U_str[2, 1] = 1; U_str[3, 1] = 1
    V_str[0, 1] = 1
    W_str[2, 1] = 1 # C10: i=1, k=0 -> idx 1*2+0 = 2
    
    # M3: A00 * (B01 - B11) -> C01
    U_str[0, 2] = 1
    V_str[1, 2] = 1; V_str[3, 2] = -1
    W_str[1, 2] = 1 # C01: i=0, k=1 -> idx 1*2+0 = 2? No.
    # W index is k*n + i.
    # C01: i=0, k=1. idx = 1*2 + 0 = 2.
    # Wait, C[i,k].
    # W[k*n + i, c].
    # If i=0, k=1, idx = 1*2 + 0 = 2.
    # If i=1, k=0, idx = 0*2 + 1 = 1.
    
    # Let's re-verify W indices.
    # C00: i=0, k=0 -> idx 0
    # C01: i=0, k=1 -> idx 2
    # C10: i=1, k=0 -> idx 1
    # C11: i=1, k=1 -> idx 3
    
    # M3 contributes to C01.
    W_str[2, 2] = 1
    
    # M4: A11 * (B10 - B00) -> C10
    U_str[3, 3] = 1
    V_str[2, 3] = 1; V_str[0, 3] = -1
    W_str[1, 3] = 1
    
    # M5: (A00 + A01) * B11 -> C01, C11
    U_str[0, 4] = 1; U_str[1, 4] = 1
    V_str[3, 4] = 1
    W_str[2, 4] = 1; W_str[3, 4] = 1
    
    # M6: (A10 - A00) * (B00 + B01) -> C11
    U_str[2, 5] = 1; U_str[0, 5] = -1
    V_str[0, 5] = 1; V_str[1, 5] = 1
    W_str[3, 5] = 1
    
    # M7: (A01 - A11) * (B10 + B11) -> C00, C11
    U_str[1, 6] = 1; U_str[3, 6] = -1
    V_str[2, 6] = 1; V_str[3, 6] = 1
    W_str[0, 6] = 1; W_str[3, 6] = 1

    # Check Strassen correctness
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now construct global U, V, W
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for jb in range(2):
        for kb in range(3):
            for c in range(7):
                # U
                for i in range(2):
                    for jl in range(2):
                        u_val = U_str[i * 2 + jl, c]
                        if u_val != 0:
                            j_global = jb * 2 + jl
                            u_idx = i * m + j_global
                            U[u_idx][col] += u_val
                            
                # V
                for jl in range(2):
                    for kl in range(2):
                        v_val = V_str[jl * 2 + kl, c]
                        if v_val != 0:
                            j_global = jb * 2 + jl
                            k_global = kb * 2 + kl
                            v_idx = j_global * p + k_global
                            V[v_idx][col] += v_val
                            
                # W
                for kl in range(2):
                    for i in range(2):
                        w_val = W_str[kl * 2 + i, c]
                        if w_val != 0:
                            k_global = kb * 2 + kl
                            w_idx = k_global * n + i
                            W[w_idx][col] += w_val
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
