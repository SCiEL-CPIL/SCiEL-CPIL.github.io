import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # 1. Define 2x2x2 Strassen decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U indices: (i,j) -> 0:a, 1:b, 2:c, 3:d
    # V indices: (j,k) -> 0:e, 1:f, 2:g, 3:h
    # W indices: (k,i) -> 0:C00, 1:C01, 2:C10, 3:C11
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    # C00 += M1, C11 += M1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # M2 = (c+d)e
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    # C10 += M2, C11 -= M2
    W_strassen[2, 1] = 1; W_strassen[3, 1] = -1
    
    # M3 = a(f-h)
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[3, 2] = -1
    # C01 += M3, C11 += M3
    W_strassen[1, 2] = 1; W_strassen[3, 2] = 1
    
    # M4 = d(g-f)
    U_strassen[3, 3] = 1
    V_strassen[2, 3] = 1; V_strassen[1, 3] = -1
    # C00 += M4, C11 += M4
    W_strassen[0, 3] = 1; W_strassen[3, 3] = 1
    
    # M5 = (a+b)h
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    # C00 -= M5, C10 += M5
    W_strassen[0, 4] = -1; W_strassen[2, 4] = 1
    
    # M6 = (c-a)(e+h)
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[3, 5] = 1
    # C11 += M6
    W_strassen[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    # C00 += M7
    W_strassen[0, 6] = 1
    
    # 2. Build 4x4x4 decomposition using recursive Strassen
    # U_4x4x4 shape (16, 49)
    # V_4x4x4 shape (16, 49)
    # W_4x4x4 shape (16, 49)
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Map block indices (I, J) to global indices (i, j)
    # i = 2*I + i_local, j = 2*J + j_local
    # k = 2*K + k_local
    
    for m_outer in range(7):
        for m_inner in range(7):
            col = m_outer * 7 + m_inner
            
            # Extract coefficients from U_strassen[m_outer, :] for A blocks
            # U_strassen rows are 0:a, 1:b, 2:c, 3:d
            # These correspond to A00, A01, A10, A11
            
            # For each non-zero coeff in U_strassen[:, m_outer], add to U_444
            # U_strassen[i_block, m_outer] is coeff for A_i_block
            # A_i_block is a 2x2 matrix.
            # We need to multiply this coeff with the U_strassen of the inner step.
            
            # Inner step U_strassen[i_local, m_inner] is coeff for local element.
            
            # So for global i, j:
            # i = 2*I + i_local
            # j = 2*J + j_local
            # Coeff = U_strassen[I*2+J, m_outer] * U_strassen[i_local, m_inner]
            # Wait, U_strassen indices for 2x2 are (i,j) where i is row, j is col.
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            
            for i_block in range(4): # 0:A00, 1:A01, 2:A10, 3:A11
                coeff_outer = U_strassen[i_block, m_outer]
                if coeff_outer == 0:
                    continue
                
                I = i_block // 2
                J = i_block % 2
                
                for i_local in range(2):
                    for j_local in range(2):
                        coeff_inner = U_strassen[i_local * 2 + j_local, m_inner]
                        if coeff_inner == 0:
                            continue
                        
                        i_global = 2 * I + i_local
                        j_global = 2 * J + j_local
                        U_444[i_global * 4 + j_global, col] += coeff_outer * coeff_inner
            
            # Similarly for V
            # V_strassen indices (j,k) -> 0:e, 1:f, 2:g, 3:h
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            
            for j_block in range(4):
                coeff_outer = V_strassen[j_block, m_outer]
                if coeff_outer == 0:
                    continue
                
                J = j_block // 2
                K = j_block % 2
                
                for j_local in range(2):
                    for k_local in range(2):
                        coeff_inner = V_strassen[j_local * 2 + k_local, m_inner]
                        if coeff_inner == 0:
                            continue
                        
                        j_global = 2 * J + j_local
                        k_global = 2 * K + k_local
                        V_444[j_global * 4 + k_global, col] += coeff_outer * coeff_inner
            
            # Similarly for W
            # W_strassen indices (k,i) -> 0:C00, 1:C01, 2:C10, 3:C11
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            
            for k_block in range(4):
                coeff_outer = W_strassen[k_block, m_outer]
                if coeff_outer == 0:
                    continue
                
                K = k_block // 2
                I = k_block % 2
                
                for k_local in range(2):
                    for i_local in range(2):
                        coeff_inner = W_strassen[k_local * 2 + i_local, m_inner]
                        if coeff_inner == 0:
                            continue
                        
                        k_global = 2 * K + k_local
                        i_global = 2 * I + i_local
                        W_444[k_global * 4 + i_global, col] += coeff_outer * coeff_inner

    # 3. Build 4x4x1 decomposition (schoolbook)
    # U_441 shape (16, 16)
    # V_441 shape (4, 16)
    # W_441 shape (4, 16)
    
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((4, 16))
    W_441 = np.zeros((4, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i * 4 + j, col] = 1
            V_441[j, col] = 1
            W_441[i, col] = 1 # k=0, so index is 0*4 + i = i
            col += 1

    # 4. Combine for 4x4x9
    # B is 4x9.
    # Split into B1 (4x4), B2 (4x4), B3 (4x1).
    # C = A B1 + A B2 + A B3.
    
    # U is same for all parts (acts on A).
    # U_total shape (16, 49+49+16) = (16, 114)
    U_total = np.zeros((16, 114))
    V_total = np.zeros((16, 114)) # m*p = 4*9 = 36? No, V is (m, p) = (4, 9).
    # Wait, V shape is (m, p) = (4, 9).
    # In 4x4x4, V is (4,4).
    # In 4x4x1, V is (4,1).
    
    # V_total shape (36, 114)?
    # m=4, p=9. V is (4, 9).
    # V_444 is (16, 49) which is (4, 4).
    # V_441 is (4, 16) which is (4, 1).
    
    # We need to place them in the correct columns of V_total.
    # V_total rows: 0-3 for B1, 4-7 for B2, 8 for B3.
    # Wait, B is 4x9.
    # B1 is cols 0-3.
    # B2 is cols 4-7.
    # B3 is col 8.
    
    # V_444 has 4 columns (k=0,1,2,3).
    # We need to map k to the global column index.
    
    # For B1 (cols 0-3):
    # V_444[j, k] -> V_total[j, k]
    # For B2 (cols 4-7):
    # V_444[j, k] -> V_total[j, k+4]
    # For B3 (col 8):
    # V_441[j, 0] -> V_total[j, 8]
    
    # W_total shape (36, 114)?
    # p*n = 9*4 = 36.
    # W_444 is (16, 49) which is (4, 4).
    # W_441 is (4, 16) which is (1, 4).
    
    # W_444[k, i] -> W_total[k*4 + i, ...]
    # For B1: k=0..3.
    # For B2: k=4..7.
    # For B3: k=8.
    
    U_total[:, 0:49] = U_444
    U_total[:, 49:98] = U_444
    U_total[:, 98:114] = U_441
    
    # V mapping
    # V_444 is (4, 4).
    for j in range(4):
        for k in range(4):
            V_total[j, k] = V_444[j, k] # B1
            V_total[j, k + 4] = V_444[j, k] # B2
    
    # V_441 is (4, 1).
    for j in range(4):
        V_total[j, 8] = V_441[j, 0] # B3
    
    # W mapping
    # W_444 is (4, 4). Rows are k*4 + i.
    for k in range(4):
        for i in range(4):
            W_total[k * 4 + i, 0:49] = W_444[k * 4 + i, 0:49] # B1
            W_total[(k + 4) * 4 + i, 49:98] = W_444[k * 4 + i, 0:49] # B2
    
    # W_441 is (1, 4). Rows are k*4 + i. k=0.
    for i in range(4):
        W_total[8 * 4 + i, 98:114] = W_441[0, 0:16] # B3
    
    # Return
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

