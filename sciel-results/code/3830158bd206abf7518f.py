import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    # 7 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen definitions for 2x2 * 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A * B
    
    # U vectors (coeffs for A00, A01, A10, A11)
    # V vectors (coeffs for B00, B10, B01, B11) -> Note order in V matrix: j*p+k.
    # V index: j in {0,1}, k in block.
    # For block b, k0=2b, k1=2b+1.
    # V rows:
    # 0: j=0, k=k0 -> B00
    # 1: j=1, k=k0 -> B10
    # 2: j=0, k=k1 -> B01
    # 3: j=1, k=k1 -> B11
    # W vectors (coeffs for C00, C01, C10, C11)
    # W index: k*n+i.
    # For block b, k0=2b, k1=2b+1.
    # W rows:
    # 0: k=k0, i=0 -> C00
    # 1: k=k0, i=1 -> C10
    # 2: k=k1, i=0 -> C01
    # 3: k=k1, i=1 -> C11
    
    # Strassen Multiplications
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U coeffs: [A00, A01, A10, A11]
    U1 = [1, 0, 0, 1]
    U2 = [0, 0, 1, 1]
    U3 = [1, 0, 0, 0]
    U4 = [0, 0, 0, 1]
    U5 = [1, 1, 0, 0]
    U6 = [1, 0, -1, 0]
    U7 = [0, 1, 0, -1]
    
    # V coeffs: [B00, B10, B01, B11]
    V1 = [1, 1, 0, 0]
    V2 = [1, 0, 0, 0]
    V3 = [0, -1, 1, 0] # B01 - B11 -> 0*B00 -1*B10 +1*B01 +0*B11
    V4 = [1, -1, 0, 0] # B10 - B00 -> -1*B00 +1*B10
    V5 = [0, 0, 0, 1]
    V6 = [1, 0, 1, 0]
    V7 = [0, 1, 0, 1]
    
    # W coeffs: [C00, C10, C01, C11]
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    W1 = [1, 0, 0, 1]
    W2 = [0, 1, 0, -1]
    W3 = [0, 0, 1, 1]
    W4 = [1, 1, 0, 0]
    W5 = [-1, 0, 1, 0]
    W6 = [0, 0, 0, 1]
    W7 = [1, 0, 0, 1] # Wait, M7 contributes to C00 and C11?
    # Let's re-verify Strassen reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M_r, we add its contribution to W.
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 += -1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += 1, C10 += 1
    # M5: C00 += -1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1
    
    # Let's define W_r vectors [C00, C10, C01, C11]
    W_r1 = [1, 0, 0, 1]
    W_r2 = [0, 1, 0, -1]
    W_r3 = [0, 0, 1, 1]
    W_r4 = [1, 1, 0, 0]
    W_r5 = [-1, 0, 1, 0]
    W_r6 = [0, 0, 0, 1]
    W_r7 = [1, 0, 0, 0]
    
    U_list = [U1, U2, U3, U4, U5, U6, U7]
    V_list = [V1, V2, V3, V4, V5, V6, V7]
    W_list = [W_r1, W_r2, W_r3, W_r4, W_r5, W_r6, W_r7]
    
    # Process 7 full blocks
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for r in range(7):
            # Update U
            for i in range(4):
                if U_list[r][i] != 0:
                    U[i][col] += U_list[r][i]
            
            # Update V
            # V indices:
            # B00 -> j=0, k=k0 -> idx 0*p + k0 = k0
            # B10 -> j=1, k=k0 -> idx 1*p + k0 = p + k0
            # B01 -> j=0, k=k1 -> idx 0*p + k1 = k1
            # B11 -> j=1, k=k1 -> idx 1*p + k1 = p + k1
            
            v_coeff = V_list[r]
            V[k0][col] += v_coeff[0]
            V[p + k0][col] += v_coeff[1]
            V[k1][col] += v_coeff[2]
            V[p + k1][col] += v_coeff[3]
            
            # Update W
            # W indices:
            # C00 -> k=k0, i=0 -> idx k0*n + 0 = k0*2
            # C10 -> k=k0, i=1 -> idx k0*n + 1 = k0*2 + 1
            # C01 -> k=k1, i=0 -> idx k1*n + 0 = k1*2
            # C11 -> k=k1, i=1 -> idx k1*n + 1 = k1*2 + 1
            
            w_coeff = W_list[r]
            W[k0 * 2][col] += w_coeff[0]
            W[k0 * 2 + 1][col] += w_coeff[1]
            W[k1 * 2][col] += w_coeff[2]
            W[k1 * 2 + 1][col] += w_coeff[3]
            
            col += 1
    
    # Process 1 partial block (b=7, k=14)
    # Schoolbook for 2x2 * 2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # B0 = B_{0,14}, B1 = B_{1,14}
    
    k = 14
    
    # M1: A00 * B0
    # U: [1, 0, 0, 0]
    # V: B0 at j=0, k=14 -> idx 14
    # W: C0 at k=14, i=0 -> idx 14*2 = 28
    U[0][col] = 1
    V[14][col] = 1
    W[28][col] = 1
    col += 1
    
    # M2: A01 * B1
    # U: [0, 1, 0, 0]
    # V: B1 at j=1, k=14 -> idx 15+14 = 29
    # W: C0 at k=14, i=0 -> idx 28
    U[1][col] = 1
    V[29][col] = 1
    W[28][col] = 1
    col += 1
    
    # M3: A10 * B0
    # U: [0, 0, 1, 0]
    # V: B0 at j=0, k=14 -> idx 14
    # W: C1 at k=14, i=1 -> idx 28+1 = 29
    U[2][col] = 1
    V[14][col] = 1
    W[29][col] = 1
    col += 1
    
    # M4: A11 * B1
    # U: [0, 0, 0, 1]
    # V: B1 at j=1, k=14 -> idx 29
    # W: C1 at k=14, i=1 -> idx 29
    U[3][col] = 1
    V[29][col] = 1
    W[29][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
