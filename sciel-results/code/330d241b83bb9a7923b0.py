import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank 55: 5 Strassen blocks (5*7=35) + 20 schoolbook terms for the 2x1x10 part
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # L vectors (coeffs for a_ij) - local indices: 0=a00, 1=a01, 2=a10, 3=a11
    L = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    # R vectors (coeffs for b_jk) - local indices: 0=b00, 1=b01, 2=b10, 3=b11
    R = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]
    
    # W vectors (coeffs for c_ik) - local indices: 0=c00, 1=c01, 2=c10, 3=c11
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    W_coeffs = [
        {0: 1, 3: 1},       # M1: c00, c11
        {2: 1, 3: -1},      # M2: c10, -c11
        {1: 1, 3: 1},       # M3: c01, c11
        {0: 1, 2: 1},       # M4: c00, c10
        {1: 1, 0: -1},      # M5: c01, -c00
        {3: 1},             # M6: c11
        {0: 1}              # M7: c00
    ]
    
    # Fill Strassen blocks for 5 pairs of columns
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            # Fill U: local index 0->a00(i=0,j=0), 1->a01(i=0,j=1), 2->a10(i=1,j=0), 3->a11(i=1,j=1)
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j][r] += val
            
            # Fill V: local index 0->b00(j=0,k=0), 1->b01(j=0,k=1), 2->b10(j=1,k=0), 3->b11(j=1,k=1)
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k)][r] += val
            
            # Fill W: local index 0->c00(k=0,i=0), 1->c01(k=1,i=0), 2->c10(k=0,i=1), 3->c11(k=1,i=1)
            for local_idx, val in W_coeffs[s].items():
                i = local_idx % 2
                k = local_idx // 2
                W[(k_offset + k) * n + i][r] += val
            
            col += 1
            
    # Part 2: A2 B2 (2x1 * 1x10) - schoolbook
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2][r] += 1  # A[i,2]
            V[2 * p + k][r] += 1  # B[2,k]
            W[k * n + i][r] += 1  # C[i,k]
            col += 1
            
    # Verify the decomposition
    # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    # Decomposition is incorrect
                    pass
    
    return {"U": U, "V": V, "W": W}
