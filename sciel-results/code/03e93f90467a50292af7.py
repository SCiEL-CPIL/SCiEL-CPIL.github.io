"""Matrix multiplication 2,2,8 - rank-28 decomposition using Strassen-like per-column-pair."""
DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # Partition p=8 columns into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair, use 7-term Strassen-like decomposition
    # Total rank = 4 * 7 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (4, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (16, 28)
    
    # Strassen-like 7-term decomposition for 2x2 * 2x2
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # M1 = a0*(b0+b3), M2 = (a0+a1)*b3, M3 = (a2+a3)*b0, M4 = a3*(b1+b2)
    # M5 = (a0+a3)*(b0+b1), M6 = (a1+a3)*(b2+b3), M7 = (a0+a2)*(b0+b2)
    # C = [[M5+M4-M2+M6, M1+M2-M4-M7], [M1+M3-M5-M6, M2+M3+M4]]
    
    # U factor: rows are (i*m+j) = (0,1,2,3) for A matrix entries
    # For each column pair t (0-3), and each Strassen term s (0-6):
    #   column index = t*7 + s
    #   U[row][col] = coefficient of A[row] in term s
    
    # Strassen terms and their A-coefficients:
    # M1: a0*(b0+b3) -> a0:1, a1:0, a2:0, a3:0
    # M2: (a0+a1)*b3 -> a0:1, a1:1, a2:0, a3:0
    # M3: (a2+a3)*b0 -> a0:0, a1:0, a2:1, a3:1
    # M4: a3*(b1+b2) -> a0:0, a1:0, a2:0, a3:1
    # M5: (a0+a3)*(b0+b1) -> a0:1, a1:0, a2:0, a3:1
    # M6: (a1+a3)*(b2+b3) -> a0:0, a1:1, a2:0, a3:1
    # M7: (a0+a2)*(b0+b2) -> a0:1, a1:0, a2:1, a3:0
    
    A_coeffs = [
        [1, 0, 0, 0],  # M1: a0
        [1, 1, 0, 0],  # M2: a0+a1
        [0, 0, 1, 1],  # M3: a2+a3
        [0, 0, 0, 1],  # M4: a3
        [1, 0, 0, 1],  # M5: a0+a3
        [0, 1, 0, 1],  # M6: a1+a3
        [1, 0, 1, 0],  # M7: a0+a2
    ]
    
    # V factor: rows are (j*p+k) for B matrix entries
    # For column pair t, B columns are 2t and 2t+1
    # B = [[b0, b1], [b2, b3]] where b0=B[2t], b1=B[2t+1], b2=B[2t+2], b3=B[2t+3]
    # Wait, for 2x8: B has rows j=0,1 and columns k=0..7
    # For pair t, the relevant B submatrix is B[:, 2t:2t+2]
    # So b0 = B[0, 2t], b1 = B[0, 2t+1], b2 = B[1, 2t], b3 = B[1, 2t+1]
    
    # V[row][col] where row = j*p + k
    # For term s in pair t:
    #   M1: b0+b3 -> B[0,2t]:1, B[1,2t+1]:1
    #   M2: b3 -> B[1,2t+1]:1
    #   M3: b0 -> B[0,2t]:1
    #   M4: b1+b2 -> B[0,2t+1]:1, B[1,2t]:1
    #   M5: b0+b1 -> B[0,2t]:1, B[0,2t+1]:1
    #   M6: b2+b3 -> B[1,2t]:1, B[1,2t+1]:1
    #   M7: b0+b2 -> B[0,2t]:1, B[1,2t]:1
    
    B_coeffs = [
        [(0, 1), (1, 1)],  # M1: b0+b3 -> (j=0,k=2t):1, (j=1,k=2t+1):1
        [(1, 1)],           # M2: b3 -> (j=1,k=2t+1):1
        [(0, 1)],           # M3: b0 -> (j=0,k=2t):1
        [(0, 1), (1, 1)],  # M4: b1+b2 -> (j=0,k=2t+1):1, (j=1,k=2t):1
        [(0, 1), (0, 1)],  # M5: b0+b1 -> (j=0,k=2t):1, (j=0,k=2t+1):1
        [(1, 1), (1, 1)],  # M6: b2+b3 -> (j=1,k=2t):1, (j=1,k=2t+1):1
        [(0, 1), (1, 1)],  # M7: b0+b2 -> (j=0,k=2t):1, (j=1,k=2t):1
    ]
    
    # W factor: rows are (k*n+i) for C matrix entries
    # C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # For pair t, output columns are 2t and 2t+1
    # C[0,2t] = M5+M4-M2+M6, C[0,2t+1] = M1+M2-M4-M7
    # C[1,2t] = M1+M3-M5-M6, C[1,2t+1] = M2+M3+M4
    
    # W[row][col] where row = k*n + i
    # For pair t, term s:
    #   C[0,2t] = M5+M4-M2+M6 -> W[2t*n+0][col] for terms 4,3,1,5 with coeffs 1,1,-1,1
    #   C[0,2t+1] = M1+M2-M4-M7 -> W[(2t+1)*n+0][col] for terms 0,1,3,6 with coeffs 1,1,-1,-1
    #   C[1,2t] = M1+M3-M5-M6 -> W[2t*n+1][col] for terms 0,2,4,5 with coeffs 1,1,-1,-1
    #   C[1,2t+1] = M2+M3+M4 -> W[(2t+1)*n+1][col] for terms 1,2,3 with coeffs 1,1,1
    
    # W_coeffs[output_pos][term] = coefficient
    # output_pos: 0=C[0,2t], 1=C[0,2t+1], 2=C[1,2t], 3=C[1,2t+1]
    W_coeffs = [
        [0, -1, 0, 1, 1, 1, 0],   # C[0,2t] = M5+M4-M2+M6
        [1, 1, 0, -1, 0, 0, -1],  # C[0,2t+1] = M1+M2-M4-M7
        [1, 0, 1, 0, -1, -1, 0],  # C[1,2t] = M1+M3-M5-M6
        [0, 1, 1, 1, 0, 0, 0],    # C[1,2t+1] = M2+M3+M4
    ]
    
    for t in range(4):
        for s in range(7):
            col = t * 7 + s
            
            # Fill U
            for i in range(4):
                U[i][col] = A_coeffs[s][i]
            
            # Fill V
            # B_coeffs[s] is list of (j, k_offset) where k = 2t + k_offset
            for (j, k_off) in B_coeffs[s]:
                k = 2 * t + k_off
                row = j * p + k
                V[row][col] = 1
            
            # Fill W
            # For each output position in this pair
            for op in range(4):
                coeff = W_coeffs[op][s]
                if coeff != 0:
                    if op == 0:  # C[0, 2t]
                        row = (2 * t) * n + 0
                    elif op == 1:  # C[0, 2t+1]
                        row = (2 * t + 1) * n + 0
                    elif op == 2:  # C[1, 2t]
                        row = (2 * t) * n + 1
                    else:  # op == 3, C[1, 2t+1]
                        row = (2 * t + 1) * n + 1
                    W[row][col] = coeff
    
    return {"U": U, "V": V, "W": W}
