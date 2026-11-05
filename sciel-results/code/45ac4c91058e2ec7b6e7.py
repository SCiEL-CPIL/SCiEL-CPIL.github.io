import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen factors
    # U2, V2, W2 are 2x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [1, 1, 0, 1, 1, 1, -1]
    ]
    V2 = [
        [1, 1, 1, -1, 0, 1, 1],
        [1, 0, -1, 1, 1, 1, 1]
    ]
    W2 = [
        [1, 1, 1, 1, -1, 0, 1],
        [1, -1, 1, 1, 1, 1, 0]
    ]
    
    # Part 1: 4x4x4 (first 4 columns of B)
    # Indices: i (0..3), j (0..3), k (0..3)
    # Block indices: ib (0..1), jb (0..1), kb (0..1)
    # Local indices: i' (0..1), j' (0..1), k' (0..1)
    # i = 2*ib + i', j = 2*jb + j', k = 2*kb + k'
    
    col = 0
    for l in range(7): # Strassen products
        # X_l and Y_l are 2x2 matrices
        # X_l[i', j'] is coeff of A[2*ib+i', 2*jb+j']
        # Y_l[j', k'] is coeff of B[2*jb+j', 2*kb+k']
        
        # Define X_l and Y_l for each l
        # l=0: M1 = (A00+A11)(B00+B11)
        # l=1: M2 = (A10+A11)B00
        # l=2: M3 = A00(B01-B11)
        # l=3: M4 = A11(B10-B00)
        # l=4: M5 = (A00+A01)B11
        # l=5: M6 = (A10-A00)(B00+B01)
        # l=6: M7 = (A01-A11)(B10+B11)
        
        # X_l[i', j']
        X = [[0]*2 for _ in range(2)]
        Y = [[0]*2 for _ in range(2)]
        
        if l == 0:
            # A00+A11
            X[0][0]=1; X[0][1]=1; X[1][0]=1; X[1][1]=1
            # B00+B11
            Y[0][0]=1; Y[0][1]=1; Y[1][0]=1; Y[1][1]=1
        elif l == 1:
            # A10+A11
            X[0][0]=1; X[0][1]=1; X[1][0]=1; X[1][1]=1
            # B00
            Y[0][0]=1
        elif l == 2:
            # A00
            X[0][0]=1
            # B01-B11
            Y[0][1]=1; Y[1][1]=-1
        elif l == 3:
            # A11
            X[1][1]=1
            # B10-B00
            Y[1][0]=1; Y[0][0]=-1
        elif l == 4:
            # A00+A01
            X[0][0]=1; X[0][1]=1
            # B11
            Y[1][1]=1
        elif l == 5:
            # A10-A00
            X[0][0]=-1; X[1][0]=1
            # B00+B01
            Y[0][0]=1; Y[0][1]=1
        elif l == 6:
            # A01-A11
            X[0][1]=1; X[1][1]=-1
            # B10+B11
            Y[1][0]=1; Y[1][1]=1
            
        # Now decompose Z = X Y using U2, V2, W2
        # Z[i', k'] = sum_j' X[i', j'] Y[j', k']
        # Z[i', k'] = sum_r U2[i', r] V2[j', r] W2[k', r]
        # So for each r:
        for r in range(7):
            # U part
            for ib in range(2):
                for i_prime in range(2):
                    i = 2*ib + i_prime
                    # U[i, col] += sum_j' X[i_prime, j'] * U2[i_prime, r] ?
                    # No, X[i', j'] is the coefficient.
                    # U[i, col] is coeff of A[i, j].
                    # A[i, j] = A[2*ib+i', 2*jb+j'].
                    # X[i', j'] is coeff of A[2*ib+i', 2*jb+j'] in X_l.
                    # So U[2*ib+i', 2*jb+j', col] += X[i', j'] * U2[i', r]
                    for jb in range(2):
                        for j_prime in range(2):
                            j = 2*jb + j_prime
                            U[i*m + j][col] += X[i_prime][j_prime] * U2[i_prime][r]
                            
            # V part
            for jb in range(2):
                for j_prime in range(2):
                    j = 2*jb + j_prime
                    # V[j, col] += sum_k' Y[j', k'] * V2[j', r] ?
                    # Y[j', k'] is coeff of B[2*jb+j', 2*kb+k'].
                    # V[2*jb+j', col] += Y[j', k'] * V2[j', r]
                    for kb in range(2):
                        for k_prime in range(2):
                            k = 2*kb + k_prime
                            V[j*p + k][col] += Y[j_prime][k_prime] * V2[j_prime][r]
                            
            # W part
            for kb in range(2):
                for k_prime in range(2):
                    k = 2*kb + k_prime
                    # W[k, col] += W2[k', r]
                    # W index is k*n + i?
                    # Task says W[k*n+i, r].
                    # So W[k, i, col] += W2[k', r] * (coeff from C_l)
                    # C_l contributes to C blocks.
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    coeff = 0
                    if l == 0: # M1
                        if kb == 0 and k_prime == 0: coeff = 1
                        elif kb == 1 and k_prime == 0: coeff = 1
                        elif kb == 0 and k_prime == 1: coeff = 1
                        elif kb == 1 and k_prime == 1: coeff = 1
                    elif l == 1: # M2
                        if kb == 1 and k_prime == 0: coeff = 1
                        elif kb == 1 and k_prime == 1: coeff = -1
                    elif l == 2: # M3
                        if kb == 0 and k_prime == 1: coeff = 1
                        elif kb == 1 and k_prime == 1: coeff = 1
                    elif l == 3: # M4
                        if kb == 0 and k_prime == 0: coeff = 1
                        elif kb == 1 and k_prime == 0: coeff = 1
                    elif l == 4: # M5
                        if kb == 0 and k_prime == 0: coeff = -1
                        elif kb == 0 and k_prime == 1: coeff = 1
                    elif l == 5: # M6
                        if kb == 1 and k_prime == 1: coeff = 1
                    elif l == 6: # M7
                        if kb == 0 and k_prime == 0: coeff = 1
                        elif kb == 1 and k_prime == 0: coeff = 1
                        
                    if coeff != 0:
                        for ib in range(2):
                            for i_prime in range(2):
                                i = 2*ib + i_prime
                                W[k*n + i][col] += coeff * W2[k_prime][r]
                                
            col += 1
            
    # Part 2: 4x4x1 (5th column of B)
    # Rank 16
    for i in range(4):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
