import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 internal Strassen steps for 2x2 matrices
    # U2, V2, W2 are 4x7 matrices
    # Indices: U2[i'*2+j'], V2[j'*2+k'], W2[k'*2+i']
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Step 0: m1 = (A00+A11)(B00+B11)
    # X = A00+A11, Y = B00+B11
    # U2[0,0]=1 (A00), U2[3,0]=1 (A11)
    # V2[0,0]=1 (B00), V2[3,0]=1 (B11)
    # W2[0,0]=1 (C00), W2[3,0]=1 (C11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Step 1: m2 = (A10+A11)B00
    # X = A10+A11, Y = B00
    # U2[2,1]=1 (A10), U2[3,1]=1 (A11)
    # V2[0,1]=1 (B00)
    # W2[1,1]=1 (C10)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    
    # Step 2: m3 = A00(B01-B11)
    # X = A00, Y = B01-B11
    # U2[0,2]=1 (A00)
    # V2[1,2]=1 (B01), V2[3,2]=-1 (B11)
    # W2[2,2]=1 (C01)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1
    
    # Step 3: m4 = A11(B10-B00)
    # X = A11, Y = B10-B00
    # U2[3,3]=1 (A11)
    # V2[2,3]=1 (B10), V2[0,3]=-1 (B00)
    # W2[1,3]=1 (C10), W2[0,3]=1 (C00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[1][3] = 1; W2[0][3] = 1
    
    # Step 4: m5 = (A00+A01)B11
    # X = A00+A01, Y = B11
    # U2[0,4]=1 (A00), U2[1,4]=1 (A01)
    # V2[3,4]=1 (B11)
    # W2[2,4]=1 (C01), W2[0,4]=-1 (C00)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[2][4] = 1; W2[0][4] = -1
    
    # Step 5: m6 = (A10-A00)(B00+B01)
    # X = A10-A00, Y = B00+B01
    # U2[2,5]=1 (A10), U2[0,5]=-1 (A00)
    # V2[0,5]=1 (B00), V2[1,5]=1 (B01)
    # W2[3,5]=1 (C11)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Step 6: m7 = (A01-A11)(B10+B11)
    # X = A01-A11, Y = B10+B11
    # U2[1,6]=1 (A01), U2[3,6]=-1 (A11)
    # V2[2,6]=1 (B10), V2[3,6]=1 (B11)
    # W2[0,6]=1 (C00)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Define the 7 block-level Strassen steps
    # For each r, define X_r and Y_r as 2x2 matrices of linear forms
    # X_r[p][q] is a list of (coeff, I, J, i', j') or similar
    # Simpler: X_r is a 2x2 matrix, each entry is a 4x4 matrix of coefficients for A
    # A[i,j] where i=2I+i', j=2J+j'
    
    # Helper to get coeff of A[i,j] in X_r[p,q]
    # X_r[p,q] is a linear combination of A blocks
    # We represent X_r as a list of 4 4x4 matrices? No.
    # Let's just hardcode the logic for each r.
    
    def get_A_coeff(r, p, q, i, j):
        I = i // 2
        J = j // 2
        ip = i % 2
        jp = j % 2
        
        if r == 0: # X = A00 + A11
            if p == 0 and q == 0:
                if I == 0 and J == 0: return 1
                if I == 1 and J == 1: return 1
            return 0
        elif r == 1: # X = A10 + A11
            if p == 0 and q == 0:
                if I == 1 and J == 0: return 1
                if I == 1 and J == 1: return 1
            return 0
        elif r == 2: # X = A00
            if p == 0 and q == 0:
                if I == 0 and J == 0: return 1
            return 0
        elif r == 3: # X = A11
            if p == 0 and q == 0:
                if I == 1 and J == 1: return 1
            return 0
        elif r == 4: # X = A00 + A01
            if p == 0 and q == 0:
                if I == 0 and J == 0: return 1
                if I == 0 and J == 1: return 1
            return 0
        elif r == 5: # X = A10 - A00
            if p == 0 and q == 0:
                if I == 1 and J == 0: return 1
                if I == 0 and J == 0: return -1
            return 0
        elif r == 6: # X = A01 - A11
            if p == 0 and q == 0:
                if I == 0 and J == 1: return 1
                if I == 1 and J == 1: return -1
            return 0
        return 0

    def get_B_coeff(r, q, k, j, k_idx):
        J = j // 2
        K = k_idx // 2
        jp = j % 2
        kp = k_idx % 2
        
        if r == 0: # Y = B00 + B11
            if q == 0 and k == 0:
                if J == 0 and K == 0: return 1
                if J == 1 and K == 1: return 1
            return 0
        elif r == 1: # Y = B00
            if q == 0 and k == 0:
                if J == 0 and K == 0: return 1
            return 0
        elif r == 2: # Y = B01 - B11
            if q == 0 and k == 0:
                if J == 0 and K == 1: return 1
                if J == 1 and K == 1: return -1
            return 0
        elif r == 3: # Y = B10 - B00
            if q == 0 and k == 0:
                if J == 1 and K == 0: return 1
                if J == 0 and K == 0: return -1
            return 0
        elif r == 4: # Y = B11
            if q == 0 and k == 0:
                if J == 1 and K == 1: return 1
            return 0
        elif r == 5: # Y = B00 + B01
            if q == 0 and k == 0:
                if J == 0 and K == 0: return 1
                if J == 0 and K == 1: return 1
            return 0
        elif r == 6: # Y = B10 + B11
            if q == 0 and k == 0:
                if J == 1 and K == 0: return 1
                if J == 1 and K == 1: return 1
            return 0
        return 0

    def get_C_coeff(r, I, K):
        if r == 0: # C00, C11
            if I == 0 and K == 0: return 1
            if I == 1 and K == 1: return 1
            return 0
        elif r == 1: # C10
            if I == 1 and K == 0: return 1
            return 0
        elif r == 2: # C01
            if I == 0 and K == 1: return 1
            return 0
        elif r == 3: # C00
            if I == 0 and K == 0: return 1
            return 0
        elif r == 4: # C00 (-1)
            if I == 0 and K == 0: return -1
            return 0
        elif r == 5: # C11
            if I == 1 and K == 1: return 1
            return 0
        elif r == 6: # C00
            if I == 0 and K == 0: return 1
            return 0
        return 0

    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            
            # Fill U
            for i in range(4):
                for j in range(4):
                    val = 0
                    for p in range(2):
                        for q in range(2):
                            # U2 index: p*2 + q
                            u2_val = U2[p * 2 + q][s]
                            if u2_val != 0:
                                a_coeff = get_A_coeff(r, p, q, i, j)
                                val += u2_val * a_coeff
                    U[i * 4 + j][col] = val
                    
            # Fill V
            for j in range(4):
                for k in range(4):
                    val = 0
                    for q in range(2):
                        for k_prime in range(2):
                            # V2 index: q*2 + k_prime
                            v2_val = V2[q * 2 + k_prime][s]
                            if v2_val != 0:
                                b_coeff = get_B_coeff(r, q, k_prime, j, k)
                                val += v2_val * b_coeff
                    V[j * 4 + k][col] = val
                    
            # Fill W
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    I = i // 2
                    kp = k % 2
                    ip = i % 2
                    val = 0
                    # W2 index: kp*2 + ip
                    w2_val = W2[kp * 2 + ip][s]
                    if w2_val != 0:
                        c_coeff = get_C_coeff(r, I, K)
                        val = w2_val * c_coeff
                    W[k * 4 + i][col] = val

    return {"U": U, "V": V, "W": W}
