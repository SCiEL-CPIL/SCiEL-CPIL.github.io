import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # Base 2x2x2 Strassen factors
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                if abs(s - 1) > 1e-10:
                    raise ValueError(f"2x2x2 verification failed at ({i},{j},{k}): got {s}")
    
    # Build 4x4x4
    rank_444 = 49
    U444 = np.zeros((16, rank_444))
    V444 = np.zeros((16, rank_444))
    W444 = np.zeros((16, rank_444))
    
    col = 0
    for s in range(7):
        L_s = np.zeros((4, 4))
        R_s = np.zeros((4, 4))
        
        if s == 0:
            L_s[0:2, 0:2] = np.eye(2); L_s[2:4, 2:4] = np.eye(2)
            R_s[0:2, 0:2] = np.eye(2); R_s[2:4, 2:4] = np.eye(2)
        elif s == 1:
            L_s[2:4, 0:2] = np.eye(2); L_s[2:4, 2:4] = np.eye(2)
            R_s[0:2, 0:2] = np.eye(2)
        elif s == 2:
            L_s[0:2, 0:2] = np.eye(2)
            R_s[0:2, 2:4] = np.eye(2); R_s[2:4, 2:4] = -np.eye(2)
        elif s == 3:
            L_s[2:4, 2:4] = np.eye(2)
            R_s[2:4, 0:2] = np.eye(2); R_s[0:2, 0:2] = -np.eye(2)
        elif s == 4:
            L_s[0:2, 0:2] = np.eye(2); L_s[0:2, 2:4] = np.eye(2)
            R_s[2:4, 2:4] = np.eye(2)
        elif s == 5:
            L_s[2:4, 0:2] = np.eye(2); L_s[0:2, 0:2] = -np.eye(2)
            R_s[0:2, 0:2] = np.eye(2); R_s[0:2, 2:4] = np.eye(2)
        elif s == 6:
            L_s[0:2, 2:4] = np.eye(2); L_s[2:4, 2:4] = -np.eye(2)
            R_s[2:4, 0:2] = np.eye(2); R_s[2:4, 2:4] = np.eye(2)
        
        L00, L01 = L_s[0:2, 0:2], L_s[0:2, 2:4]
        L10, L11 = L_s[2:4, 0:2], L_s[2:4, 2:4]
        R00, R01 = R_s[0:2, 0:2], R_s[0:2, 2:4]
        R10, R11 = R_s[2:4, 0:2], R_s[2:4, 2:4]
        
        inner_terms = [
            (L00 + L11, R00 + R11),
            (L10 + L11, R00),
            (L00, R01 - R11),
            (L11, R10 - R00),
            (L00 + L01, R11),
            (L10 - L00, R00 + R01),
            (L01 - L11, R10 + R11)
        ]
        
        for t in range(7):
            X, Y = inner_terms[t]
            for t_prime in range(7):
                for i in range(2):
                    for j in range(2):
                        cx = X[i, j]
                        if cx != 0:
                            U444[i*4+j, col] += cx * U2[i*2+j, t_prime]
                for j in range(2):
                    for k in range(2):
                        cy = Y[j, k]
                        if cy != 0:
                            V444[j*4+k, col] += cy * V2[j*2+k, t_prime]
                for i in range(2):
                    for k in range(2):
                        cw = W2[k*2+i, t_prime]
                        if cw != 0:
                            W444[k*4+i, col] += cw
                col += 1
    
    # --- Part 2: 4x4x1 for column 4 of B (Rank 16) ---
    rank_441 = 16
    U441 = np.zeros((16, rank_441))
    V441 = np.zeros((20, rank_441))
    W441 = np.zeros((20, rank_441))
    
    col2 = 0
    for i in range(4):
        for j in range(4):
            U441[i*4+j, col2] = 1
            V441[j*5+4, col2] = 1
            W441[4*4+i, col2] = 1
            col2 += 1
    
    # Combine
    rank_total = 65
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    U[:, 0:49] = U444
    V[0:16, 0:49] = V444
    W[0:16, 0:49] = W444
    
    U[:, 49:65] = U441
    V[:, 49:65] = V441
    W[:, 49:65] = W441
    
    return {"U": U, "V": V, "W": W}
