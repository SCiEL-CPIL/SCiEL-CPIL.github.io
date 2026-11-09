import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = 4, 4, 5
    
    # Strassen 2x2x2 parameters
    # U_2: 4x7, V_2: 4x7, W_2: 4x7
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # P1: (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # P2: (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # P3: A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # P4: A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # P5: (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # P6: (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # P7: (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Outer 4x4x4 Strassen parameters
    # M_k, N_k are 2x2 matrices of coefficients for blocks
    M = []
    N = []
    # P1: A00+A11, B00+B11
    M.append(np.array([[1, 0], [0, 1]]))
    N.append(np.array([[1, 0], [0, 1]]))
    # P2: A10+A11, B00
    M.append(np.array([[0, 0], [1, 1]]))
    N.append(np.array([[1, 0], [0, 0]]))
    # P3: A00, B01-B11
    M.append(np.array([[1, 0], [0, 0]]))
    N.append(np.array([[0, 1], [0, -1]]))
    # P4: A11, B10-B00
    M.append(np.array([[0, 0], [0, 1]]))
    N.append(np.array([[-1, 0], [1, 0]]))
    # P5: A00+A01, B11
    M.append(np.array([[1, 1], [0, 0]]))
    N.append(np.array([[0, 0], [0, 1]]))
    # P6: A10-A00, B00+B01
    M.append(np.array([[-1, 0], [1, 0]]))
    N.append(np.array([[1, 1], [0, 0]]))
    # P7: A01-A11, B10+B11
    M.append(np.array([[0, 1], [0, -1]]))
    N.append(np.array([[0, 1], [1, 1]]))

    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))

    for k_out in range(7):
        for r_in in range(7):
            col = k_out * 7 + r_in
            
            # Fill U_444
            for i in range(4):
                for j in range(4):
                    ib = i // 2
                    jb = j // 2
                    ip = i % 2
                    jp = j % 2
                    U_444[i * 4 + j, col] = M[k_out][ib, jb] * U_2[ip * 2 + jp, r_in]
            
            # Fill V_444
            for j in range(4):
                for k in range(4):
                    jb = j // 2
                    kb = k // 2
                    jp = j % 2
                    kp = k % 2
                    V_444[j * 4 + k, col] = N[k_out][jb, kb] * V_2[jp * 2 + kp, r_in]
            
            # Fill W_444
            for k in range(4):
                for i in range(4):
                    kb = k // 2
                    ib = i // 2
                    kp = k % 2
                    ip = i % 2
                    # W_2 index is k_inner * 2 + i_inner
                    W_444[k * 4 + i, col] = W_2[kp * 2 + ip, r_in] * W_2[kb * 2 + ib, k_out] # Wait, W_2 for outer?
                    # Outer W is not in W_2.
                    # Outer W coefficients:
                    # We need W_outer[k_out, ib, kb]
                    
    # Let's define W_outer explicitly
    W_outer = np.zeros((7, 2, 2))
    # P1: C00, C11
    W_outer[0, 0, 0] = 1; W_outer[0, 1, 1] = 1
    # P2: C10, C11(-1)
    W_outer[1, 1, 0] = 1; W_outer[1, 1, 1] = -1
    # P3: C01, C11
    W_outer[2, 0, 1] = 1; W_outer[2, 1, 1] = 1
    # P4: C00, C10
    W_outer[3, 0, 0] = 1; W_outer[3, 1, 0] = 1
    # P5: C00(-1), C01
    W_outer[4, 0, 0] = -1; W_outer[4, 0, 1] = 1
    # P6: C11
    W_outer[5, 1, 1] = 1
    # P7: C00
    W_outer[6, 0, 0] = 1

    # Refill W_444
    for k_out in range(7):
        for r_in in range(7):
            col = k_out * 7 + r_in
            for k in range(4):
                for i in range(4):
                    kb = k // 2
                    ib = i // 2
                    kp = k % 2
                    ip = i % 2
                    W_444[k * 4 + i, col] = W_2[kp * 2 + ip, r_in] * W_outer[k_out, ib, kb]

    # Combine with 4x4x1 part
    rank_total = 65
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    U[:, :49] = U_444
    V[:, :49] = V_444
    W[:, :49] = W_444
    
    # 4x4x1 part
    # A (4x4) * B_right (4x1) -> C_right (4x1)
    # B_right is column 4 of B (index 4).
    # C_right is column 4 of C (index 4).
    for i in range(4):
        for j in range(4):
            t = 49 + i * 4 + j
            U[i * 4 + j, t] = 1
            V[j * 5 + 4, t] = 1
            W[4 * 4 + i, t] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
