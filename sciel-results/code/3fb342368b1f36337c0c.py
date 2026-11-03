import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 4, 4, 8
    
    # 1. Define 2x2x2 Strassen tensors (rank 7)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    # M3
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[0, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    # M4
    U2[3, 3] = 1
    V2[3, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[3, 3] = 1
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[1, 5] = 1; W2[2, 5] = 1
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1
    
    # 2. Define 4x4x4 Strassen tensors (rank 49)
    # Outer coefficients for 2x2 block Strassen
    # M1: A00+A11, B00+B11, C00+C11
    # M2: A10+A11, B00, C10-C11
    # M3: A00, B10-B00, C10+C11
    # M4: A11, B11-B00, C00+C11
    # M5: A00+A01, B11, C01-C00
    # M6: A10-A00, B00+B01, C01+C10
    # M7: A01-A11, B10+B11, C00+C11
    
    outer_A = [
        np.array([[1,0],[0,1]]),
        np.array([[0,0],[1,1]]),
        np.array([[1,0],[0,0]]),
        np.array([[0,0],[0,1]]),
        np.array([[1,1],[0,0]]),
        np.array([[-1,0],[1,0]]),
        np.array([[0,1],[-1,0]])
    ]
    outer_B = [
        np.array([[1,0],[0,1]]),
        np.array([[1,0],[0,0]]),
        np.array([[0,1],[0,-1]]),
        np.array([[-1,0],[1,0]]),
        np.array([[0,0],[0,1]]),
        np.array([[1,1],[0,0]]),
        np.array([[0,0],[1,1]])
    ]
    outer_C = [
        np.array([[1,0],[0,1]]),
        np.array([[0,0],[1,0]]),
        np.array([[0,1],[0,0]]),
        np.array([[0,1],[1,0]]),
        np.array([[-1,1],[0,0]]),
        np.array([[0,0],[0,1]]),
        np.array([[1,0],[0,1]])
    ]
    
    # Wait, I need to check the outer coefficients against the 2x2 Strassen formulas above.
    # M1: A(0,0)+A(1,1) -> [[1,0],[0,1]]. Correct.
    # M2: A(1,0)+A(1,1) -> [[0,0],[1,1]]. Correct.
    # M3: A(0,0) -> [[1,0],[0,0]]. Correct.
    # M4: A(1,1) -> [[0,0],[0,1]]. Correct.
    # M5: A(0,0)+A(0,1) -> [[1,1],[0,0]]. Correct.
    # M6: A(1,0)-A(0,0) -> [[-1,0],[1,0]]. Correct.
    # M7: A(0,1)-A(1,1) -> [[0,1],[-1,0]]. Correct.
    
    # M1: B(0,0)+B(1,1) -> [[1,0],[0,1]]. Correct.
    # M2: B(0,0) -> [[1,0],[0,0]]. Correct.
    # M3: B(1,0)-B(0,0) -> [[-1,0],[1,0]]? 
    # My outer_B[2] is [[0,1],[0,-1]]. This is B(0,1)-B(1,1). 
    # But M3 in 2x2x2 was A(0,0)(B(1,0)-B(0,0)).
    # So outer_B[2] should be [[-1,0],[1,0]].
    # Let's re-verify M3.
    # Standard Strassen:
    # P3 = A00 * (B10 - B00)
    # So B part is B10 - B00.
    # B10 is index (1,0). B00 is index (0,0).
    # Matrix: [[-1, 0], [1, 0]].
    # Code has: np.array([[0,1],[0,-1]]). This is WRONG.
    
    # M4: B(1,1)-B(0,0) -> [[-1,0],[0,1]].
    # Code has: np.array([[-1,0],[1,0]]). This is WRONG.
    
    # M5: B(1,1) -> [[0,0],[0,1]]. Correct.
    # M6: B(0,0)+B(0,1) -> [[1,1],[0,0]]. Correct.
    # M7: B(1,0)+B(1,1) -> [[0,0],[1,1]]. Correct.
    
    # So outer_B[2] and outer_B[3] are wrong in the previous snippet I analyzed.
    # I will fix them in the replacement.
    
    outer_B = [
        np.array([[1,0],[0,1]]), # M1
        np.array([[1,0],[0,0]]), # M2
        np.array([[-1,0],[1,0]]), # M3: B10 - B00
        np.array([[-1,0],[0,1]]), # M4: B11 - B00
        np.array([[0,0],[0,1]]), # M5
        np.array([[1,1],[0,0]]), # M6
        np.array([[0,0],[1,1]])  # M7
    ]
    
    # Check outer_C
    # M1: C00+C11 -> [[1,0],[0,1]]. Correct.
    # M2: C10-C11 -> [[0,0],[1,-1]]?
    # C10 is (1,0). C11 is (1,1).
    # Matrix: [[0,0],[1,-1]].
    # Code has: np.array([[0,0],[1,0]]). This is WRONG.
    
    # M3: C10+C11 -> [[0,0],[1,1]].
    # Code has: np.array([[0,1],[0,0]]). This is WRONG.
    
    # M4: C00+C11 -> [[1,0],[0,1]].
    # Code has: np.array([[0,1],[1,0]]). This is WRONG.
    
    # M5: C01-C00 -> [[-1,1],[0,0]]. Correct.
    # M6: C01+C10 -> [[0,1],[1,0]].
    # Code has: np.array([[0,0],[0,1]]). This is WRONG.
    
    # M7: C00+C11 -> [[1,0],[0,1]]. Correct.
    
    outer_C = [
        np.array([[1,0],[0,1]]), # M1
        np.array([[0,0],[1,-1]]), # M2
        np.array([[0,0],[1,1]]), # M3
        np.array([[1,0],[0,1]]), # M4
        np.array([[-1,1],[0,0]]), # M5
        np.array([[0,1],[1,0]]), # M6
        np.array([[1,0],[0,1]])  # M7
    ]

    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for r_out in range(7):
        for r_in in range(7):
            r = r_out * 7 + r_in
            
            # Construct U44[:, r]
            for bi in range(2):
                for bj in range(2):
                    coeff_a = outer_A[r_out][bi, bj]
                    if coeff_a == 0: continue
                    for ri in range(2):
                        for cj in range(2):
                            flat_a = (bi*2+ri)*4 + (bj*2+cj)
                            inner_idx = ri*2 + cj
                            U44[flat_a, r] += coeff_a * U2[inner_idx, r_in]
            
            # Construct V44[:, r]
            for bi in range(2):
                for bj in range(2):
                    coeff_b = outer_B[r_out][bi, bj]
                    if coeff_b == 0: continue
                    for ri in range(2):
                        for cj in range(2):
                            flat_b = (bi*2+ri)*4 + (bj*2+cj)
                            inner_idx = ri*2 + cj
                            V44[flat_b, r] += coeff_b * V2[inner_idx, r_in]
            
            # Construct W44[:, r]
            for bi in range(2):
                for bj in range(2):
                    coeff_c = outer_C[r_out][bi, bj]
                    if coeff_c == 0: continue
                    for ri in range(2):
                        for cj in range(2):
                            flat_c = (bi*2+ri)*4 + (bj*2+cj)
                            inner_idx = ri*2 + cj
                            W44[flat_c, r] += coeff_c * W2[inner_idx, r_in]
                            
    # 3. Construct 4x4x8 decomposition (rank 98)
    U98 = np.zeros((16, 98))
    V98 = np.zeros((32, 98))
    W98 = np.zeros((32, 98))
    
    # Left half
    for r in range(49):
        U98[:, r] = U44[:, r]
        V98[0:16, r] = V44[:, r]
        for k_44 in range(16):
            row = k_44 // 4
            col = k_44 % 4
            k_48 = row * 8 + col
            W98[k_48, r] = W44[k_44, r]
            
    # Right half
    for r in range(49):
        r_98 = 49 + r
        U98[:, r_98] = U44[:, r]
        for k_B_44 in range(16):
            row = k_B_44 // 4
            col_local = k_B_44 % 4
            k_B_48 = row * 8 + col_local + 4
            V98[k_B_48, r_98] = V44[k_B_44, r]
        for k_C_44 in range(16):
            row = k_C_44 // 4
            col_local = k_C_44 % 4
            k_C_48 = row * 8 + col_local + 4
            W98[k_C_48, r_98] = W44[k_C_44, r]
            
    return {"U": U98.tolist(), "V": V98.tolist(), "W": W98.tolist()}
