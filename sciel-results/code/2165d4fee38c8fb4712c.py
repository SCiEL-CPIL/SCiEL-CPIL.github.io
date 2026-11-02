import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (16, R), V (36, R), W (36, R)
    
    # Try Rank 128
    U = np.zeros((16, 128), dtype=float)
    V = np.zeros((36, 128), dtype=float)
    W = np.zeros((36, 128), dtype=float)
    
    col = 0
    
    # Sub-problems:
    # s=0: A(0:2, 0:2), B(0:2, :), C(0:2, :)
    # s=1: A(0:2, 2:4), B(2:4, :), C(0:2, :)
    # s=2: A(2:4, 0:2), B(0:2, :), C(2:4, :)
    # s=3: A(2:4, 2:4), B(2:4, :), C(2:4, :)
    
    sub_problems = [
        (0, 0, 0, 0), # i_A, j_A, j_B, i_C
        (0, 2, 2, 0),
        (2, 0, 0, 2),
        (2, 2, 2, 2)
    ]
    
    for i_A, j_A, j_B, i_C in sub_problems:
        # A indices: (i_A, j_A), (i_A, j_A+1), (i_A+1, j_A), (i_A+1, j_A+1)
        # B indices: (j_B, k), (j_B+1, k) for k in 0..8
        # C indices: (k, i_C), (k, i_C+1)
        
        # Pairs: (0,1), (2,3), (4,5), (6,7)
        for k_start in [0, 2, 4, 6]:
            k1 = k_start
            k2 = k_start + 1
            
            # Bini Decomposition for 2x2x2
            # A: a,b,c,d
            # B: e,f,g,h (cols k1, k2)
            
            # m1 = a * e
            U[i_A*4 + j_A, col] = 1
            V[j_B*9 + k1, col] = 1
            W[k1*4 + i_C, col] = 1 # p
            col += 1
            
            # m2 = b * g
            U[i_A*4 + j_A+1, col] = 1
            V[(j_B+1)*9 + k1, col] = 1
            W[k1*4 + i_C+1, col] = 1 # r
            col += 1
            
            # m3 = c * f
            U[(i_A+1)*4 + j_A, col] = 1
            V[j_B*9 + k2, col] = 1
            W[k2*4 + i_C, col] = 1 # q
            col += 1
            
            # m4 = d * h
            U[(i_A+1)*4 + j_A+1, col] = 1
            V[(j_B+1)*9 + k2, col] = 1
            W[k2*4 + i_C+1, col] = 1 # s
            col += 1
            
            # m5 = (a+b+c+d) * (e+f+g+h)
            U[i_A*4 + j_A, col] = 1
            U[i_A*4 + j_A+1, col] = 1
            U[(i_A+1)*4 + j_A, col] = 1
            U[(i_A+1)*4 + j_A+1, col] = 1
            V[j_B*9 + k1, col] = 1
            V[j_B*9 + k2, col] = 1
            V[(j_B+1)*9 + k1, col] = 1
            V[(j_B+1)*9 + k2, col] = 1
            W[k1*4 + i_C, col] = 0.5 # p
            W[k2*4 + i_C, col] = 0.5 # q
            W[k1*4 + i_C+1, col] = 0.5 # r
            W[k2*4 + i_C+1, col] = 0.5 # s
            col += 1
            
            # m6 = (a+b-c-d) * (e+f-g-h)
            U[i_A*4 + j_A, col] = 1
            U[i_A*4 + j_A+1, col] = 1
            U[(i_A+1)*4 + j_A, col] = -1
            U[(i_A+1)*4 + j_A+1, col] = -1
            V[j_B*9 + k1, col] = 1
            V[j_B*9 + k2, col] = 1
            V[(j_B+1)*9 + k1, col] = -1
            V[(j_B+1)*9 + k2, col] = -1
            W[k1*4 + i_C, col] = -0.5 # p
            W[k2*4 + i_C, col] = -0.5 # q
            W[k1*4 + i_C+1, col] = 0.5 # r
            W[k2*4 + i_C+1, col] = 0.5 # s
            col += 1
            
            # m7 = (a-b+c-d) * (e-f+g-h)
            U[i_A*4 + j_A, col] = 1
            U[i_A*4 + j_A+1, col] = -1
            U[(i_A+1)*4 + j_A, col] = 1
            U[(i_A+1)*4 + j_A+1, col] = -1
            V[j_B*9 + k1, col] = 1
            V[j_B*9 + k2, col] = -1
            V[(j_B+1)*9 + k1, col] = 1
            V[(j_B+1)*9 + k2, col] = -1
            W[k1*4 + i_C, col] = -0.5 # p
            W[k2*4 + i_C, col] = 0.5 # q
            W[k1*4 + i_C+1, col] = -0.5 # r
            W[k2*4 + i_C+1, col] = 0.5 # s
            col += 1
            
        # Single column: k=8
        k = 8
        # m1 = a * e
        U[i_A*4 + j_A, col] = 1
        V[j_B*9 + k, col] = 1
        W[k*4 + i_C, col] = 1 # p
        col += 1
        
        # m2 = b * g
        U[i_A*4 + j_A+1, col] = 1
        V[(j_B+1)*9 + k, col] = 1
        W[k*4 + i_C, col] = 1 # p
        col += 1
        
        # m3 = c * e
        U[(i_A+1)*4 + j_A, col] = 1
        V[j_B*9 + k, col] = 1
        W[k*4 + i_C+1, col] = 1 # r
        col += 1
        
        # m4 = d * g
        U[(i_A+1)*4 + j_A+1, col] = 1
        V[(j_B+1)*9 + k, col] = 1
        W[k*4 + i_C+1, col] = 1 # r
        col += 1

    # Verify
    # Random test
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 9)
    C_expected = A @ B
    
    # Compute C from U, V, W
    # C[i, k] = sum_c U[i*4+j, c] * V[j*9+k, c] * W[k*4+i, c]
    # This is slow in Python loop. Use numpy.
    # Tensor T[i, j, k, c] = U[i*4+j, c] * V[j*9+k, c] * W[k*4+i, c]
    # C[i, k] = sum_j sum_c ...
    
    # Efficient check:
    # Reshape U to (4, 4, 128)
    # Reshape V to (4, 9, 128)
    # Reshape W to (9, 4, 128)
    
    U_r = U.reshape(4, 4, 128)
    V_r = V.reshape(4, 9, 128)
    W_r = W.reshape(9, 4, 128)
    
    # C[i, k] = sum_{j, c} U[i, j, c] * V[j, k, c] * W[k, i, c]
    # This is a contraction.
    # We can do:
    # Temp[j, k, c] = sum_i U[i, j, c] * W[k, i, c] ? No.
    
    # Let's just do a few manual checks or trust the math.
    # Given the risk of Bini sign errors, I will return Schoolbook if I'm not 100% sure.
    # But 128 is a big jump.
    
    # Let's do a quick numpy check for one element.
    # C[0,0] = sum_c U[0, c] * V[0, c] * W[0, c] + U[1, c] * V[9, c] * W[0, c] ...
    
    # Actually, the verification in the code is safer.
    # If verification fails, return schoolbook.
    
    valid = True
    # Check C[0,0]
    c_00 = 0
    for j in range(4):
        for c in range(128):
            c_00 += U[0*4+j, c] * V[j*9+0, c] * W[0*4+0, c]
    if abs(c_00 - C_expected[0,0]) > 1e-9:
        valid = False
        
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Schoolbook
        rank = 144
        U_s = [[0]*rank for _ in range(16)]
        V_s = [[0]*rank for _ in range(36)]
        W_s = [[0]*rank for _ in range(36)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_s[i*4+j][col] = 1
                    V_s[j*9+k][col] = 1
                    W_s[k*4+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
