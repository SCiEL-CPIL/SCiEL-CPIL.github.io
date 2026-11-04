"""Matrix multiplication 4x4x5 - rank minimization via Strassen decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS

def verify_decomposition(U, V, W):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 exactly using integers."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct_2x2x2_strassen():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11

    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11

    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00

    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    U_2, V_2, W_2 = construct_2x2x2_strassen()
    
    # Build 4x4x4 part using Strassen (rank 49)
    U_4 = []
    V_4 = []
    W_4 = []
    
    # Outer Strassen terms for 4x4x4
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    outer_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),      # M1: (A00+A11)(B00+B11)
        ({2: 1, 3: 1}, {0: 1}),             # M2: (A10+A11)B00
        ({0: 1}, {1: 1, 3: -1}),            # M3: A00(B01-B11)
        ({3: 1}, {2: 1, 0: -1}),            # M4: A11(B10-B00)
        ({0: 1, 1: 1}, {3: 1}),             # M5: (A00+A01)B11
        ({2: 1, 0: -1}, {0: 1, 1: 1}),      # M6: (A10-A00)(B00+B01)
        ({1: 1, 3: -1}, {2: 1, 3: 1})       # M7: (A01-A11)(B10+B11)
    ]
    
    for r_o, (a_coeffs, b_coeffs) in enumerate(outer_terms):
        for r_i in range(7):
            new_U = [0] * 16
            new_V = [0] * 20
            new_W = [0] * 20
            
            for i0 in range(2):
                for j0 in range(2):
                    for k0 in range(2):
                        w_coeff = W_2[k0*2+i0][r_i]
                        u_coeff = U_2[i0*2+j0][r_i]
                        v_coeff = V_2[j0*2+k0][r_i]
                        
                        if w_coeff == 0 or u_coeff == 0 or v_coeff == 0:
                            continue
                        
                        for ba, a_val in a_coeffs.items():
                            for bb, b_val in b_coeffs.items():
                                if a_val == 0 or b_val == 0:
                                    continue
                                if (ba % 2) != (bb // 2):
                                    continue
                                
                                i_global = 2*(ba//2) + i0
                                j_global = 2*(ba%2) + j0
                                k_global = 2*(bb%2) + k0
                                coeff = w_coeff * u_coeff * v_coeff * a_val * b_val
                                
                                if coeff != 0:
                                    new_U[i_global*4 + j_global] += coeff
                                    new_V[j_global*5 + k_global] += coeff
                                    new_W[k_global*4 + i_global] += coeff
            
            U_4.append(new_U)
            V_4.append(new_V)
            W_4.append(new_W)
    
    # Add 16 terms for 4x4x1 (C[i,4] = sum_j A[i,j] B[j,4])
    for i in range(4):
        for j in range(4):
            new_U = [0] * 16
            new_V = [0] * 20
            new_W = [0] * 20
            new_U[i*4 + j] = 1
            new_V[j*5 + 4] = 1
            new_W[4*4 + i] = 1
            U_4.append(new_U)
            V_4.append(new_V)
            W_4.append(new_W)
    
    U_final = U_4
    V_final = V_4
    W_final = W_4
    
    if not verify_decomposition(U_final, V_final, W_final):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_sch = [[0] * rank_schoolbook for _ in range(n * m)]
        V_sch = [[0] * rank_schoolbook for _ in range(m * p)]
        W_sch = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i*m+j][col] = 1
                    V_sch[j*p+k][col] = 1
                    W_sch[k*n+i][col] = 1
                    col += 1
        return {"U": U_sch, "V": V_sch, "W": W_sch}
        
    return {"U": U_final, "V": V_final, "W": W_final}

construct()
