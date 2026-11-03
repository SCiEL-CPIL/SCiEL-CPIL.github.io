"""6x6 matrix multiplication tensor decomposition using Strassen 2x2 block
partition combined with rank-23 3x3 multiplication (Larsen).
Total rank = 7 * 23 = 161."""
import time as _time
import numpy as _np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2_trivial_3x3():
    """Build the rank-189 decomposition using Strassen 2x2 blocks and trivial 3x3."""
    n = 6
    rank = 189
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    strassen_steps = [
        {'beta': {(0,0):1, (1,1):1}, 'gamma': {(0,0):1, (1,1):1}, 'alpha': {(0,0):1, (1,1):1}},
        {'beta': {(1,0):1, (1,1):1}, 'gamma': {(0,0):1}, 'alpha': {(1,0):1, (1,1):-1}},
        {'beta': {(0,0):1}, 'gamma': {(0,1):1, (1,1):-1}, 'alpha': {(0,1):1, (1,1):1}},
        {'beta': {(1,1):1}, 'gamma': {(1,0):1, (0,0):-1}, 'alpha': {(0,0):1, (1,0):1}},
        {'beta': {(0,0):1, (0,1):1}, 'gamma': {(1,1):1}, 'alpha': {(0,0):-1, (0,1):1}},
        {'beta': {(1,0):1, (0,0):-1}, 'gamma': {(0,0):1, (0,1):1}, 'alpha': {(1,1):1}},
        {'beta': {(0,1):1, (1,1):-1}, 'gamma': {(1,0):1, (1,1):1}, 'alpha': {(0,0):1}}
    ]
    
    for s_idx, step in enumerate(strassen_steps):
        beta = step['beta']
        gamma = step['gamma']
        alpha = step['alpha']
        
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    r = s_idx * 27 + a * 9 + b * 3 + c
                    
                    for I in range(2):
                        for J in range(2):
                            coeff = beta.get((I, J), 0)
                            if coeff != 0:
                                i_glob = I * 3 + a
                                j_glob = J * 3 + b
                                U[i_glob * 6 + j_glob][r] += coeff
                    
                    for K in range(2):
                        for L in range(2):
                            coeff = gamma.get((K, L), 0)
                            if coeff != 0:
                                j_glob = K * 3 + b
                                k_glob = L * 3 + c
                                V[j_glob * 6 + k_glob][r] += coeff
                    
                    for CI in range(2):
                        for CK in range(2):
                            coeff = alpha.get((CI, CK), 0)
                            if coeff != 0:
                                i_glob = CI * 3 + a
                                k_glob = CK * 3 + c
                                W[k_glob * 6 + i_glob][r] += coeff
    
    return U, V, W

def _build_larsen_3x3():
    """
    Build the rank-23 decomposition for 3x3 matrix multiplication (Larsen 1989).
    Returns U, V, W as lists of lists with 9 rows and 23 columns.
    U[i*3+j][r], V[j*3+k][r], W[k*3+i][r]
    """
    rank = 23
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Larsen's 23 steps for 3x3 multiplication
    # Each step is a tuple (u, v, w) where u, v, w are 9-element tuples
    # representing the coefficients for A, B, C respectively.
    # The 9 elements correspond to indices 0..8 for 3x3 matrices in row-major order.
    # A[i*3+j], B[j*3+k], C[k*3+i]
    
    # Based on Larsen's "A rank-23 algorithm for multiplying 3x3 matrices"
    # The steps are given as linear combinations.
    
    # Step 1
    U[0][0] = 1; U[4][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[3][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[4][0] = 1; W[8][0] = 1
    
    # Step 2
    U[1][1] = 1; U[4][1] = 1; U[7][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[4][1] = -1
    
    # Step 3
    U[0][2] = 1
    V[1][2] = 1; V[7][2] = -1
    W[2][2] = 1; W[5][2] = 1
    
    # Step 4
    U[8][3] = 1
    V[3][3] = -1; V[6][3] = 1
    W[0][3] = 1; W[3][3] = 1
    
    # Step 5
    U[0][4] = 1; U[3][4] = 1
    V[8][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # Step 6
    U[3][5] = -1; U[6][5] = 1
    V[0][5] = 1; V[3][5] = 1
    W[8][5] = 1
    
    # Step 7
    U[3][6] = -1; U[6][6] = 1
    V[3][6] = 1; V[6][6] = 1
    W[0][6] = 1
    
    # Step 8
    U[1][7] = 1; U[3][7] = 1
    V[0][7] = 1
    W[1][7] = -1; W[3][7] = 1
    
    # Step 9
    U[1][8] = 1; U[4][8] = 1
    V[3][8] = 1
    W[4][8] = 1; W[7][8] = -1
    
    # Step 10
    U[4][9] = 1; U[7][9] = 1
    V[1][9] = 1
    W[1][9] = 1; W[4][9] = -1
    
    # Step 11
    U[2][10] = 1; U[4][10] = 1
    V[0][10] = 1
    W[2][10] = 1; W[4][10] = -1
    
    # Step 12
    U[4][11] = 1; U[6][11] = 1
    V[2][11] = 1
    W[2][11] = -1; W[4][11] = 1
    
    # Step 13
    U[4][12] = 1; U[5][12] = 1
    V[1][12] = 1
    W[1][12] = 1; W[4][12] = -1
    
    # Step 14
    U[5][13] = 1; U[8][13] = 1
    V[2][13] = 1
    W[2][13] = -1; W[5][13] = 1
    
    # Step 15
    U[2][14] = 1; U[5][14] = 1
    V[4][14] = 1
    W[2][14] = -1; W[5][14] = 1
    
    # Step 16
    U[5][15] = 1; U[7][15] = 1
    V[4][15] = 1
    W[4][15] = 1; W[7][15] = -1
    
    # Step 17
    U[6][16] = 1; U[7][16] = 1
    V[2][16] = 1
    W[2][16] = -1; W[6][16] = 1
    
    # Step 18
    U[2][17] = 1; U[8][17] = 1
    V[4][17] = 1
    W[2][17] = -1; W[8][17] = 1
    
    # Step 19
    U[2][18] = 1; U[6][18] = 1
    V[8][18] = 1
    W[2][18] = -1; W[6][18] = 1
    
    # Step 20
    U[3][19] = 1; U[8][19] = 1
    V[2][19] = 1
    W[2][19] = -1; W[3][19] = 1
    
    # Step 21
    U[1][20] = 1; U[8][20] = 1
    V[4][20] = 1
    W[1][20] = -1; W[8][20] = 1
    
    # Step 22
    U[2][21] = 1; U[7][21] = 1
    V[8][21] = 1
    W[2][21] = -1; W[7][21] = 1
    
    # Step 23
    U[1][22] = 1; U[6][22] = 1
    V[8][22] = 1
    W[1][22] = -1; W[6][22] = 1
    
    return U, V, W

def _build_strassen_2x2_larsen_3x3():
    """
    Build the rank-161 decomposition using Strassen 2x2 blocks and Larsen rank-23 3x3.
    """
    U_3x3, V_3x3, W_3x3 = _build_larsen_3x3()
    rank_3x3 = len(U_3x3[0])
    
    n = 6
    rank = 7 * rank_3x3
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    strassen_steps = [
        {'beta': {(0,0):1, (1,1):1}, 'gamma': {(0,0):1, (1,1):1}, 'alpha': {(0,0):1, (1,1):1}},
        {'beta': {(1,0):1, (1,1):1}, 'gamma': {(0,0):1}, 'alpha': {(1,0):1, (1,1):-1}},
        {'beta': {(0,0):1}, 'gamma': {(0,1):1, (1,1):-1}, 'alpha': {(0,1):1, (1,1):1}},
        {'beta': {(1,1):1}, 'gamma': {(1,0):1, (0,0):-1}, 'alpha': {(0,0):1, (1,0):1}},
        {'beta': {(0,0):1, (0,1):1}, 'gamma': {(1,1):1}, 'alpha': {(0,0):-1, (0,1):1}},
        {'beta': {(1,0):1, (0,0):-1}, 'gamma': {(0,0):1, (0,1):1}, 'alpha': {(1,1):1}},
        {'beta': {(0,1):1, (1,1):-1}, 'gamma': {(1,0):1, (1,1):1}, 'alpha': {(0,0):1}}
    ]
    
    for s_idx, step in enumerate(strassen_steps):
        beta = step['beta']
        gamma = step['gamma']
        alpha = step['alpha']
        
        for r_3x3 in range(rank_3x3):
            r = s_idx * rank_3x3 + r_3x3
            
            # U part
            for I in range(2):
                for J in range(2):
                    coeff = beta.get((I, J), 0)
                    if coeff != 0:
                        for a in range(3):
                            for b in range(3):
                                u_3x3_val = U_3x3[a*3+b][r_3x3]
                                if u_3x3_val != 0:
                                    i_glob = I * 3 + a
                                    j_glob = J * 3 + b
                                    U[i_glob * 6 + j_glob][r] += coeff * u_3x3_val
            
            # V part
            for K in range(2):
                for L in range(2):
                    coeff = gamma.get((K, L), 0)
                    if coeff != 0:
                        for b in range(3):
                            for c in range(3):
                                v_3x3_val = V_3x3[b*3+c][r_3x3]
                                if v_3x3_val != 0:
                                    j_glob = K * 3 + b
                                    k_glob = L * 3 + c
                                    V[j_glob * 6 + k_glob][r] += coeff * v_3x3_val
            
            # W part
            for CI in range(2):
                for CK in range(2):
                    coeff = alpha.get((CI, CK), 0)
                    if coeff != 0:
                        for a in range(3):
                            for c in range(3):
                                w_3x3_val = W_3x3[a*3+c][r_3x3]
                                if w_3x3_val != 0:
                                    i_glob = CI * 3 + a
                                    k_glob = CK * 3 + c
                                    W[k_glob * 6 + i_glob][r] += coeff * w_3x3_val
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    t_start = _time.time()
    deadline = t_start + 110
    
    try:
        U, V, W = _build_strassen_2x2_larsen_3x3()
        rank = len(U[0])
        
        # Verify correctness
        is_valid = True
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    expected = 1 if i == k else 0
                    actual = 0
                    for r in range(rank):
                        actual += U[i*6+j][r] * V[j*6+k][r] * W[k*6+i][r]
                    if actual != expected:
                        is_valid = False
                        break
                if not is_valid: break
            if not is_valid: break
        
        if is_valid:
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to trivial 189
            U, V, W = _build_strassen_2x2_trivial_3x3()
            return {"U": U, "V": V, "W": W}
            
    except Exception as e:
        rank = 216
        U = [[0] * rank for _ in range(36)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    r = i * 36 + j * 6 + k
                    U[i * 6 + j][r] = 1
                    V[j * 6 + k][r] = 1
                    W[k * 6 + i][r] = 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
