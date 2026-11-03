import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    rank = 88
    
    U = np.zeros((n * m, rank))  # (12, 88)
    V = np.zeros((m * p, rank))  # (32, 88)
    W = np.zeros((p * n, rank))  # (24, 88)
    
    col = 0
    
    # Strassen factors for 2x2x2
    # Each factor is (a_coeffs, b_coeffs, c_coeffs)
    # where a_coeffs maps (i,j) -> coeff for A[i,j]
    #       b_coeffs maps (i,j) -> coeff for B[i,j]  
    #       c_coeffs maps (i,j) -> coeff for C[i,j]
    
    strassen_factors = [
        # M1 = (A11+A22)(B11+B22)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A21+A22)B11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A11(B12-B22)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A22(B21-B11)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A11+A12)B22
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A21-A11)(B11+B12)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A12-A22)(B21+B22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For the 2x4x8 part (rows 0,1 of A):
    # Split A columns into 2 groups of 2: A[:, 0:2] and A[:, 2:4]
    # Split B columns into 4 groups of 2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # For each (a_group, b_group) pair, we compute a 2x2x2 product
    # A_group is 2x2, B_group is 2x2 (taking rows corresponding to the 2 A-columns), C_group is 2x2
    
    # Actually, let me think about this more carefully.
    # A is 2x4, B is 4x8, C is 2x8
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # 
    # Group A columns: j in {0,1} and j in {2,3}
    # Group B columns: k in {0,1}, {2,3}, {4,5}, {6,7}
    # 
    # For A columns {0,1} and B columns {0,1}:
    #   C[i, 0:2] += A[i, 0:2] @ B[0:2, 0:2]
    # For A columns {0,1} and B columns {2,3}:
    #   C[i, 2:4] += A[i, 0:2] @ B[0:2, 2:4]
    # etc.
    #
    # So we have 2 (A groups) * 4 (B groups) = 8 subproblems, each 2x2x2
    # Each uses 7 Strassen multiplications, total 56
    
    for a_group in range(2):  # A column groups: 0 (cols 0,1), 1 (cols 2,3)
        for b_group in range(4):  # B column groups: 0 (cols 0,1), 1 (cols 2,3), etc.
            for factor in strassen_factors:
                a_coeffs, b_coeffs, c_coeffs = factor
                
                for (ai, aj), au in a_coeffs.items():
                    # A index: row ai, column a_group*2 + aj
                    u_idx = ai * 4 + (a_group * 2 + aj)
                    U[u_idx, col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    # B index: row bi (corresponding to A column a_group*2+bi), column b_group*2+bj
                    v_idx = (a_group * 2 + bi) * 8 + (b_group * 2 + bj)
                    V[v_idx, col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    # C index: row ci, column b_group*2 + cj
                    w_idx = (b_group * 2 + cj) * 3 + ci
                    W[w_idx, col] += cw
                
                col += 1
    
    # For the 1x4x8 part (row 2 of A):
    # C[2,k] = sum_{j=0}^{3} A[2,j] B[j,k]
    # This is 4*8 = 32 rank-1 terms
    for j in range(4):
        for k in range(8):
            u_idx = 2 * 4 + j  # A[2,j]
            v_idx = j * 8 + k  # B[j,k]
            w_idx = k * 3 + 2  # C[2,k] -> index k*3+2
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
