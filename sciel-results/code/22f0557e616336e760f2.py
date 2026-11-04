"""Rewritten program for 4x4 * 4x5 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 4, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check that U @ V^T == T == W @ (some contraction) for the matmul tensor.
    The tensor T[i,j,k] = delta[i, k] for (i in [n], j in [m], k in [p]).
    Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4 * 4x4 multiplication (rank 49)
    # Part 2: 4x4 * 4x1 multiplication (rank 1)
    # Total rank: 50
    
    rank_total = 50
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # --- Part 1: 4x4 * 4x4 Strassen (Rank 49) ---
    # Indices for 4x4x4: i, j, k in 0..3
    # i = 2*I + i', j = 2*J + j', k = 2*K + k'
    # I, J, K in 0..1
    # i', j', k' in 0..1
    
    # Block coefficients for 7 Strassen terms
    # U_block[I, J, m]
    U_block = [
        [0]*7 for _ in range(2)
    ]
    # Flatten U_block to [I, J, m]
    U_block[0][0] = 1 # M1: A00
    U_block[1][1] = 1 # M1: A11
    
    U_block[1][0] = 1 # M2: A10
    U_block[1][1] = 1 # M2: A11
    
    U_block[0][0] = 1 # M3: A00
    
    U_block[1][1] = 1 # M4: A11
    
    U_block[0][0] = 1 # M5: A00
    U_block[0][1] = 1 # M5: A01
    
    U_block[1][0] = 1 # M6: A10
    U_block[0][0] = -1 # M6: A00
    
    U_block[0][1] = 1 # M7: A01
    U_block[1][1] = -1 # M7: A11
    
    # This simple list approach is error prone. Let's hardcode the 49 terms.
    # We will generate the 49 terms by iterating m in 0..6 and c in 0..6.
    
    # Define U2, V2, W2 for 2x2x2 Strassen
    # U2[i', j', c]
    U2 = [[0]*7 for _ in range(4)]
    U2[0][0] = 1; U2[3][0] = 1 # M1: A00, A11 (i'j' 00, 11)
    U2[2][1] = 1; U2[3][1] = 1 # M2: A10, A11 (i'j' 10, 11)
    U2[0][2] = 1 # M3: A00
    U2[3][3] = 1 # M4: A11
    U2[0][4] = 1; U2[1][4] = 1 # M5: A00, A01
    U2[2][5] = 1; U2[0][5] = -1 # M6: A10, A00
    U2[1][6] = 1; U2[3][6] = -1 # M7: A01, A11
    
    # V2[j', k', c]
    V2 = [[0]*7 for _ in range(4)]
    V2[0][0] = 1; V2[1][0] = 1; V2[2][0] = 1; V2[3][0] = 1 # M1: B00, B01, B10, B11
    V2[0][1] = 1 # M2: B00
    V2[1][2] = 1; V2[3][2] = -1 # M3: B01, B11
    V2[2][3] = 1; V2[0][3] = -1 # M4: B10, B00
    V2[3][4] = 1 # M5: B11
    V2[0][5] = 1; V2[1][5] = 1 # M6: B00, B01
    V2[2][6] = 1; V2[3][6] = 1 # M7: B10, B11
    
    # W2[k', i', c]
    W2 = [[0]*7 for _ in range(4)]
    W2[0][0] = 1; W2[3][0] = 1 # M1: C00, C11
    W2[1][1] = 1; W2[3][1] = -1 # M2: C10, C11
    W2[2][2] = 1; W2[3][2] = 1 # M3: C01, C11
    W2[0][3] = 1; W2[1][3] = 1 # M4: C00, C10
    W2[0][4] = -1; W2[2][4] = 1 # M5: C00, C01
    W2[3][5] = 1 # M6: C11
    W2[0][6] = 1 # M7: C00
    
    # Block coefficients for 7 super-terms
    # U_b[I, J, m]
    U_b = [[0]*7 for _ in range(2)] # U_b[I][J] is list of 7? No, U_b[I][J][m]
    # Let's use 3D lists or just hardcode the logic.
    
    # We'll iterate m (0..6) and c (0..6)
    for m in range(7):
        for c in range(7):
            col = m * 7 + c
            if U2[0][c] == 0 and U2[1][c] == 0 and U2[2][c] == 0 and U2[3][c] == 0:
                continue
            if V2[0][c] == 0 and V2[1][c] == 0 and V2[2][c] == 0 and V2[3][c] == 0:
                continue
            if W2[0][c] == 0 and W2[1][c] == 0 and W2[2][c] == 0 and W2[3][c] == 0:
                continue

            # Determine which blocks are involved in term m
            # Based on Strassen definitions
            if m == 0: # M1
                U_b = [(0,0,1), (1,1,1)]
                V_b = [(0,0,1), (1,1,1)]
                W_b = [(0,0,1), (1,1,1)]
            elif m == 1: # M2
                U_b = [(1,0,1), (1,1,1)]
                V_b = [(0,0,1)]
                W_b = [(1,0,1), (1,1,-1)]
            elif m == 2: # M3
                U_b = [(0,0,1)]
                V_b = [(0,1,1), (1,1,-1)]
                W_b = [(0,1,1), (1,1,1)]
            elif m == 3: # M4
                U_b = [(1,1,1)]
                V_b = [(1,0,1), (0,0,-1)]
                W_b = [(0,0,1), (1,0,1)]
            elif m == 4: # M5
                U_b = [(0,0,1), (0,1,1)]
                V_b = [(1,1,1)]
                W_b = [(0,0,-1), (0,1,1)]
            elif m == 5: # M6
                U_b = [(1,0,1), (0,0,-1)]
                V_b = [(0,0,1), (0,1,1)]
                W_b = [(1,1,1)]
            elif m == 6: # M7
                U_b = [(0,1,1), (1,1,-1)]
                V_b = [(1,0,1), (1,1,1)]
                W_b = [(0,0,1)]
            
            # Fill U, V, W
            # U[i, j, col] = sum over blocks
            # i = 2*I + i', j = 2*J + j'
            for I, J, coeff in U_b:
                for i_prime in range(2):
                    for j_prime in range(2):
                        u_val = U2[i_prime * 2 + j_prime][c]
                        if u_val != 0:
                            i = 2 * I + i_prime
                            j = 2 * J + j_prime
                            U[i * 4 + j][col] = coeff * u_val
            
            for J, K, coeff in V_b:
                for j_prime in range(2):
                    for k_prime in range(2):
                        v_val = V2[j_prime * 2 + k_prime][c]
                        if v_val != 0:
                            j = 2 * J + j_prime
                            k = 2 * K + k_prime
                            V[j * 5 + k][col] = coeff * v_val
                            # Note: V index is j*p + k. p=5.
                            # But in Part 1, we are multiplying 4x4 by 4x4.
                            # So k is 0..3.
                            # The V matrix for the full problem is (m*p, rank) = (4*5, 50) = (20, 50).
                            # In Part 1, we only use k=0..3.
            
            for K, I, coeff in W_b:
                for k_prime in range(2):
                    for i_prime in range(2):
                        w_val = W2[k_prime * 2 + i_prime][c]
                        if w_val != 0:
                            k = 2 * K + k_prime
                            i = 2 * I + i_prime
                            W[k * 4 + i][col] = coeff * w_val
                            # W index is k*n + i. n=4.
    
    # --- Part 2: 4x4 * 4x1 multiplication (Rank 1) ---
    # A (4x4) * v (4x1) = u (4x1)
    # T[i, j, 4] = 1 (k=4 is the 5th column)
    # Decomposition: U[i, j, 49] * V[j, 4, 49] * W[4, i, 49] = 1
    # U[i, j, 49] = 1
    # V[j, 4, 49] = 1
    # W[4, i, 49] = 1
    
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][49] = 1
            V[j * 5 + 4][49] = 1
            W[4 * 4 + i][49] = 1
            
    return {"U": U, "V": V, "W": W}
