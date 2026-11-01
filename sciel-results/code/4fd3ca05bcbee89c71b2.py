DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    # Strassen-based decomposition
    def strassen_decomposition():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for 2x2x2
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # p1 = (a00+a11)*(b00+b11)
        # p2 = (a10+a11)*b00
        # p3 = a00*(b10-b11)
        # p4 = a11*(b00-b01)
        # p5 = (a00+a01)*b11
        # p6 = (a10-a00)*(b01+b11)
        # p7 = (a01-a11)*(b00+b10)
        # c00 = p1 + p4 - p5 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p6
        
        # Coefficients for A (U), B (V), C (W)
        # U[row, col] where row = i*m + j
        # V[row, col] where row = j*p + k
        # W[row, col] where row = k*n + i
        
        a_coeffs = [
            [1, 0, 0, 0],  # p1 a00
            [0, 0, 1, 1],  # p2 a10, a11 -> indices 2,3 in 0-based? No.
            # Let's map local indices 0,1,2,3 to a00, a01, a10, a11
            # a00: idx 0
            # a01: idx 1
            # a10: idx 2
            # a11: idx 3
            
            # p1: a00+a11 -> [1, 0, 0, 1]
            [1, 0, 0, 1],
            # p2: a10+a11 -> [0, 0, 1, 1]
            [0, 0, 1, 1],
            # p3: a00 -> [1, 0, 0, 0]
            [1, 0, 0, 0],
            # p4: a11 -> [0, 0, 0, 1]
            [0, 0, 0, 1],
            # p5: a00+a01 -> [1, 1, 0, 0]
            [1, 1, 0, 0],
            # p6: a10-a00 -> [-1, 0, 1, 0]
            [-1, 0, 1, 0],
            # p7: a01-a11 -> [0, 1, 0, -1]
            [0, 1, 0, -1]
        ]
        
        b_coeffs = [
            # p1: b00+b11 -> [1, 0, 0, 1] (b00, b01, b10, b11)
            [1, 0, 0, 1],
            # p2: b00 -> [1, 0, 0, 0]
            [1, 0, 0, 0],
            # p3: b10-b11 -> [0, 0, 1, -1]
            [0, 0, 1, -1],
            # p4: b00-b01 -> [1, -1, 0, 0]
            [1, -1, 0, 0],
            # p5: b11 -> [0, 0, 0, 1]
            [0, 0, 0, 1],
            # p6: b01+b11 -> [0, 1, 0, 1]
            [0, 1, 0, 1],
            # p7: b00+b10 -> [1, 0, 1, 0]
            [1, 0, 1, 0]
        ]
        
        w_coeffs = [
            # c00: p1 + p4 - p5 - p7
            # c01: p3 + p5
            # c10: p2 + p4
            # c11: p1 - p2 + p3 + p6
            # We need W[k*n+i, r] = coeff of P_r in C[i,k]
            # C[i,k] local indices: c00 (i=0,k=0), c01 (i=0,k=1), c10 (i=1,k=0), c11 (i=1,k=1)
            # For each local output (i_loc, k_loc), we have a vector of 7 coeffs.
            
            # c00 (i=0, k=0)
            [1, 0, 0, 1, -1, 0, -1],
            # c01 (i=0, k=1)
            [0, 0, 1, 0, 1, 0, 0],
            # c10 (i=1, k=0)
            [0, 1, 0, 1, 0, 0, 0],
            # c11 (i=1, k=1)
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        # 6 blocks
        # Block b corresponds to j_block in {0,1} and k_block in {0,1,2}
        # j_block 0 -> j in {0,1}
        # j_block 1 -> j in {2,3}
        # k_block 0 -> k in {0,1}
        # k_block 1 -> k in {2,3}
        # k_block 2 -> k in {4,5}
        
        for jb in range(2):
            for kb in range(3):
                base_col = (jb * 3 + kb) * 7
                
                # Local j indices: 0, 1 -> Global j0, j1
                j0 = jb * 2
                j1 = jb * 2 + 1
                
                # Local k indices: 0, 1 -> Global k0, k1
                k0 = kb * 2
                k1 = kb * 2 + 1
                
                # Iterate over 7 Strassen steps
                for s in range(7):
                    col = base_col + s
                    
                    # Update U (A)
                    # Local A indices: 00, 01, 10, 11
                    # Global i in {0,1}, j in {j0, j1}
                    # Local a00 -> i=0, j=j0
                    # Local a01 -> i=0, j=j1
                    # Local a10 -> i=1, j=j0
                    # Local a11 -> i=1, j=j1
                    
                    for i_loc in range(2):
                        for j_loc in range(2):
                            global_i = i_loc
                            global_j = j0 + j_loc
                            coeff = a_coeffs[s][i_loc * 2 + j_loc]
                            if coeff != 0:
                                U[global_i * m + global_j][col] += coeff
                                
                    # Update V (B)
                    # Local B indices: 00, 01, 10, 11
                    # Global j in {j0, j1}, k in {k0, k1}
                    # Local b00 -> j=j0, k=k0
                    # Local b01 -> j=j0, k=k1
                    # Local b10 -> j=j1, k=k0
                    # Local b11 -> j=j1, k=k1
                    
                    for j_loc in range(2):
                        for k_loc in range(2):
                            global_j = j0 + j_loc
                            global_k = k0 + k_loc
                            coeff = b_coeffs[s][j_loc * 2 + k_loc]
                            if coeff != 0:
                                V[global_j * p + global_k][col] += coeff
                                
                    # Update W (C)
                    # Local C indices: c00, c01, c10, c11
                    # Global i in {0,1}, k in {k0, k1}
                    # Local c00 -> i=0, k=k0
                    # Local c01 -> i=0, k=k1
                    # Local c10 -> i=1, k=k0
                    # Local c11 -> i=1, k=k1
                    
                    for i_loc in range(2):
                        for k_loc in range(2):
                            global_i = i_loc
                            global_k = k0 + k_loc
                            coeff = w_coeffs[i_loc * 2 + k_loc][s]
                            if coeff != 0:
                                W[global_k * n + global_i][col] += coeff
                                
        return {"U": U, "V": V, "W": W}, rank

    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try Strassen first
    result, rank = strassen_decomposition()
    if check_decomposition(result["U"], result["V"], result["W"]):
        return result

    # Fallback to schoolbook
    result, rank = schoolbook()
    return result

# EVOLVE-BLOCK-END
