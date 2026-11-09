import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We are doing 6x6 matrix multiplication.
    # Block size is 2x2.
    # Number of blocks per dimension is 3.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for X, Y (2x2 matrices)
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X01-X11)(Y01+Y11)
    
    # Reconstruction:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # X is A_pc, Y is B_cr
    # X indices: (2p+u, 2c+v)
    # Y indices: (2c+u, 2r+v)
    
    # Define Strassen steps as (coeffs for X, coeffs for Y, coeffs for Z)
    # X coeffs: [c00, c01, c10, c11] corresponding to X00, X01, X10, X11
    # Y coeffs: [c00, c01, c10, c11] corresponding to Y00, Y01, Y10, Y11
    # Z coeffs: [c00, c01, c10, c11] corresponding to Z00, Z01, Z10, Z11
    
    strassen_steps = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # Z00 += M1
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # Z10 += M2
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]), # Z01 += M3
        # M4
        ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 1, 0]), # Z00 += M4, Z10 += M4
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # Z00 -= M5, Z01 += M5
        # M6
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 0, 1]), # Z11 += M6
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 1])  # Z00 += M7, Z11 += M7
    ]
    
    for p_idx in range(3):
        for c_idx in range(3):
            for r_idx in range(3):
                for s_idx in range(7):
                    x_coeffs, y_coeffs, z_coeffs = strassen_steps[s_idx]
                    
                    # Map X coefficients to U
                    # X00 -> A[2*p_idx + 0, 2*c_idx + 0]
                    # X01 -> A[2*p_idx + 0, 2*c_idx + 1]
                    # X10 -> A[2*p_idx + 1, 2*c_idx + 0]
                    # X11 -> A[2*p_idx + 1, 2*c_idx + 1]
                    x_indices = [
                        (2*p_idx + 0) * m + (2*c_idx + 0),
                        (2*p_idx + 0) * m + (2*c_idx + 1),
                        (2*p_idx + 1) * m + (2*c_idx + 0),
                        (2*p_idx + 1) * m + (2*c_idx + 1)
                    ]
                    for k in range(4):
                        if x_coeffs[k] != 0:
                            U[x_indices[k]][col] = x_coeffs[k]
                            
                    # Map Y coefficients to V
                    # Y00 -> B[2*c_idx + 0, 2*r_idx + 0]
                    # Y01 -> B[2*c_idx + 0, 2*r_idx + 1]
                    # Y10 -> B[2*c_idx + 1, 2*r_idx + 0]
                    # Y11 -> B[2*c_idx + 1, 2*r_idx + 1]
                    y_indices = [
                        (2*c_idx + 0) * p + (2*r_idx + 0),
                        (2*c_idx + 0) * p + (2*r_idx + 1),
                        (2*c_idx + 1) * p + (2*r_idx + 0),
                        (2*c_idx + 1) * p + (2*r_idx + 1)
                    ]
                    for k in range(4):
                        if y_coeffs[k] != 0:
                            V[y_indices[k]][col] = y_coeffs[k]
                            
                    # Map Z coefficients to W
                    # Z00 -> C[2*p_idx + 0, 2*r_idx + 0]
                    # Z01 -> C[2*p_idx + 0, 2*r_idx + 1]
                    # Z10 -> C[2*p_idx + 1, 2*r_idx + 0]
                    # Z11 -> C[2*p_idx + 1, 2*r_idx + 1]
                    # Note: W index is k*n + i? 
                    # Task says: W[k*n+i, r]. 
                    # In standard matrix mult C = AB, C[i,j] = sum A[i,k]B[k,j].
                    # Tensor T[i,j,k] = delta_{i,k} delta_{j,l}? No.
                    # The tensor for matrix mult is usually defined such that 
                    # sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k].
                    # Here indices are:
                    # U index: i, j (row, col of A) -> i*m+j
                    # V index: j, k (row, col of B) -> j*p+k
                    # W index: k, i (row, col of C) -> k*n+i
                    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
                    # This means W index is (k, i) where k is col of A/row of B? 
                    # Let's check the schoolbook example in the prompt:
                    # U[i*m+j][column] = 1
                    # V[j*p+k][column] = 1
                    # W[k*n+i][column] = 1
                    # This corresponds to C[k,i] = sum_j A[i,j] B[j,k].
                    # So C row is k, col is i.
                    # Standard C = AB has C[row, col] = sum A[row, k] B[k, col].
                    # So if we identify C[k,i] with standard C[standard_row, standard_col],
                    # then standard_row = k, standard_col = i.
                    # So W index (k, i) corresponds to C[k, i].
                    
                    z_indices = [
                        (2*p_idx + 0) * n + (2*r_idx + 0), # C[2p, 2r] -> k=2p, i=2r? 
                        # Wait. W[k*n+i].
                        # In the schoolbook: W[k*n+i].
                        # The sum is over j. U[i,j] V[j,k] -> C[k,i].
                        # So the result matrix is indexed by (k, i).
                        # My Strassen derivation was for C[p,r] block, which is C[2p+u, 2r+v].
                        # So the result index is (row=2p+u, col=2r+v).
                        # This matches W[k*n+i] if k=row, i=col.
                        # So z_indices should be (2*p_idx + u) * n + (2*r_idx + v).
                        (2*p_idx + 0) * n + (2*r_idx + 0),
                        (2*p_idx + 0) * n + (2*r_idx + 1),
                        (2*p_idx + 1) * n + (2*r_idx + 0),
                        (2*p_idx + 1) * n + (2*r_idx + 1)
                    ]
                    for k in range(4):
                        if z_coeffs[k] != 0:
                            W[z_indices[k]][col] = z_coeffs[k]
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
