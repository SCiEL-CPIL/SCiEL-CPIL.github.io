import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # We'll build the Strassen-based rank-28 decomposition
    # A is 2x4, B is 4x4, C = A*B is 2x4
    # A = [A1 | A2], A1 = A[:,0:2], A2 = A[:,2:4]
    # B = [[B11, B12], [B21, B22]], each 2x2
    # C1 = A1*B11 + A2*B21 (cols 0,1 of C)
    # C2 = A1*B12 + A2*B22 (cols 2,3 of C)
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8 x 28
    V = [[0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0]*rank for _ in range(p*n)]  # 8 x 28
    
    # Strassen helper: given X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]
    # Returns list of (coeffs_X, coeffs_Y, result_coeffs)
    # where coeffs_X is dict {(i,j): coeff}, coeffs_Y is dict {(i,j): coeff}
    # and result_coeffs is dict {(out_i, out_k): coeff} for the 2x2 result
    
    def strassen_terms(X_idx, Y_idx, C_offset_i, C_offset_k):
        """
        X_idx: function to get (i,j) index in A for X element (xi, xj)
        Y_idx: function to get (j,k) index in B for Y element (yj, yk)
        C_offset_i: offset for C row
        C_offset_k: offset for C col
        """
        # X = [[x00, x01], [x10, x11]]
        # Y = [[y00, y01], [y10, y11]]
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = x00(y01-y11)
        # M4 = x11(y10-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        # z00 = M1 + M4 - M5 + M7
        # z01 = M3 + M5
        # z10 = M2 + M4
        # z11 = M1 - M2 + M3 + M6
        
        terms = []
        
        # M1 = (x00+x11)(y00+y11)
        x_coeffs = {X_idx(0,0): 1, X_idx(1,1): 1}
        y_coeffs = {Y_idx(0,0): 1, Y_idx(1,1): 1}
        result = {(0,0): 1, (1,1): 1}  # z00 += 1, z11 += 1
        terms.append((x_coeffs, y_coeffs, result))
        
        # M2 = (x10+x11)y00
        x_coeffs = {X_idx(1,0): 1, X_idx(1,1): 1}
        y_coeffs = {Y_idx(0,0): 1}
        result = {(1,0): 1, (1,1): -1}  # z10 += 1, z11 -= 1
        terms.append((x_coeffs, y_coeffs, result))
        
        # M3 = x00(y01-y11)
        x_coeffs = {X_idx(0,0): 1}
        y_coeffs = {Y_idx(0,1): 1, Y_idx(1,1): -1}
        result = {(0,1): 1, (1,1): 1}  # z01 += 1, z11 += 1
        terms.append((x_coeffs, y_coeffs, result))
        
        # M4 = x11(y10-y00)
        x_coeffs = {X_idx(1,1): 1}
        y_coeffs = {Y_idx(1,0): 1, Y_idx(0,0): -1}
        result = {(0,0): 1, (1,0): 1}  # z00 += 1, z10 += 1
        terms.append((x_coeffs, y_coeffs, result))
        
        # M5 = (x00+x01)y11
        x_coeffs = {X_idx(0,0): 1, X_idx(0,1): 1}
        y_coeffs = {Y_idx(1,1): 1}
        result = {(0,0): -1, (0,1): 1}  # z00 -= 1, z01 += 1
        terms.append((x_coeffs, y_coeffs, result))
        
        # M6 = (x10-x00)(y00+y01)
        x_coeffs = {X_idx(1,0): 1, X_idx(0,0): -1}
        y_coeffs = {Y_idx(0,0): 1, Y_idx(0,1): 1}
        result = {(1,1): 1}  # z11 += 1
        terms.append((x_coeffs, y_coeffs, result))
        
        # M7 = (x01-x11)(y10+y11)
        x_coeffs = {X_idx(0,1): 1, X_idx(1,1): -1}
        y_coeffs = {Y_idx(1,0): 1, Y_idx(1,1): 1}
        result = {(0,0): 1}  # z00 += 1
        terms.append((x_coeffs, y_coeffs, result))
        
        return terms
    
    # Define index functions
    # A1 = A[:, 0:2]: X_idx(xi, xj) -> (xi, xj) in A
    # A2 = A[:, 2:4]: X_idx(xi, xj) -> (xi, xj+2) in A
    # B11 = B[0:2, 0:2]: Y_idx(yj, yk) -> (yj, yk) in B
    # B12 = B[0:2, 2:4]: Y_idx(yj, yk) -> (yj, yk+2) in B
    # B21 = B[2:4, 0:2]: Y_idx(yj, yk) -> (yj+2, yk) in B
    # B22 = B[2:4, 2:4]: Y_idx(yj, yk) -> (yj+2, yk+2) in B
    
    col = 0
    
    # C1 = A1*B11 + A2*B21, output at C[:, 0:2]
    # A1*B11
    for x_coeffs, y_coeffs, result in strassen_terms(
        lambda xi, xj: (xi, xj),  # A1
        lambda yj, yk: (yj, yk),  # B11
        0, 0  # C offset
    ):
        for (ai, aj), cv in x_coeffs.items():
            U[ai*m + aj][col] += cv
        for (bj, bk), cv in y_coeffs.items():
            V[bj*p + bk][col] += cv
        for (ci, ck), cv in result.items():
            # Output position: i = ci, k = ck (since C1 is at cols 0,1)
            W[ck*n + ci][col] += cv
        col += 1
    
    # A2*B21
    for x_coeffs, y_coeffs, result in strassen_terms(
        lambda xi, xj: (xi, xj+2),  # A2
        lambda yj, yk: (yj+2, yk),  # B21
        0, 0  # C offset
    ):
        for (ai, aj), cv in x_coeffs.items():
            U[ai*m + aj][col] += cv
        for (bj, bk), cv in y_coeffs.items():
            V[bj*p + bk][col] += cv
        for (ci, ck), cv in result.items():
            W[ck*n + ci][col] += cv
        col += 1
    
    # C2 = A1*B12 + A2*B22, output at C[:, 2:4]
    # A1*B12
    for x_coeffs, y_coeffs, result in strassen_terms(
        lambda xi, xj: (xi, xj),  # A1
        lambda yj, yk: (yj, yk+2),  # B12
        0, 2  # C col offset
    ):
        for (ai, aj), cv in x_coeffs.items():
            U[ai*m + aj][col] += cv
        for (bj, bk), cv in y_coeffs.items():
            V[bj*p + bk][col] += cv
        for (ci, ck), cv in result.items():
            # Output position: i = ci, k = ck + 2
            W[(ck+2)*n + ci][col] += cv
        col += 1
    
    # A2*B22
    for x_coeffs, y_coeffs, result in strassen_terms(
        lambda xi, xj: (xi, xj+2),  # A2
        lambda yj, yk: (yj+2, yk+2),  # B22
        0, 2  # C col offset
    ):
        for (ai, aj), cv in x_coeffs.items():
            U[ai*m + aj][col] += cv
        for (bj, bk), cv in y_coeffs.items():
            V[bj*p + bk][col] += cv
        for (ci, ck), cv in result.items():
            W[(ck+2)*n + ci][col] += cv
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
