import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    def a_idx(i, j):
        """A[i,j] row index in U"""
        return i * m + j
    
    def b_idx(i, j):
        """B[i,j] row index in V"""
        return i * p + j
    
    def c_idx(i, k):
        """C[i,k] row index in W: k*n + i"""
        return k * n + i
    
    # Define the 4 sub-problems
    # Each sub-problem: (X_elem_map, Y_elem_map, Z_elem_map)
    # X_elem_map: {'X00': (i,j), 'X01': (i,j), 'X10': (i,j), 'X11': (i,j)}
    # Y_elem_map: {'Y00': (i,j), 'Y01': (i,j), 'Y10': (i,j), 'Y11': (i,j)}
    # Z_elem_map: {'Z00': (i,k), 'Z01': (i,k), 'Z10': (i,k), 'Z11': (i,k)}
    
    sub_problems = [
        # Sub-problem 1: A_left * B_top_left
        {
            'X': {'X00': (0,0), 'X01': (0,1), 'X10': (1,0), 'X11': (1,1)},
            'Y': {'Y00': (0,0), 'Y01': (0,1), 'Y10': (1,0), 'Y11': (1,1)},
            'Z': {'Z00': (0,0), 'Z01': (0,1), 'Z10': (1,0), 'Z11': (1,1)}
        },
        # Sub-problem 2: A_left * B_top_right
        {
            'X': {'X00': (0,0), 'X01': (0,1), 'X10': (1,0), 'X11': (1,1)},
            'Y': {'Y00': (0,2), 'Y01': (0,3), 'Y10': (1,2), 'Y11': (1,3)},
            'Z': {'Z00': (0,2), 'Z01': (0,3), 'Z10': (1,2), 'Z11': (1,3)}
        },
        # Sub-problem 3: A_right * B_bottom_left
        {
            'X': {'X00': (0,2), 'X01': (0,3), 'X10': (1,2), 'X11': (1,3)},
            'Y': {'Y00': (2,0), 'Y01': (2,1), 'Y10': (3,0), 'Y11': (3,1)},
            'Z': {'Z00': (0,0), 'Z01': (0,1), 'Z10': (1,0), 'Z11': (1,1)}
        },
        # Sub-problem 4: A_right * B_bottom_right
        {
            'X': {'X00': (0,2), 'X01': (0,3), 'X10': (1,2), 'X11': (1,3)},
            'Y': {'Y00': (2,2), 'Y01': (2,3), 'Y10': (3,2), 'Y11': (3,3)},
            'Z': {'Z00': (0,2), 'Z01': (0,3), 'Z10': (1,2), 'Z11': (1,3)}
        },
    ]
    
    # Strassen terms: each is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (X_elem, coeff)
    # V_coeffs: list of (Y_elem, coeff)
    # W_coeffs: list of (Z_elem, coeff)
    
    strassen_terms = [
        # Term 1: M1 = (X00+X11)(Y00+Y11)
        {
            'U': [('X00', 1), ('X11', 1)],
            'V': [('Y00', 1), ('Y11', 1)],
            'W': [('Z00', 1), ('Z11', 1)]
        },
        # Term 2: M2 = (X10+X11)Y00
        {
            'U': [('X10', 1), ('X11', 1)],
            'V': [('Y00', 1)],
            'W': [('Z10', 1), ('Z11', -1)]
        },
        # Term 3: M3 = X00(Y10-Y11)
        {
            'U': [('X00', 1)],
            'V': [('Y10', 1), ('Y11', -1)],
            'W': [('Z01', 1), ('Z11', 1)]
        },
        # Term 4: M4 = X11(Y00-Y10)
        {
            'U': [('X11', 1)],
            'V': [('Y00', 1), ('Y10', -1)],
            'W': [('Z00', 1), ('Z10', 1)]
        },
        # Term 5: M5 = (X00+X01)Y11
        {
            'U': [('X00', 1), ('X01', 1)],
            'V': [('Y11', 1)],
            'W': [('Z00', -1), ('Z01', 1)]
        },
        # Term 6: M6 = (X10-X00)(Y00+Y01)
        {
            'U': [('X10', 1), ('X00', -1)],
            'V': [('Y00', 1), ('Y01', 1)],
            'W': [('Z11', 1)]
        },
        # Term 7: M7 = (X01-X11)(Y10+Y11)
        {
            'U': [('X01', 1), ('X11', -1)],
            'V': [('Y10', 1), ('Y11', 1)],
            'W': [('Z00', 1), ('Z11', -1)]
        },
    ]
    
    col = 0
    for sp in sub_problems:
        X_map = sp['X']
        Y_map = sp['Y']
        Z_map = sp['Z']
        
        for term in strassen_terms:
            # Fill U
            for x_elem, coeff in term['U']:
                i, j = X_map[x_elem]
                U[a_idx(i, j)][col] += coeff
            
            # Fill V
            for y_elem, coeff in term['V']:
                i, j = Y_map[y_elem]
                V[b_idx(i, j)][col] += coeff
            
            # Fill W
            for z_elem, coeff in term['W']:
                i, k = Z_map[z_elem]
                W[c_idx(i, k)][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
