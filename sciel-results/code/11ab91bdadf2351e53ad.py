import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 18
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 18)
    # V: (m*p, rank) = (10, 18)
    # W: (p*n, rank) = (10, 18)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx: list of (row, coeff)
    # v_idx: list of (row, coeff)
    # w_idx: list of (row, coeff)
    def add_term(col, u_coeffs, v_coeffs, w_coeffs):
        for row, coeff in u_coeffs:
            U[row][col] += coeff
        for row, coeff in v_coeffs:
            V[row][col] += coeff
        for row, coeff in w_coeffs:
            W[row][col] += coeff
            
    # Indices mapping:
    # U row for A[i,j] is i*m + j = 2i + j
    # V row for B[j,k] is j*p + k = 2j + k
    # W row for C[i,k] is k*n + i = 2k + i
    
    # Block 1: k in {0, 1} using Strassen
    # A indices: A00(0), A01(1), A10(2), A11(3)
    # B indices: B00(0), B01(1), B10(2), B11(3)
    # C indices: C00(0), C10(1), C01(2), C11(3)
    
    # Col 0: M1 = (A00+A11)(B00+B11)
    add_term(0, [(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)])
    # Col 1: M2 = (A10+A11)B00
    add_term(1, [(2,1), (3,1)], [(0,1)], [(1,1), (3,-1)])
    # Col 2: M3 = A00(B10-B00)
    add_term(2, [(0,1)], [(2,1), (0,-1)], [(2,1), (3,1)])
    # Col 3: M4 = A11(B01-B11)
    add_term(3, [(3,1)], [(1,1), (3,-1)], [(0,1), (1,1)])
    # Col 4: M5 = (A00+A01)B11
    add_term(4, [(0,1), (1,1)], [(3,1)], [(0,-1), (2,1)])
    # Col 5: M6 = (A10-A00)(B00+B01)
    add_term(5, [(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)])
    # Col 6: M7 = (A01-A11)(B10+B11)
    add_term(6, [(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
    
    # Block 2: k in {2, 3} using Strassen
    # B indices: B02(2), B03(3), B12(4), B13(5)
    # C indices: C02(4), C12(5), C03(6), C13(7)
    
    # Col 7: M1
    add_term(7, [(0,1), (3,1)], [(2,1), (5,1)], [(4,1), (7,1)])
    # Col 8: M2
    add_term(8, [(2,1), (3,1)], [(2,1)], [(5,1), (7,-1)])
    # Col 9: M3
    add_term(9, [(0,1)], [(4,1), (2,-1)], [(6,1), (7,1)])
    # Col 10: M4
    add_term(10, [(3,1)], [(3,1), (5,-1)], [(4,1), (5,1)])
    # Col 11: M5
    add_term(11, [(0,1), (1,1)], [(5,1)], [(4,-1), (6,1)])
    # Col 12: M6
    add_term(12, [(2,1), (0,-1)], [(2,1), (3,1)], [(7,1)])
    # Col 13: M7
    add_term(13, [(1,1), (3,-1)], [(4,1), (5,1)], [(4,1)])
    
    # Block 3: k in {4} using Schoolbook
    # B indices: B04(4), B14(5)
    # C indices: C04(8), C14(9)
    
    # Col 14: A00 B04
    add_term(14, [(0,1)], [(4,1)], [(8,1)])
    # Col 15: A01 B14
    add_term(15, [(1,1)], [(5,1)], [(8,1)])
    # Col 16: A10 B04
    add_term(16, [(2,1)], [(4,1)], [(9,1)])
    # Col 17: A11 B14
    add_term(17, [(3,1)], [(5,1)], [(9,1)])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
