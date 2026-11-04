import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (y00+y11)(z00+z11)
    # M2 = (y10+y11)z00
    # M3 = y00(z01-z11)
    # M4 = y11(z10-z00)
    # M5 = (y00+y01)z11
    # M6 = (y10-y00)(z00+z01)
    # M7 = (y01-y11)(z10+z11)
    
    # Reconstruction
    # x00 = M1 + M4 - M5 + M7
    # x01 = M3 + M5
    # x10 = M2 + M4
    # x11 = M1 - M2 + M3 + M6
    
    # We define the 7 multipliers' contributions to U, V, W
    # For each multiplier c, we add +/- 1 to specific rows in U, V and W.
    
    # Helper to add a term to U, V, W
    # u_idx: index in U (i*m+j)
    # v_idx: index in V (j*p+k)
    # w_idx: index in W (k*n+i)
    # coeff: coefficient (+1 or -1)
    # col: column index in the rank dimension
    
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # Group 1: A0 * B00 -> C00, C01 (cols 0,1)
    # A0 indices: 0,1,4,5
    # B00 indices: 0,1,4,5
    # C indices: 0,2,1,3 (x00->0, x01->2, x10->1, x11->3)
    
    # M1
    add_term(0, 0, 0, 1, 0) # y00 z00 -> x00
    add_term(0, 5, 0, 1, 0) # y00 z11 -> x00
    add_term(4, 0, 0, 1, 0) # y10 z00 -> x00 (Wait, M1 is (y00+y11)(z00+z11))
    # Let's do it systematically for each M
    
    # M1 = (y00+y11)(z00+z11)
    # Terms: y00z00, y00z11, y11z00, y11z11
    # Contributions to x00: +1, to x11: +1
    # y00=0, y11=5, z00=0, z11=5
    add_term(0, 0, 0, 1, 0)
    add_term(0, 5, 0, 1, 0)
    add_term(5, 0, 0, 1, 0)
    add_term(5, 5, 0, 1, 0)
    add_term(0, 0, 3, 1, 0)
    add_term(0, 5, 3, 1, 0)
    add_term(5, 0, 3, 1, 0)
    add_term(5, 5, 3, 1, 0)
    
    # M2 = (y10+y11)z00
    # Terms: y10z00, y11z00
    # Contributions to x10: +1, to x11: -1
    # y10=4, y11=5, z00=0
    add_term(4, 0, 1, 1, 1)
    add_term(5, 0, 1, 1, 1)
    add_term(4, 0, 3, -1, 1)
    add_term(5, 0, 3, -1, 1)
    
    # M3 = y00(z01-z11)
    # Terms: y00z01, -y00z11
    # Contributions to x01: +1, to x11: +1
    # y00=0, z01=1, z11=5
    add_term(0, 1, 2, 1, 2)
    add_term(0, 5, 2, -1, 2)
    add_term(0, 1, 3, 1, 2)
    add_term(0, 5, 3, -1, 2)
    
    # M4 = y11(z10-z00)
    # Terms: y11z10, -y11z00
    # Contributions to x00: +1, to x10: +1
    # y11=5, z10=4, z00=0
    add_term(5, 4, 0, 1, 3)
    add_term(5, 0, 0, -1, 3)
    add_term(5, 4, 1, 1, 3)
    add_term(5, 0, 1, -1, 3)
    
    # M5 = (y00+y01)z11
    # Terms: y00z11, y01z11
    # Contributions to x00: -1, to x01: +1
    # y00=0, y01=1, z11=5
    add_term(0, 5, 0, -1, 4)
    add_term(1, 5, 0, -1, 4)
    add_term(0, 5, 2, 1, 4)
    add_term(1, 5, 2, 1, 4)
    
    # M6 = (y10-y00)(z00+z01)
    # Terms: y10z00, y10z01, -y00z00, -y00z01
    # Contributions to x11: +1
    # y10=4, y00=0, z00=0, z01=1
    add_term(4, 0, 3, 1, 5)
    add_term(4, 1, 3, 1, 5)
    add_term(0, 0, 3, -1, 5)
    add_term(0, 1, 3, -1, 5)
    
    # M7 = (y01-y11)(z10+z11)
    # Terms: y01z10, y01z11, -y11z10, -y11z11
    # Contributions to x00: +1
    # y01=1, y11=5, z10=4, z11=5
    add_term(1, 4, 0, 1, 6)
    add_term(1, 5, 0, 1, 6)
    add_term(5, 4, 0, -1, 6)
    add_term(5, 5, 0, -1, 6)
    
    # Group 2: A1 * B10 -> C00, C01 (cols 0,1)
    # A1 indices: 2,3,6,7
    # B10 indices: 8,9,12,13
    # C indices: 0,2,1,3
    
    # M1
    add_term(2, 8, 0, 1, 7)
    add_term(2, 13, 0, 1, 7)
    add_term(7, 8, 0, 1, 7)
    add_term(7, 13, 0, 1, 7)
    add_term(2, 8, 3, 1, 7)
    add_term(2, 13, 3, 1, 7)
    add_term(7, 8, 3, 1, 7)
    add_term(7, 13, 3, 1, 7)
    
    # M2
    add_term(6, 8, 1, 1, 8)
    add_term(7, 8, 1, 1, 8)
    add_term(6, 8, 3, -1, 8)
    add_term(7, 8, 3, -1, 8)
    
    # M3
    add_term(2, 9, 2, 1, 9)
    add_term(2, 13, 2, -1, 9)
    add_term(2, 9, 3, 1, 9)
    add_term(2, 13, 3, -1, 9)
    
    # M4
    add_term(7, 12, 0, 1, 10)
    add_term(7, 8, 0, -1, 10)
    add_term(7, 12, 1, 1, 10)
    add_term(7, 8, 1, -1, 10)
    
    # M5
    add_term(2, 13, 0, -1, 11)
    add_term(3, 13, 0, -1, 11)
    add_term(2, 13, 2, 1, 11)
    add_term(3, 13, 2, 1, 11)
    
    # M6
    add_term(6, 8, 3, 1, 12)
    add_term(6, 9, 3, 1, 12)
    add_term(2, 8, 3, -1, 12)
    add_term(2, 9, 3, -1, 12)
    
    # M7
    add_term(3, 12, 0, 1, 13)
    add_term(3, 13, 0, 1, 13)
    add_term(7, 12, 0, -1, 13)
    add_term(7, 13, 0, -1, 13)
    
    # Group 3: A0 * B01 -> C02, C03 (cols 2,3)
    # A0 indices: 0,1,4,5
    # B01 indices: 2,3,6,7
    # C indices: 4,6,5,7 (x00->4, x01->6, x10->5, x11->7)
    
    # M1
    add_term(0, 2, 4, 1, 14)
    add_term(0, 7, 4, 1, 14)
    add_term(5, 2, 4, 1, 14)
    add_term(5, 7, 4, 1, 14)
    add_term(0, 2, 7, 1, 14)
    add_term(0, 7, 7, 1, 14)
    add_term(5, 2, 7, 1, 14)
    add_term(5, 7, 7, 1, 14)
    
    # M2
    add_term(4, 2, 5, 1, 15)
    add_term(5, 2, 5, 1, 15)
    add_term(4, 2, 7, -1, 15)
    add_term(5, 2, 7, -1, 15)
    
    # M3
    add_term(0, 3, 6, 1, 16)
    add_term(0, 7, 6, -1, 16)
    add_term(0, 3, 7, 1, 16)
    add_term(0, 7, 7, -1, 16)
    
    # M4
    add_term(5, 6, 4, 1, 17)
    add_term(5, 2, 4, -1, 17)
    add_term(5, 6, 5, 1, 17)
    add_term(5, 2, 5, -1, 17)
    
    # M5
    add_term(0, 7, 4, -1, 18)
    add_term(1, 7, 4, -1, 18)
    add_term(0, 7, 6, 1, 18)
    add_term(1, 7, 6, 1, 18)
    
    # M6
    add_term(4, 2, 7, 1, 19)
    add_term(4, 3, 7, 1, 19)
    add_term(0, 2, 7, -1, 19)
    add_term(0, 3, 7, -1, 19)
    
    # M7
    add_term(1, 6, 4, 1, 20)
    add_term(1, 7, 4, 1, 20)
    add_term(5, 6, 4, -1, 20)
    add_term(5, 7, 4, -1, 20)
    
    # Group 4: A1 * B11 -> C02, C03 (cols 2,3)
    # A1 indices: 2,3,6,7
    # B11 indices: 10,11,14,15
    # C indices: 4,6,5,7
    
    # M1
    add_term(2, 10, 4, 1, 21)
    add_term(2, 15, 4, 1, 21)
    add_term(7, 10, 4, 1, 21)
    add_term(7, 15, 4, 1, 21)
    add_term(2, 10, 7, 1, 21)
    add_term(2, 15, 7, 1, 21)
    add_term(7, 10, 7, 1, 21)
    add_term(7, 15, 7, 1, 21)
    
    # M2
    add_term(6, 10, 5, 1, 22)
    add_term(7, 10, 5, 1, 22)
    add_term(6, 10, 7, -1, 22)
    add_term(7, 10, 7, -1, 22)
    
    # M3
    add_term(2, 11, 6, 1, 23)
    add_term(2, 15, 6, -1, 23)
    add_term(2, 11, 7, 1, 23)
    add_term(2, 15, 7, -1, 23)
    
    # M4
    add_term(7, 14, 4, 1, 24)
    add_term(7, 10, 4, -1, 24)
    add_term(7, 14, 5, 1, 24)
    add_term(7, 10, 5, -1, 24)
    
    # M5
    add_term(2, 15, 4, -1, 25)
    add_term(3, 15, 4, -1, 25)
    add_term(2, 15, 6, 1, 25)
    add_term(3, 15, 6, 1, 25)
    
    # M6
    add_term(6, 10, 7, 1, 26)
    add_term(6, 11, 7, 1, 26)
    add_term(2, 10, 7, -1, 26)
    add_term(2, 11, 7, -1, 26)
    
    # M7
    add_term(3, 14, 4, 1, 27)
    add_term(3, 15, 4, 1, 27)
    add_term(7, 14, 4, -1, 27)
    add_term(7, 15, 4, -1, 27)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
