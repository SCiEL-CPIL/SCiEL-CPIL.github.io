"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen product
    # a_coeffs: dict {u_index: coeff} for A
    # b_coeffs: dict {v_index: coeff} for B
    # c_coeffs: dict {w_index: coeff} for C
    def add_product(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        for u, val in a_coeffs.items():
            U[u][col] += val
        for v, val in b_coeffs.items():
            V[v][col] += val
        for w, val in c_coeffs.items():
            W[w][col] += val
        col += 1
        
    # Indices for A (U): i*4 + j
    # A[0][0]=0, A[0][1]=1, A[0][2]=2, A[0][3]=3
    # A[1][0]=4, A[1][1]=5, A[1][2]=6, A[1][3]=7
    
    # Indices for B (V): j*4 + k
    # B[0][0]=0, B[0][1]=1, B[0][2]=2, B[0][3]=3
    # B[1][0]=4, B[1][1]=5, B[1][2]=6, B[1][3]=7
    # B[2][0]=8, B[2][1]=9, B[2][2]=10, B[2][3]=11
    # B[3][0]=12, B[3][1]=13, B[3][2]=14, B[3][3]=15
    
    # Indices for C (W): k*2 + i
    # C[0][0]=0, C[1][0]=1
    # C[0][1]=2, C[1][1]=3
    # C[0][2]=4, C[1][2]=5
    # C[0][3]=6, C[1][3]=7
    
    # --- Left Part (k=0,1) ---
    
    # P_L = A_L * B^(1)
    # A_L: [[A00, A01], [A10, A11]] -> U: 0,1,4,5
    # B^(1): [[B00, B01], [B10, B11]] -> V: 0,1,4,5
    
    # M1 = (A00+A11)(B00+B11)
    add_product({0:1, 5:1}, {0:1, 5:1}, {0:1, 3:1, 4:1, 7:1}) # C00, C11, C02? No, just Left C00, C11
    # Wait, the c_coeffs depend on which part (Left/Right) and which Strassen product.
    # For Left Part:
    # C00 (W0) gets M1
    # C11 (W3) gets M1
    # C02 (W4) is Right Part.
    # So for P_L (Left), c_coeffs should only touch W0, W1, W2, W3.
    
    # Let's re-verify the C indices for Left Part (k=0,1):
    # C[0][0] -> W[0]
    # C[1][0] -> W[1]
    # C[0][1] -> W[2]
    # C[1][1] -> W[3]
    
    # Reset col to 0 to be safe, or just keep tracking.
    # I'll restart col tracking logic in the helper or just pass col.
    
    # Let's define the 14 products for Left.
    
    # P_L Products
    # PL1: (A00+A11)(B00+B11)
    add_product({0:1, 5:1}, {0:1, 5:1}, {0:1, 3:1}) # C00, C11
    # PL2: (A10+A11)B00
    add_product({4:1, 5:1}, {0:1}, {1:1, 3:-1}) # C10, C11
    # PL3: A00(B01-B11)
    add_product({0:1}, {1:1, 5:-1}, {2:1, 3:1}) # C01, C11
    # PL4: A11(B10-B00)
    add_product({5:1}, {4:1, 0:-1}, {0:1, 1:1}) # C00, C10
    # PL5: (A00+A01)B11
    add_product({0:1, 1:1}, {5:1}, {0:-1, 2:1}) # C00, C01
    # PL6: (A10-A00)(B00+B01)
    add_product({4:1, 0:-1}, {0:1, 1:1}, {3:1}) # C11
    # PL7: (A01-A11)(B10+B11)
    add_product({1:1, 5:-1}, {4:1, 5:1}, {0:1}) # C00
    
    # Q_L Products
    # A_R: [[A02, A03], [A12, A13]] -> U: 2,3,6,7
    # B^(2): [[B20, B21], [B30, B31]] -> V: 8,9,12,13
    
    # QL1: (A02+A13)(B20+B31)
    add_product({2:1, 7:1}, {8:1, 13:1}, {0:1, 3:1})
    # QL2: (A12+A13)B20
    add_product({6:1, 7:1}, {8:1}, {1:1, 3:-1})
    # QL3: A02(B21-B31)
    add_product({2:1}, {9:1, 13:-1}, {2:1, 3:1})
    # QL4: A13(B30-B20)
    add_product({7:1}, {12:1, 8:-1}, {0:1, 1:1})
    # QL5: (A02+A03)B31
    add_product({2:1, 3:1}, {13:1}, {0:-1, 2:1})
    # QL6: (A12-A02)(B20+B21)
    add_product({6:1, 2:-1}, {8:1, 9:1}, {3:1})
    # QL7: (A03-A13)(B30+B31)
    add_product({3:1, 7:-1}, {12:1, 13:1}, {0:1})
    
    # --- Right Part (k=2,3) ---
    # C[0][2] -> W[4]
    # C[1][2] -> W[5]
    # C[0][3] -> W[6]
    # C[1][3] -> W[7]
    
    # P_R = A_L * B^(3)
    # B^(3): [[B02, B03], [B12, B13]] -> V: 2,3,6,7
    
    # PR1: (A00+A11)(B02+B13)
    add_product({0:1, 5:1}, {2:1, 7:1}, {4:1, 7:1})
    # PR2: (A10+A11)B02
    add_product({4:1, 5:1}, {2:1}, {5:1, 7:-1})
    # PR3: A00(B03-B13)
    add_product({0:1}, {3:1, 7:-1}, {6:1, 7:1})
    # PR4: A11(B12-B02)
    add_product({5:1}, {6:1, 2:-1}, {4:1, 5:1})
    # PR5: (A00+A01)B13
    add_product({0:1, 1:1}, {7:1}, {4:-1, 6:1})
    # PR6: (A10-A00)(B02+B03)
    add_product({4:1, 0:-1}, {2:1, 3:1}, {7:1})
    # PR7: (A01-A11)(B12+B13)
    add_product({1:1, 5:-1}, {6:1, 7:1}, {4:1})
    
    # Q_R = A_R * B^(4)
    # B^(4): [[B22, B23], [B32, B33]] -> V: 10,11,14,15
    
    # QR1: (A02+A13)(B22+B33)
    add_product({2:1, 7:1}, {10:1, 15:1}, {4:1, 7:1})
    # QR2: (A12+A13)B22
    add_product({6:1, 7:1}, {10:1}, {5:1, 7:-1})
    # QR3: A02(B23-B33)
    add_product({2:1}, {11:1, 15:-1}, {6:1, 7:1})
    # QR4: A13(B32-B22)
    add_product({7:1}, {14:1, 10:-1}, {4:1, 5:1})
    # QR5: (A02+A03)B33
    add_product({2:1, 3:1}, {15:1}, {4:-1, 6:1})
    # QR6: (A12-A02)(B22+B23)
    add_product({6:1, 2:-1}, {10:1, 11:1}, {7:1})
    # QR7: (A03-A13)(B32+B33)
    add_product({3:1, 7:-1}, {14:1, 15:1}, {4:1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
