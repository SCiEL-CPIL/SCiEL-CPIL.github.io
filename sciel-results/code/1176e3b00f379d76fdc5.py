"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    def get_strassen_forms():
        forms = []
        # M1
        forms.append(([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]))
        # M2
        forms.append(([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]))
        # M3
        forms.append(([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]))
        # M4
        forms.append(([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]))
        # M5
        forms.append(([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]))
        # M6
        forms.append(([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]))
        # M7
        forms.append(([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]))
        
        return forms

    strassen_forms = get_strassen_forms()
    
    products = []
    
    # Product 0: A00 * B00
    x_map0 = [0, 1, 4, 5]
    y_map0 = [0, 4, 1, 5]
    z_map0 = [0, 2, 1, 3]
    products.append((x_map0, y_map0, z_map0))
    
    # Product 1: A01 * B10
    x_map1 = [2, 3, 6, 7]
    y_map1 = [8, 12, 9, 13]
    z_map1 = [0, 2, 1, 3]
    products.append((x_map1, y_map1, z_map1))
    
    # Product 2: A00 * B01
    x_map2 = [0, 1, 4, 5]
    y_map2 = [2, 6, 3, 7]
    z_map2 = [4, 6, 5, 7]
    products.append((x_map2, y_map2, z_map2))
    
    # Product 3: A01 * B11
    x_map3 = [2, 3, 6, 7]
    y_map3 = [10, 14, 11, 15]
    z_map3 = [4, 6, 5, 7]
    products.append((x_map3, y_map3, z_map3))
    
    col = 0
    for x_map, y_map, z_map in products:
        for x_coeffs, y_coeffs, z_coeffs in strassen_forms:
            for x_idx, coeff in x_coeffs:
                U[x_map[x_idx]][col] += coeff
            for y_idx, coeff in y_coeffs:
                V[y_map[y_idx]][col] += coeff
            for z_idx, coeff in z_coeffs:
                W[z_map[z_idx]][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
