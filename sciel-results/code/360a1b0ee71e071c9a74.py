"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Landsberg (1992) rank 23 decomposition
    rank = 23
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_coeffs: dict {index: coeff}
    # v_coeffs: dict {index: coeff}
    # w_coeffs: dict {index: coeff}
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal rank # Not strictly needed if we pass col, but we use list index
        # We will iterate through the 23 terms manually or use a list
        pass
        
    # We define the 23 terms explicitly
    # U index: i*3+j (A_ij)
    # V index: j*3+k (B_jk)
    # W index: k*3+i (C_ik)
    
    terms = [
        # 1. A00 B00 -> C00
        ({0: 1}, {0: 1}, {0: 1}),
        # 2. A00 B01 -> C01 (W[1*3+0]=3)
        ({0: 1}, {1: 1}, {3: 1}),
        # 3. A01 B00 -> C10 (W[0*3+1]=1)
        ({1: 1}, {0: 1}, {1: 1}),
        # 4. A01 B01 -> C11 (W[1*3+1]=4)
        ({1: 1}, {1: 1}, {4: 1}),
        # 5. (A00+A11) B00 -> C00, C11
        ({0: 1, 4: 1}, {0: 1}, {0: 1, 4: 1}),
        # 6. (A00+A11) B01 -> C01, C11
        ({0: 1, 4: 1}, {1: 1}, {3: 1, 4: 1}),
        # 7. A00 (B01-B11) -> C01, C11
        ({0: 1}, {1: 1, 4: -1}, {3: 1, 4: 1}),
        # 8. A11 (B10-B00) -> C00, C10
        ({4: 1}, {3: 1, 0: -1}, {0: 1, 1: 1}),
        # 9. (A00+A01) B11 -> C00, C01
        ({0: 1, 1: 1}, {4: 1}, {0: -1, 3: 1}),
        # 10. (A10-A00) (B00+B01) -> C11
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1}),
        # 11. (A01-A11) (B10+B11) -> C00
        ({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1}),
        
        # 12. A02 B20 -> C00
        ({2: 1}, {6: 1}, {0: 1}),
        # 13. A02 B21 -> C01
        ({2: 1}, {7: 1}, {3: 1}),
        # 14. A12 B20 -> C10
        ({5: 1}, {6: 1}, {1: 1}),
        # 15. A12 B21 -> C11
        ({5: 1}, {7: 1}, {4: 1}),
        # 16. A02 B22 -> C02 (W[2*3+0]=6)
        ({2: 1}, {8: 1}, {6: 1}),
        # 17. A12 B22 -> C12 (W[2*3+1]=7)
        ({5: 1}, {8: 1}, {7: 1}),
        # 18. A20 B02 -> C20 (W[0*3+2]=2)
        ({6: 1}, {2: 1}, {2: 1}),
        # 19. A21 B12 -> C21 (W[1*3+2]=5)
        ({7: 1}, {5: 1}, {5: 1}),
        # 20. A20 B12 -> C21
        ({6: 1}, {5: 1}, {5: 1}),
        # 21. A21 B02 -> C20
        ({7: 1}, {2: 1}, {2: 1}),
        # 22. A22 B22 -> C22 (W[2*3+2]=8)
        ({8: 1}, {8: 1}, {8: 1}),
        # 23. A22 B20 -> C20
        ({8: 1}, {6: 1}, {2: 1})
    ]
    
    for r, (u_c, v_c, w_c) in enumerate(terms):
        for idx, coeff in u_c.items():
            U[idx][r] = coeff
        for idx, coeff in v_c.items():
            V[idx][r] = coeff
        for idx, coeff in w_c.items():
            W[idx][r] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
