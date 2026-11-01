"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 14 decomposition using two Strassen blocks for 2x2x4 matrix multiplication.
    n, m, p = DIMENSIONS
    rank = 14
    
    # U shape (n*m, rank) = (4, 14)
    # V shape (m*p, rank) = (8, 14)
    # W shape (p*n, rank) = (8, 14)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals, col):
        for i in range(n * m):
            U[i][col] = u_vals[i]
        for j in range(m * p):
            V[j][col] = v_vals[j]
        for k in range(p * n):
            W[k][col] = w_vals[k]

    # Indices for A (4): 0:00, 1:01, 2:10, 3:11
    # Indices for B (8): 0:00, 1:01, 2:02, 3:03, 4:10, 5:11, 6:12, 7:13
    # Indices for C (8): 0:00, 1:10, 2:01, 3:11, 4:02, 5:12, 6:03, 7:13
    # Note: W index is k*n + i.
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C10)
    # k=1, i=0 -> 2 (C01)
    # k=1, i=1 -> 3 (C11)
    # k=2, i=0 -> 4 (C02)
    # k=2, i=1 -> 5 (C12)
    # k=3, i=0 -> 6 (C03)
    # k=3, i=1 -> 7 (C13)
    
    # Left Part (Cols 0, 1)
    # 1. L1 = (A00+A11)(B00+B11)
    add_term([1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], 0)
    
    # 2. L2 = (A10+A11)B00
    add_term([0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0, 0, 0], 1)
    
    # 3. L3 = A00(B01-B11)
    add_term([1, 0, 0, 0], [0, 1, 0, 0, 0, -1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], 2)
    
    # 4. L4 = A11(B10-B00)
    add_term([0, 0, 0, 1], [-1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], 3)
    
    # 5. L5 = (A00+A01)B11
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0], 4)
    
    # 6. L6 = (A10-A00)(B00+B01)
    add_term([-1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], 5)
    
    # 7. L7 = (A01-A11)(B10+B11)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], 6)
    
    # Right Part (Cols 2, 3)
    # 8. R1 = (A00+A11)(B02+B13)
    add_term([1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1], 7)
    
    # 9. R2 = (A10+A11)B02
    add_term([0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, -1], 8)
    
    # 10. R3 = A00(B03-B13)
    add_term([1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], 9)
    
    # 11. R4 = A11(B12-B02)
    add_term([0, 0, 0, 1], [0, 0, -1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0], 10)
    
    # 12. R5 = (A00+A01)B13
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0, 1, 0], 11)
    
    # 13. R6 = (A10-A00)(B02+B03)
    add_term([-1, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], 12)
    
    # 14. R7 = (A01-A11)(B12+B13)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0], 13)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
