"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=4
    # We multiply A (n x m) by B (m x p) to get C (n x p).
    # The tensor has dimensions (n*m) x (m*p) x (p*n) = 4 x 8 x 8.
    # Wait, the output factor W has shape (p*n, rank).
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This means W is indexed by (k, i) where k is in [0, p) and i is in [0, n).
    # So W index = k * n + i.
    # U is indexed by (i, j) where i in [0, n), j in [0, m). Index = i * m + j.
    # V is indexed by (j, k) where j in [0, m), k in [0, p). Index = j * p + k.
    
    # We can view the multiplication of A (2x2) and B (2x4) as two independent 2x2x2 multiplications.
    # Let B be split into B_left (2x2) and B_right (2x2).
    # C_left = A * B_left (2x2)
    # C_right = A * B_right (2x2)
    # C = [C_left, C_right] (2x4)
    
    # The rank of 2x2x2 matrix multiplication is 7 (Strassen).
    # Since the two multiplications are independent (disjoint output indices), the total rank is 7 + 7 = 14.
    
    # Strassen's algorithm for 2x2x2:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # Strassen products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Results:
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Let's define the Strassen factors for a generic 2x2x2 block.
    # U_strassen: 4 x 7
    # V_strassen: 4 x 7
    # W_strassen: 4 x 7 (but indexed differently?)
    
    # Indices for 2x2x2:
    # U index: i*m + j. For 2x2, m=2.
    # i=0, j=0 -> 0 (a11)
    # i=0, j=1 -> 1 (a12)
    # i=1, j=0 -> 2 (a21)
    # i=1, j=1 -> 3 (a22)
    
    # V index: j*p + k. For 2x2, p=2.
    # j=0, k=0 -> 0 (b11)
    # j=0, k=1 -> 1 (b12)
    # j=1, k=0 -> 2 (b21)
    # j=1, k=1 -> 3 (b22)
    
    # W index: k*n + i. For 2x2, n=2.
    # k=0, i=0 -> 0 (c11)
    # k=0, i=1 -> 1 (c21)  <- Wait, k is column index of C?
    # C is n x p. C[i, k].
    # The trace convention is W[k*n + i, r].
    # So for k=0, i=0 -> 0 (c11)
    # For k=0, i=1 -> 1 (c21)
    # For k=1, i=0 -> 2 (c12)
    # For k=1, i=1 -> 3 (c22)
    
    # Let's build U, V, W for one Strassen block.
    # U (4x7):
    # Row 0 (a11): M1(1), M3(1), M5(1), M6(-1) -> [1, 0, 1, 0, 1, -1, 0]
    # Row 1 (a12): M3(1), M5(1), M7(-1) -> [0, 0, 1, 0, 1, 0, -1]
    # Row 2 (a21): M2(1), M6(-1) -> [0, 1, 0, 0, 0, -1, 0]
    # Row 3 (a22): M1(1), M2(1), M4(1), M7(-1) -> [1, 1, 0, 1, 0, 0, -1]
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 1, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # V (4x7):
    # Row 0 (b11): M1(1), M2(1), M4(-1), M6(1) -> [1, 1, 0, -1, 0, 1, 0]
    # Row 1 (b12): M3(-1), M6(1), M7(1) -> [0, 0, -1, 0, 0, 1, 1]
    # Row 2 (b21): M4(1), M6(1), M7(1) -> [0, 0, 0, 1, 0, 1, 1]
    # Row 3 (b22): M1(1), M3(1), M5(1), M7(1) -> [1, 0, 1, 0, 1, 0, 1]
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, -1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ]
    
    # W (4x7):
    # Row 0 (c11): M1(1), M4(-1), M5(1), M7(-1) -> [1, 0, 0, -1, 1, 0, -1]
    # Row 1 (c21): M2(1), M4(1) -> [0, 1, 0, 1, 0, 0, 0]
    # Row 2 (c12): M3(1), M5(1) -> [0, 0, 1, 0, 1, 0, 0]
    # Row 3 (c22): M1(1), M2(-1), M3(1), M6(1) -> [1, -1, 1, 0, 0, 1, 0]
    
    W_s = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Now construct full U, V, W for 2x2x4.
    # U is (n*m, rank) = (4, 14).
    # V is (m*p, rank) = (8, 14).
    # W is (p*n, rank) = (8, 14).
    
    rank = 14
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(8)]
    W = [[0] * rank for _ in range(8)]
    
    # Block 0: k in {0, 1} (B_left, C_left)
    # V indices for block 0:
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 2
    # j=1, k=1 -> 3
    # These correspond to rows 0-3 of V_s.
    
    # W indices for block 0:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # These correspond to rows 0-3 of W_s.
    
    for r in range(7):
        # U is same for both blocks
        for i in range(4):
            U[i][r] = U_s[i][r]
        
        # V block 0
        for j in range(4):
            V[j][r] = V_s[j][r]
        
        # W block 0
        for k in range(4):
            W[k][r] = W_s[k][r]
        
        # V block 1 (offset 4)
        for j in range(4):
            V[j + 4][r + 7] = V_s[j][r]
        
        # W block 1 (offset 4)
        for k in range(4):
            W[k + 4][r + 7] = W_s[k][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
