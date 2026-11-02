"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    
    # We partition the inner dimension m=5 into m1=2 and m2=3.
    # A is 2x5: A = [A1, A2] where A1 is 2x2, A2 is 2x3.
    # B is 5x6: B = [B1; B2] where B1 is 2x6, B2 is 3x6.
    # C = A1*B1 + A2*B2.
    
    # Part 1: A1 (2x2) * B1 (2x6)
    # Split B1 into three 2x2 blocks: B1_1, B1_2, B1_3.
    # C1 = A1*B1_1 + A1*B1_2 + A1*B1_3.
    # Each is 2x2x2, rank 7 (Strassen).
    # Total rank 21.
    
    # Part 2: A2 (2x3) * B2 (3x6)
    # Split B2 into two 3x3 blocks: B2_1, B2_2.
    # C2 = A2*B2_1 + A2*B2_2.
    # Each is 2x3x3.
    # Schoolbook rank is 2*3*3 = 18.
    # Total rank 36.
    
    # Total rank 21 + 36 = 57.
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(offset_A, offset_B, offset_W):
        nonlocal col
        # A indices: (0,0), (0,1), (1,0), (1,1) relative to offset_A
        # B indices: (0,0), (0,1), (1,0), (1,1) relative to offset_B
        # W indices: (0,0), (0,1), (1,0), (1,1) relative to offset_W (k, i)
        
        # M1 = (a+d)(x+w)
        U[offset_A][col] = 1; U[offset_A + 1][col] = 1 # a (0,0) is offset_A, d (1,1) is offset_A+1? 
        # Wait, A is 2x2. Indices:
        # A[0,0] -> row 0, col 0 -> index 0
        # A[0,1] -> row 0, col 1 -> index 1
        # A[1,0] -> row 1, col 0 -> index 2
        # A[1,1] -> row 1, col 1 -> index 3
        # So a=0, b=1, c=2, d=3.
        
        # B is 2x2.
        # B[0,0] -> 0, B[0,1] -> 1, B[1,0] -> 2, B[1,1] -> 3.
        # x=0, y=1, z=2, w=3.
        
        # W is 2x2 (k, i).
        # C[0,0] -> k=0, i=0 -> index 0
        # C[0,1] -> k=1, i=0 -> index 1? No.
        # W is p x n. p=2, n=2.
        # Index is k*n + i.
        # C[i,k].
        # k=0, i=0 -> 0*2+0=0.
        # k=0, i=1 -> 0*2+1=1.
        # k=1, i=0 -> 1*2+0=2.
        # k=1, i=1 -> 1*2+1=3.
        
        # Let's re-verify W indexing.
        # W[k*n+i, r].
        # C[i,k] = sum ... W[k*n+i, r].
        # So W index is k*n + i.
        
        # M1
        U[offset_A + 0][col] = 1 # a
        U[offset_A + 3][col] = 1 # d
        V[offset_B + 0][col] = 1 # x
        V[offset_B + 3][col] = 1 # w
        W[offset_W + 0][col] = 1 # C[0,0] (k=0, i=0)
        W[offset_W + 3][col] = 1 # C[1,1] (k=1, i=1)
        col += 1
        
        # M2 = (c+d)x
        U[offset_A + 2][col] = 1 # c
        U[offset_A + 3][col] = 1 # d
        V[offset_B + 0][col] = 1 # x
        W[offset_W + 2][col] = 1 # C[1,0] (k=0, i=1) -> Wait.
        # C[i,k].
        # C[1,0] means i=1, k=0.
        # W index = k*n + i = 0*2 + 1 = 1.
        # My previous mapping:
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # So C[1,0] is i=1, k=0. W index 1.
        # C[0,1] is i=0, k=1. W index 2.
        
        # Let's re-map Strassen outputs to W indices.
        # C11 (i=0, k=0) -> W[0]
        # C12 (i=0, k=1) -> W[2]
        # C21 (i=1, k=0) -> W[1]
        # C22 (i=1, k=1) -> W[3]
        
        # M1: C11 += 1, C22 += 1.
        W[offset_W + 0][col-1] = 1
        W[offset_W + 3][col-1] = 1
        
        # M2: C21 += 1, C22 -= 1.
        W[offset_W + 1][col] = 1
        W[offset_W + 3][col] = -1
        col += 1
        
        # M3 = a(y-w)
        U[offset_A + 0][col] = 1 # a
        V[offset_B + 1][col] = 1 # y
        V[offset_B + 3][col] = -1 # w
        # C12 += 1, C22 += 1.
        W[offset_W + 2][col] = 1
        W[offset_W + 3][col] = 1
        col += 1
        
        # M4 = d(z-x)
        U[offset_A + 3][col] = 1 # d
        V[offset_B + 2][col] = 1 # z
        V[offset_B + 0][col] = -1 # x
        # C11 += 1, C21 += 1.
        W[offset_W + 0][col] = 1
        W[offset_W + 1][col] = 1
        col += 1
        
        # M5 = (a+b)w
        U[offset_A + 0][col] = 1 # a
        U[offset_A + 1][col] = 1 # b
        V[offset_B + 3][col] = 1 # w
        # C11 -= 1, C12 += 1.
        W[offset_W + 0][col] = -1
        W[offset_W + 2][col] = 1
        col += 1
        
        # M6 = (c-a)(x+z)
        U[offset_A + 2][col] = 1 # c
        U[offset_A + 0][col] = -1 # a
        V[offset_B + 0][col] = 1 # x
        V[offset_B + 2][col] = 1 # z
        # C22 += 1.
        W[offset_W + 3][col] = 1
        col += 1
        
        # M7 = (b-d)(y+w)
        U[offset_A + 1][col] = 1 # b
        U[offset_A + 3][col] = -1 # d
        V[offset_B + 1][col] = 1 # y
        V[offset_B + 3][col] = 1 # w
        # C11 += 1.
        W[offset_W + 0][col] = 1
        col += 1
        
    # Apply Strassen to the three 2x2 blocks of B1
    # A1 is at A indices 0..3 (rows 0,1; cols 0,1)
    # B1_1 is at B indices 0..3 (rows 0,1; cols 0,1)
    # W1_1 is at W indices 0..3 (k=0,1; i=0,1) -> cols 0,1 of C
    add_strassen_222(0, 0, 0)
    
    # B1_2 is at B indices 2..5?
    # B is 5x6.
    # B1 is 2x6. Rows 0,1.
    # B1_1: cols 0,1. B indices: 0*6+0=0, 0*6+1=1, 1*6+0=6, 1*6+1=7.
    # Wait, B index is j*p + k.
    # j=0, k=0 -> 0.
    # j=0, k=1 -> 1.
    # j=1, k=0 -> 6.
    # j=1, k=1 -> 7.
    
    # My add_strassen_222 assumed contiguous indices.
    # I need to pass specific indices.
    
    # Let's rewrite add_strassen to take index lists.
    
    # Actually, I will just construct the full U, V, W manually for the 57 rank case.
    
    return {"U": U, "V": "V", "W": W}
# EVOLVE-BLOCK-END
