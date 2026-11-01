"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split p into 5 blocks of size 2 and 1 block of size 1
    # Rank for 2x2x2 (Strassen) is 7
    # Rank for 2x2x1 (Trivial) is 4
    # Total rank = 5*7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block of size p_b
    def add_block(start_p, p_b):
        nonlocal col
        if p_b == 1:
            # Trivial 2x2x1
            # A: 2x2, B: 2x1, C: 2x1
            # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # V indices: 0:(0,0), 1:(1,0) (j*p+k, p=1)
            # W indices: 0:(0,0), 1:(1,0) (k*n+i, n=2)
            
            # 1. A00 B00 -> C00
            U[0][col] = 1; V[0][col] = 1; W[0][col] = 1; col += 1
            # 2. A01 B10 -> C00
            U[1][col] = 1; V[1][col] = 1; W[0][col] = 1; col += 1
            # 3. A10 B00 -> C10
            U[2][col] = 1; V[0][col] = 1; W[1][col] = 1; col += 1
            # 4. A11 B10 -> C10
            U[3][col] = 1; V[1][col] = 1; W[1][col] = 1; col += 1
            
        elif p_b == 2:
            # Strassen 2x2x2
            # A: 2x2, B: 2x2, C: 2x2
            # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) (j*p+k, p=2)
            # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) (k*n+i, n=2)
            
            # M1 = (A00+A11)(B00+B11)
            U[0][col] = 1; U[3][col] = 1
            V[0][col] = 1; V[3][col] = 1
            W[0][col] = 1; W[3][col] = 1
            col += 1
            
            # M2 = (A10+A11)B00
            U[2][col] = 1; U[3][col] = 1
            V[0][col] = 1
            W[2][col] = 1; W[3][col] = -1
            col += 1
            
            # M3 = A00(B10-B11)
            U[0][col] = 1
            V[2][col] = 1; V[3][col] = -1
            W[1][col] = 1; W[3][col] = 1
            col += 1
            
            # M4 = A11(B01-B00)
            U[3][col] = 1
            V[1][col] = 1; V[0][col] = -1
            W[0][col] = 1; W[2][col] = 1
            col += 1
            
            # M5 = (A00+A01)B11
            U[0][col] = 1; U[1][col] = 1
            V[3][col] = 1
            W[0][col] = -1; W[1][col] = 1
            col += 1
            
            # M6 = (A10-A00)(B00+B10)
            U[2][col] = 1; U[0][col] = -1
            V[0][col] = 1; V[2][col] = 1
            W[3][col] = 1
            col += 1
            
            # M7 = (A01-A11)(B11+B01)
            U[1][col] = 1; U[3][col] = -1
            V[3][col] = 1; V[1][col] = 1
            W[0][col] = 1; W[3][col] = 1
            col += 1

    # 5 blocks of size 2
    for i in range(5):
        add_block(i * 2, 2)
        
    # 1 block of size 1
    add_block(10, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
