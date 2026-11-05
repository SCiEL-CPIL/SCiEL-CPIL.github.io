"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # Rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_2 * B_2
    # A_2 is column 2 of A (indices 2, 5 in flattened A)
    # B_2 is row 2 of B (indices 16..23 in flattened B)
    # C is 2x8 (indices 0..15 in flattened C)
    # C[i, k] = A[i, 2] * B[2, k]
    col = 0
    for i in range(n):
        for k in range(p):
            # U index: i*m + 2
            U[i * m + 2][col] = 1
            # V index: 2*p + k
            V[2 * p + k][col] = 1
            # W index: k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_1 * B_1
    # A_1 is 2x2 submatrix of A (cols 0, 1)
    # B_1 is 2x8 submatrix of B (rows 0, 1)
    # Split B_1 into 4 blocks of 2x2
    for q in range(4):
        # Strassen for block q
        # A indices:
        # a=A[0,0] (0), b=A[0,1] (1)
        # c=A[1,0] (3), d=A[1,1] (4)
        # B indices for block q:
        # e=B[0, 2q] (0*p + 2q), f=B[0, 2q+1] (0*p + 2q+1)
        # g=B[1, 2q] (1*p + 2q), h=B[1, 2q+1] (1*p + 2q+1)
        # C indices for block q:
        # C[0, 2q] (2q*n + 0), C[0, 2q+1] (2q+1)*n + 0
        # C[1, 2q] (2q*n + 1), C[1, 2q+1] (2q+1)*n + 1
        
        # M1 = (a+d)(e+h)
        U[0][col] += 1
        U[4][col] += 1
        V[2*q][col] += 1
        V[2*q+1][col] += 1
        W[2*q*n + 0][col] += 1
        W[2*q*n + 1][col] += 1
        W[(2*q+1)*n + 0][col] += -1
        W[(2*q+1)*n + 1][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[3][col] += 1
        U[4][col] += 1
        V[2*q][col] += 1
        W[2*q*n + 1][col] += 1
        W[(2*q+1)*n + 1][col] += -1
        col += 1
        
        # M3 = a(f-h)
        U[0][col] += 1
        V[2*q+1][col] += 1
        V[2*q+1+8][col] += -1 # g is at 1*p + 2q = 8+2q? No.
        # B indices: row 0 is 0..7, row 1 is 8..15.
        # e = B[0, 2q] -> idx 2q
        # f = B[0, 2q+1] -> idx 2q+1
        # g = B[1, 2q] -> idx 8+2q
        # h = B[1, 2q+1] -> idx 8+2q+1
        # Correct V indices:
        # e: 2q
        # f: 2q+1
        # g: 8+2q
        # h: 8+2q+1
        
        # Let's re-do M3
        # M3 = a(f-h)
        # a = A[0,0] idx 0
        # f = B[0, 2q+1] idx 2q+1
        # h = B[1, 2q+1] idx 8+2q+1
        # W contributions:
        # C[0, 2q+1] += M3 -> W[(2q+1)*n + 0] += 1
        # C[1, 2q+1] += M3 -> W[(2q+1)*n + 1] += 1
        # Wait, Strassen formulas:
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # Let's restart the Strassen mapping carefully.
        
        # Block q starts at col.
        # M1
        U[0][col] += 1; U[4][col] += 1
        V[2*q][col] += 1; V[8+2*q+1][col] += 1 # e+h? No.
        # e = B[0, 2q], h = B[1, 2q+1]
        # M1 = (a+d)(e+h)
        # a=0, d=4. e=2q, h=8+2q+1.
        # Wait, h is B[1, 2q+1].
        # e is B[0, 2q].
        # So e+h means V[2q] and V[8+2q+1].
        # But M1 is product of (a+d) and (e+h).
        # So U has 1 at 0 and 4.
        # V has 1 at 2q and 1 at 8+2q+1?
        # No, (e+h) means the linear form is e+h.
        # So V[2q]=1, V[8+2q+1]=1.
        # W contributions:
        # C00 += M1, C11 += M1
        # C00 is W[2q*n+0], C11 is W[(2q+1)*n+1]
        W[2*q*n + 0][col] += 1
        W[(2*q+1)*n + 1][col] += 1
        col += 1
        
        # M2 = (c+d)e
        # c=3, d=4. e=2q.
        U[3][col] += 1; U[4][col] += 1
        V[2*q][col] += 1
        # C10 += M2, C11 -= M2
        # C10 is W[2q*n+1], C11 is W[(2q+1)*n+1]
        W[2*q*n + 1][col] += 1
        W[(2*q+1)*n + 1][col] += -1
        col += 1
        
        # M3 = a(f-h)
        # a=0. f=2q+1, h=8+2q+1.
        U[0][col] += 1
        V[2*q+1][col] += 1
        V[8+2*q+1][col] += -1
        # C01 += M3, C11 += M3
        # C01 is W[(2q+1)*n+0], C11 is W[(2q+1)*n+1]
        W[(2*q+1)*n + 0][col] += 1
        W[(2*q+1)*n + 1][col] += 1
        col += 1
        
        # M4 = d(g-e)
        # d=4. g=8+2q, e=2q.
        U[4][col] += 1
        V[8+2*q][col] += 1
        V[2*q][col] += -1
        # C00 += M4, C10 += M4
        # C00 is W[2q*n+0], C10 is W[2q*n+1]
        W[2*q*n + 0][col] += 1
        W[2*q*n + 1][col] += 1
        col += 1
        
        # M5 = (a+b)h
        # a=0, b=1. h=8+2q+1.
        U[0][col] += 1; U[1][col] += 1
        V[8+2*q+1][col] += 1
        # C00 -= M5, C01 += M5
        # C00 is W[2q*n+0], C01 is W[(2q+1)*n+0]
        W[2*q*n + 0][col] += -1
        W[(2*q+1)*n + 0][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        # c=3, a=0. e=2q, f=2q+1.
        U[3][col] += 1; U[0][col] += -1
        V[2*q][col] += 1; V[2*q+1][col] += 1
        # C11 += M6
        # C11 is W[(2q+1)*n+1]
        W[(2*q+1)*n + 1][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        # b=1, d=4. g=8+2q, h=8+2q+1.
        U[1][col] += 1; U[4][col] += -1
        V[8+2*q][col] += 1; V[8+2*q+1][col] += 1
        # C00 += M7
        # C00 is W[2q*n+0]
        W[2*q*n + 0][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
