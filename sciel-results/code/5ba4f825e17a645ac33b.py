"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Use Strassen decomposition for each pair of output columns
    # 2x2 * 2x10 = 5 blocks of 2x2 * 2x2
    # Each 2x2*2x2 uses rank 7 (Strassen)
    # Total rank = 5 * 7 = 35
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen factors for 2x2 * 2x2:
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C11+=1, C22+=1
    # M2 = (a+b)(h-g): U: a=1,b=1; V: h=1,g=-1; W: C22-=1
    # M3 = (c+d)(e+f): U: c=1,d=1; V: e=1,f=1; W: C21+=1, C22+=1
    # M4 = d(f-g): U: d=1; V: f=1,g=-1; W: C11+=1, C21+=1
    # M5 = a(e-h): U: a=1; V: e=1,h=-1; W: C11-=1, C12+=1
    # M6 = b(g-e): U: b=1; V: g=1,e=-1; W: C12+=1
    # M7 = c(h-f): U: c=1; V: h=1,f=-1; W: C11-=1, C22+=1
    
    # For block t (t=0..4), output columns are 2t and 2t+1
    # B row 0, col 2t: index 0*10 + 2t = 2t
    # B row 0, col 2t+1: index 0*10 + 2t+1 = 2t+1
    # B row 1, col 2t: index 1*10 + 2t = 10+2t
    # B row 1, col 2t+1: index 1*10 + 2t+1 = 10+2t+1
    
    # C row 0, col 2t: index 2t*2 + 0 = 4t
    # C row 0, col 2t+1: index (2t+1)*2 + 0 = 4t+2
    # C row 1, col 2t: index 2t*2 + 1 = 4t+1
    # C row 1, col 2t+1: index (2t+1)*2 + 1 = 4t+3
    
    # A indices: i*m+j where i=row, j=col
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    
    for t in range(5):
        base = t * 7
        # B indices for this block
        b00 = 0 * p + 2*t      # B[0, 2t]
        b01 = 0 * p + 2*t + 1  # B[0, 2t+1]
        b10 = 1 * p + 2*t      # B[1, 2t]
        b11 = 1 * p + 2*t + 1  # B[1, 2t+1]
        
        # C indices for this block
        c00 = (2*t) * n + 0    # C[0, 2t]
        c01 = (2*t+1) * n + 0  # C[0, 2t+1]
        c10 = (2*t) * n + 1    # C[1, 2t]
        c11 = (2*t+1) * n + 1  # C[1, 2t+1]
        
        # M1 = (a+d)(e+h)
        col = base + 0
        U[0][col] = 1  # a
        U[3][col] = 1  # d
        V[b00][col] = 1  # e
        V[b11][col] = 1  # h
        W[c00][col] = 1  # C11 += M1
        W[c11][col] = 1  # C22 += M1
        
        # M2 = (a+b)(h-g)
        col = base + 1
        U[0][col] = 1  # a
        U[1][col] = 1  # b
        V[b11][col] = 1  # h
        V[b10][col] = -1  # -g
        W[c11][col] = -1  # C22 -= M2
        
        # M3 = (c+d)(e+f)
        col = base + 2
        U[2][col] = 1  # c
        U[3][col] = 1  # d
        V[b00][col] = 1  # e
        V[b01][col] = 1  # f
        W[c10][col] = 1  # C21 += M3
        W[c11][col] = 1  # C22 += M3
        
        # M4 = d(f-g)
        col = base + 3
        U[3][col] = 1  # d
        V[b01][col] = 1  # f
        V[b10][col] = -1  # -g
        W[c00][col] = 1  # C11 += M4
        W[c10][col] = 1  # C21 += M4
        
        # M5 = a(e-h)
        col = base + 4
        U[0][col] = 1  # a
        V[b00][col] = 1  # e
        V[b11][col] = -1  # -h
        W[c00][col] = -1  # C11 -= M5
        W[c01][col] = 1  # C12 += M5
        
        # M6 = b(g-e)
        col = base + 5
        U[1][col] = 1  # b
        V[b10][col] = 1  # g
        V[b00][col] = -1  # -e
        W[c01][col] = 1  # C12 += M6
        
        # M7 = c(h-f)
        col = base + 6
        U[2][col] = 1  # c
        V[b11][col] = 1  # h
        V[b01][col] = -1  # -f
        W[c00][col] = -1  # C11 -= M7
        W[c11][col] = 1  # C22 += M7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
