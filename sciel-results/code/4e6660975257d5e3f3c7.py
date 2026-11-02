"""Matrix multiplication 2,2,8 with Strassen-based rank-28 decomposition."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We have 4 blocks of 2 columns each (since p=8, n=2, so 4 blocks of size 2x2)
    # Each block uses 7 Strassen multiplications
    rank = 4 * 7
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 16 x rank
    W = [[0] * rank for _ in range(p * n)]  # 16 x rank
    
    # Strassen formulas for 2x2 matrix multiplication
    # Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # For each block b (0..3), the B submatrix columns are 4*b, 4*b+1, 4*b+2, 4*b+3
    # W index convention: W[k*n + i, r] where k is the column in output (which corresponds to B's column k), i is row in output
    
    for b in range(4):
        base_col = b * 7  # Starting rank column for this block
        # B block columns: 4*b, 4*b+1, 4*b+2, 4*b+3
        # Output C columns: 4*b, 4*b+1, 4*b+2, 4*b+3 (same as B columns since p=8 and we're doing 2x2 blocks)
        
        # Strassen term 1: (a+d)(e+h)
        # U contributions: a (i=0,j=0), d (i=1,j=1)
        # V contributions: e (j=0,k=4b), h (j=1,k=4b+1)
        # W contributions: C[0,0] += 1, C[1,1] += 1
        U[0*2+0][base_col+0] += 1
        U[1*2+1][base_col+0] += 1
        V[0*8+4*b][base_col+0] += 1
        V[1*8+4*b+1][base_col+0] += 1
        W[(4*b)*2+0][base_col+0] += 1   # C[0, 4b]
        W[(4*b+3)*2+1][base_col+0] += 1 # C[1, 4b+3]
        
        # Strassen term 2: (c+d)e
        # U contributions: c (i=1,j=0), d (i=1,j=1)
        # V contributions: e (j=0,k=4b)
        # W contributions: C[1,0] += 1, C[1,1] -= 1
        U[1*2+0][base_col+1] += 1
        U[1*2+1][base_col+1] += 1
        V[0*8+4*b][base_col+1] += 1
        W[(4*b)*2+1][base_col+1] += 1   # C[1, 4b]
        W[(4*b+3)*2+1][base_col+1] -= 1 # C[1, 4b+3]
        
        # Strassen term 3: a(f-h)
        # U contributions: a (i=0,j=0)
        # V contributions: f (j=1,k=4b+1), -h (j=1,k=4b+1)
        # Wait, f is B[1, 4b+1]? No, B is 2x8, so B[j, k]
        # In the 2x2 block, B[0, 4b]=e, B[0, 4b+1]=f, B[1, 4b]=g, B[1, 4b+1]=h
        # So f = B[0, 4b+1], h = B[1, 4b+1]
        U[0*2+0][base_col+2] += 1
        V[0*8+4*b+1][base_col+2] += 1  # f
        V[1*8+4*b+1][base_col+2] -= 1  # -h
        W[(4*b+1)*2+0][base_col+2] += 1 # C[0, 4b+1]
        W[(4*b+3)*2+0][base_col+2] += 1 # C[0, 4b+3]
        
        # Strassen term 4: d(g-e)
        # U contributions: d (i=1,j=1)
        # V contributions: g (j=1,k=4b), -e (j=0,k=4b)
        U[1*2+1][base_col+3] += 1
        V[1*8+4*b][base_col+3] += 1   # g
        V[0*8+4*b][base_col+3] -= 1   # -e
        W[(4*b)*2+0][base_col+3] += 1   # C[0, 4b]
        W[(4*b)*2+1][base_col+3] += 1   # C[1, 4b]
        
        # Strassen term 5: (a+b)h
        # U contributions: a (i=0,j=0), b (i=0,j=1)
        # V contributions: h (j=1,k=4b+1)
        # W contributions: C[0,0] -= 1, C[0,1] += 1
        U[0*2+0][base_col+4] += 1
        U[0*2+1][base_col+4] += 1
        V[1*8+4*b+1][base_col+4] += 1  # h
        W[(4*b)*2+0][base_col+4] -= 1   # C[0, 4b]
        W[(4*b+1)*2+0][base_col+4] += 1 # C[0, 4b+1]
        
        # Strassen term 6: (c-a)(e+f)
        # U contributions: c (i=1,j=0), -a (i=0,j=0)
        # V contributions: e (j=0,k=4b), f (j=0,k=4b+1)
        # W contributions: C[1,1] += 1
        U[1*2+0][base_col+5] += 1
        U[0*2+0][base_col+5] -= 1
        V[0*8+4*b][base_col+5] += 1   # e
        V[0*8+4*b+1][base_col+5] += 1 # f
        W[(4*b+3)*2+1][base_col+5] += 1 # C[1, 4b+3]
        
        # Strassen term 7: (b-d)(g+h)
        # U contributions: b (i=0,j=1), -d (i=1,j=1)
        # V contributions: g (j=1,k=4b), h (j=1,k=4b+1)
        # W contributions: C[0,0] += 1
        U[0*2+1][base_col+6] += 1
        U[1*2+1][base_col+6] -= 1
        V[1*8+4*b][base_col+6] += 1   # g
        V[1*8+4*b+1][base_col+6] += 1 # h
        W[(4*b)*2+0][base_col+6] += 1   # C[0, 4b]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
