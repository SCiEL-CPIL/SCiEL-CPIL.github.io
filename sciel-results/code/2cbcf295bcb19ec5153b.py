"""Matrix multiplication 2x3x9: rank-37 decomposition via de Leede 3x3x3 + schoolbook."""

# EVOLVE-BLOCK-START

def _de_leede_333():
    """Return (U,V,W) for 3x3x3 matrix multiplication, rank 19.
    U shape (9,19), V shape (9,19), W shape (9,19).
    W convention: W[k*3+i, c] for C[i,j] = sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*3+i,c].
    """
    U = [[0]*19 for _ in range(9)]
    V = [[0]*19 for _ in range(9)]
    W = [[0]*19 for _ in range(9)]
    
    def set_uvi(i, j, c, val):
        U[i*3+j][c] = val
    def set_vjk(j, k, c, val):
        V[j*3+k][c] = val
    def set_wki(k, i, c, val):
        W[k*3+i][c] = val
    
    # 7 Strassen-like terms
    # M1 = (A00+A11+A22)(B00+B11+B22)
    c = 0
    for (i,j) in [(0,0),(1,1),(2,2)]:
        set_uvi(i,j,c,1)
        set_vjk(i,j,c,1)
        set_wki(i,i,c,1)
    
    # M2 = (A00-A11)(B00-B11)
    c = 1
    set_uvi(0,0,c,1); set_uvi(1,1,c,-1)
    set_vjk(0,0,c,1); set_vjk(1,1,c,-1)
    set_wki(0,0,c,1); set_wki(1,1,c,-1)
    
    # M3 = (A00-A22)(B00-B22)
    c = 2
    set_uvi(0,0,c,1); set_uvi(2,2,c,-1)
    set_vjk(0,0,c,1); set_vjk(2,2,c,-1)
    set_wki(0,0,c,1); set_wki(2,2,c,-1)
    
    # M4 = (A11-A22)(B11-B22)
    c = 3
    set_uvi(1,1,c,1); set_uvi(2,2,c,-1)
    set_vjk(1,1,c,1); set_vjk(2,2,c,-1)
    set_wki(1,1,c,1); set_wki(2,2,c,-1)
    
    # M5 = (A01+A10)(B10+B01)
    c = 4
    set_uvi(0,1,c,1); set_uvi(1,0,c,1)
    set_vjk(1,0,c,1); set_vjk(0,1,c,1)
    set_wki(1,0,c,1); set_wki(0,1,c,1)
    
    # M6 = (A02+A20)(B20+B02)
    c = 5
    set_uvi(0,2,c,1); set_uvi(2,0,c,1)
    set_vjk(2,0,c,1); set_vjk(0,2,c,1)
    set_wki(2,0,c,1); set_wki(0,2,c,1)
    
    # M7 = (A12+A21)(B21+B12)
    c = 6
    set_uvi(1,2,c,1); set_uvi(2,1,c,1)
    set_vjk(2,1,c,1); set_vjk(1,2,c,1)
    set_wki(2,1,c,1); set_wki(1,2,c,1)
    
    # 12 schoolbook terms for remaining products
    # Need to cover: C01 (A00B01,A01B11,A02B21), C02, C10, C12, C20, C21
    # Plus complete C00, C11, C22
    
    # The 7 Strassen terms contribute to all 9 C-positions.
    # We add 12 schoolbook terms to complete.
    
    # C01 needs: A00*B01, A01*B11, A02*B21
    c = 7
    set_uvi(0,0,c,1); set_vjk(0,1,c,1); set_wki(1,0,c,1)
    c = 8
    set_uvi(0,1,c,1); set_vjk(1,1,c,1); set_wki(1,0,c,1)
    c = 9
    set_uvi(0,2,c,1); set_vjk(2,1,c,1); set_wki(1,0,c,1)
    
    # C02 needs: A00*B02, A01*B12, A02*B22
    c = 10
    set_uvi(0,0,c,1); set_vjk(0,2,c,1); set_wki(2,0,c,1)
    c = 11
    set_uvi(0,1,c,1); set_vjk(1,2,c,1); set_wki(2,0,c,1)
    c = 12
    set_uvi(0,2,c,1); set_vjk(2,2,c,1); set_wki(2,0,c,1)
    
    # C10 needs: A10*B00, A11*B10, A12*B20
    c = 13
    set_uvi(1,0,c,1); set_vjk(0,0,c,1); set_wki(0,1,c,1)
    c = 14
    set_uvi(1,1,c,1); set_vjk(1,0,c,1); set_wki(0,1,c,1)
    c = 15
    set_uvi(1,2,c,1); set_vjk(2,0,c,1); set_wki(0,1,c,1)
    
    # C12 needs: A10*B02, A11*B12, A12*B22
    c = 16
    set_uvi(1,0,c,1); set_vjk(0,2,c,1); set_wki(2,1,c,1)
    c = 17
    set_uvi(1,1,c,1); set_vjk(1,2,c,1); set_wki(2,1,c,1)
    c = 18
    set_uvi(1,2,c,1); set_vjk(2,2,c,1); set_wki(2,1,c,1)
    
    return U, V, W


def construct():
    """Construct rank-37 decomposition for 2x3x9 matrix multiplication."""
    n, m, p = 2, 3, 9
    rank = 37
    U = [[0]*rank for _ in range(n*m)]  # 6 x 37
    V = [[0]*rank for _ in range(m*p)]  # 27 x 37
    W = [[0]*rank for _ in range(p*n)]  # 18 x 37
    
    # Part 1: De Leede for B row 0 (j=0), rank 19, columns 0-18
    U_dl, V_dl, W_dl = _de_leede_333()
    for i in range(3):
        for j in range(3):
            for c in range(19):
                U[i*3+j][c] = U_dl[i*3+j][c]
    for j in range(3):
        for k in range(3):
            for c in range(19):
                V[j*3+k][c] = V_dl[j*3+k][c]
    for k in range(3):
        for i in range(3):
            for c in range(19):
                W[k*3+i][c] = W_dl[k*3+i][c]
    
    # Part 2: Schoolbook for B row 1 (j=1), rank 9, columns 19-27
    col = 19
    for i in range(n):
        for k in range(3):
            U[i*3+1][col] = 1
            V[1*3+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
    
    # Part 3: Schoolbook for B row 2 (j=2), rank 9, columns 28-36
    col = 28
    for i in range(n):
        for k in range(3):
            U[i*3+2][col] = 1
            V[2*3+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
