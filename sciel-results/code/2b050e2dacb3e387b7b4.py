"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # B is split into 3 blocks of size 3x2.
    # For each block, we use a rank 11 decomposition for 2x3 by 3x2.
    # Total rank = 3 * 11 = 33.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(3):
        # b is the block index (0, 1, 2)
        # This block handles columns 2*b and 2*b+1 of B and C.
        
        # Strassen terms for the top-left 2x2 submatrix of the block
        # A submatrix: rows 0,1; cols 0,1
        # B submatrix: rows 0,1; cols 0,1 (local) -> Global cols 2*b, 2*b+1
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[0*3+0][col] = 1 # a00
        U[1*3+1][col] = 1 # a11
        V[0*6 + (2*b+0)][col] = 1 # b00
        V[1*6 + (2*b+1)][col] = 1 # b11
        # W for C00 and C11
        # C00 += M1
        W[(2*b+0)*2 + 0][col] = 1
        # C11 += M1
        W[(2*b+1)*2 + 1][col] = 1
        # C01 -= M1? No, Strassen formula:
        # C00 = M1 - M4 + M7
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2
        # Wait, standard Strassen for C = AB:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 + M7
        # C12 = M5 + M7
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2
        
        # My indices:
        # A11=a00, A12=a01, A21=a10, A22=a11
        # B11=b00, B12=b01, B21=b10, B22=b11
        
        # Let's re-map carefully.
        # Term 1: (a00+a11)(b00+b11)
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: C00+=1, C11+=1
        # (Already set above)
        
        col += 1
        
        # M2 = (a10+a11) * b00
        U[1*3+0][col] = 1 # a10
        U[1*3+1][col] = 1 # a11
        V[0*6 + (2*b+0)][col] = 1 # b00
        # W: C10+=1, C11+=1
        W[(2*b+0)*2 + 1][col] = 1
        W[(2*b+1)*2 + 1][col] = 1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[0*3+0][col] = 1 # a00
        V[0*6 + (2*b+1)][col] = 1 # b01
        V[1*6 + (2*b+1)][col] = -1 # b11
        # W: C11 -= 1
        W[(2*b+1)*2 + 1][col] = -1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[1*3+1][col] = 1 # a11
        V[1*6 + (2*b+0)][col] = 1 # b10
        V[0*6 + (2*b+0)][col] = -1 # b00
        # W: C00 -= 1, C10 += 1
        W[(2*b+0)*2 + 0][col] = -1
        W[(2*b+0)*2 + 1][col] = 1
        col += 1
        
        # M5 = (a00+a01) * b11
        U[0*3+0][col] = 1 # a00
        U[0*3+1][col] = 1 # a01
        V[1*6 + (2*b+1)][col] = 1 # b11
        # W: C01 += 1
        W[(2*b+1)*2 + 0][col] = 1
        col += 1
        
        # M6 = (a10-a00) * (b00+b01)
        U[1*3+0][col] = 1 # a10
        U[0*3+0][col] = -1 # a00
        V[0*6 + (2*b+0)][col] = 1 # b00
        V[0*6 + (2*b+1)][col] = 1 # b01
        # W: C10 += 1, C11 -= 1
        W[(2*b+0)*2 + 1][col] = 1
        W[(2*b+1)*2 + 1][col] = -1
        col += 1
        
        # M7 = (a01-a11) * (b10+b11)
        U[0*3+1][col] = 1 # a01
        U[1*3+1][col] = -1 # a11
        V[1*6 + (2*b+0)][col] = 1 # b10
        V[1*6 + (2*b+1)][col] = 1 # b11
        # W: C00 += 1, C01 += 1
        W[(2*b+0)*2 + 0][col] = 1
        W[(2*b+1)*2 + 0][col] = 1
        col += 1
        
        # Now the remaining 4 terms involving a02, a12 and b20, b21 (local)
        # Local A col 2 is global col 2.
        # Local B row 2 is global row 2.
        
        # M8 = a02 * b20
        U[0*3+2][col] = 1
        V[2*6 + (2*b+0)][col] = 1
        # W: C00 += 1
        W[(2*b+0)*2 + 0][col] = 1
        col += 1
        
        # M9 = a02 * b21
        U[0*3+2][col] = 1
        V[2*6 + (2*b+1)][col] = 1
        # W: C01 += 1
        W[(2*b+1)*2 + 0][col] = 1
        col += 1
        
        # M10 = a12 * b20
        U[1*3+2][col] = 1
        V[2*6 + (2*b+0)][col] = 1
        # W: C10 += 1
        W[(2*b+0)*2 + 1][col] = 1
        col += 1
        
        # M11 = a12 * b21
        U[1*3+2][col] = 1
        V[2*6 + (2*b+1)][col] = 1
        # W: C11 += 1
        W[(2*b+1)*2 + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
