"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Output reconstruction
    # M00 = M1 + M4 - M5 + M7
    # M01 = M3 + M5
    # M10 = M2 + M4
    # M11 = M1 - M2 + M3 + M6
    
    # We have 4 sub-problems (s, t)
    for s in range(2):
        for t in range(2):
            k0 = 4 * s + 2 * t
            k1 = 4 * s + 2 * t + 1
            
            # --- Top part: A_top (cols 0,1) * B_top (rows 0,1) ---
            # A indices: 0, 1, 4, 5
            # B indices: 0*8+k0, 0*8+k1, 1*8+k0, 1*8+k1
            
            a00, a01, a10, a11 = 0, 1, 4, 5
            b00, b01, b10, b11 = 0*8+k0, 0*8+k1, 1*8+k0, 1*8+k1
            
            # M1
            U[a00][col] += 1; U[a11][col] += 1
            V[b00][col] += 1; V[b11][col] += 1
            W[k0*2+0][col] += 1; W[k1*2+0][col] += 0 # M00
            W[k0*2+1][col] += 0; W[k1*2+1][col] += 1 # M11
            col += 1
            
            # M2
            U[a10][col] += 1; U[a11][col] += 1
            V[b00][col] += 1
            W[k0*2+1][col] += 1 # M10
            col += 1
            
            # M3
            U[a00][col] += 1
            V[b01][col] += 1; V[b11][col] -= 1
            W[k1*2+0][col] += 1 # M01
            W[k1*2+1][col] += 1 # M11
            col += 1
            
            # M4
            U[a11][col] += 1
            V[b10][col] += 1; V[b00][col] -= 1
            W[k0*2+0][col] += 1 # M00
            W[k0*2+1][col] += 1 # M10
            col += 1
            
            # M5
            U[a00][col] += 1; U[a01][col] += 1
            V[b11][col] += 1
            W[k0*2+0][col] -= 1 # M00
            W[k1*2+0][col] += 1 # M01
            col += 1
            
            # M6
            U[a10][col] += 1; U[a00][col] -= 1
            V[b00][col] += 1; V[b01][col] += 1
            W[k1*2+1][col] += 1 # M11
            col += 1
            
            # M7
            U[a01][col] += 1; U[a11][col] -= 1
            V[b10][col] += 1; V[b11][col] += 1
            W[k0*2+0][col] += 1 # M00
            col += 1
            
            # --- Bottom part: A_bot (cols 2,3) * B_bot (rows 2,3) ---
            # A indices: 2, 3, 6, 7
            # B indices: 2*8+k0, 2*8+k1, 3*8+k0, 3*8+k1
            
            a02, a03, a12, a13 = 2, 3, 6, 7
            b20, b21, b30, b31 = 2*8+k0, 2*8+k1, 3*8+k0, 3*8+k1
            
            # M1 (bot)
            U[a02][col] += 1; U[a13][col] += 1
            V[b20][col] += 1; V[b31][col] += 1
            W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
            col += 1
            
            # M2 (bot)
            U[a12][col] += 1; U[a13][col] += 1
            V[b20][col] += 1
            W[k0*2+1][col] += 1
            col += 1
            
            # M3 (bot)
            U[a02][col] += 1
            V[b21][col] += 1; V[b31][col] -= 1
            W[k1*2+0][col] += 1
            W[k1*2+1][col] += 1
            col += 1
            
            # M4 (bot)
            U[a13][col] += 1
            V[b30][col] += 1; V[b20][col] -= 1
            W[k0*2+0][col] += 1
            W[k0*2+1][col] += 1
            col += 1
            
            # M5 (bot)
            U[a02][col] += 1; U[a03][col] += 1
            V[b31][col] += 1
            W[k0*2+0][col] -= 1
            W[k1*2+0][col] += 1
            col += 1
            
            # M6 (bot)
            U[a12][col] += 1; U[a02][col] -= 1
            V[b20][col] += 1; V[b21][col] += 1
            W[k1*2+1][col] += 1
            col += 1
            
            # M7 (bot)
            U[a03][col] += 1; U[a13][col] -= 1
            V[b30][col] += 1; V[b31][col] += 1
            W[k0*2+0][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
