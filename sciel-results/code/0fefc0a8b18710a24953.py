"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block decomposition: treat 6x6 as 2x2 blocks of 3x3 matrices
    # Winograd for 2x2: rank 7 instead of 8
    # Each term: 3x3 matrix multiplication, rank 27
    # Total rank: 7 * 27 = 189
    
    block_size = 3
    num_blocks = 2
    rank_per_block_mult = block_size ** 3  # 27
    winograd_rank = 7  # Winograd 2x2
    total_rank = winograd_rank * rank_per_block_mult  # 189
    
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189
    
    # Winograd 2x2 decomposition:
    # Given blocks A00, A01, A10, A11 and B00, B01, B10, B11
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Winograd uses 7 products:
    # P1 = A00 * B00
    # P2 = A01 * B10
    # P3 = A10 * B01
    # P4 = A11 * B11
    # P5 = (A00+A01+A10+A11) * (B00+B01+B10+B11)  -- wait, this isn't standard
    #
    # Actually, let me use the standard Winograd 2x2:
    # m1 = A00 + A10
    # m2 = A00 + A01
    # m3 = A11 + A10
    # m4 = A11 + A01
    # p1 = m1 * (B00 + B10)
    # p2 = m2 * (B01 + B11)
    # p3 = m3 * (B10 + B11)
    # p4 = m4 * (B01 + B10)
    # p5 = A00 * B00
    # p6 = A01 * B11
    # p7 = A11 * B11  -- hmm, this doesn't look right
    #
    # Let me look up the correct Winograd 2x2 formula.
    # The standard Winograd for 2x2 matrix multiplication uses 7 multiplications:
    # 
    # For C = A * B where A, B are 2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    #
    # Winograd:
    # t1 = a00 * b00
    # t2 = a01 * b10
    # t3 = a10 * b01
    # t4 = a11 * b11
    # t5 = (a00 + a10) * (b00 + b10)
    # t6 = (a00 + a01) * (b01 + b11)
    # t7 = (a10 + a11) * (b10 + b11)
    #
    # C00 = t1 + t5 - t6 + t7 - t4  -- no, let me verify
    #
    # Actually, the correct Winograd 2x2 is:
    # m1 = a00
    # m2 = a01
    # m3 = a10
    # m4 = a11
    # m5 = a00 + a10
    # m6 = a00 + a01
    # m7 = a10 + a11
    #
    # p1 = b00
    # p2 = b10
    # p3 = b01
    # p4 = b11
    # p5 = b00 + b10
    # p6 = b01 + b11
    # p7 = b10 + b11
    #
    # Products:
    # q1 = m1 * p1 = a00*b00
    # q2 = m2 * p2 = a01*b10
    # q3 = m3 * p3 = a10*b01
    # q4 = m4 * p4 = a11*b11
    # q5 = m5 * p5 = (a00+a10)*(b00+b10)
    # q6 = m6 * p6 = (a00+a01)*(b01+b11)
    # q7 = m7 * p7 = (a10+a11)*(b10+b11)
    #
    # C00 = q1 + q5 - q6 + q7 - q4  -- let me verify:
    # q1 + q5 - q6 + q7 - q4
    # = a00*b00 + (a00+a10)*(b00+b10) - (a00+a01)*(b01+b11) + (a10+a11)*(b10+b11) - a11*b11
    # This is getting complex. Let me just use a known correct Winograd formulation.
    
    # I'll use a simpler approach: just use the schoolbook for now but with a better structure.
    # Actually, let me try a different known rank-23 for 3x3 and use block structure.
    
    # For 6x6, let me try: decompose into 3x3 blocks (4 blocks), use rank-23 for 3x3
    # 2x2 block structure with 3x3 blocks
    # If 3x3 has rank 23, then 2x2 block mult with 3x3 blocks:
    # Each of the 8 schoolbook terms becomes a 3x3 mult of rank 23
    # Total: 8 * 23 = 184
    # But we can use Winograd for the 2x2 block structure: 7 * 23 = 161
    
    # However, I need to verify that rank-23 for 3x3 has integer/half-integer entries.
    # The best known rank-23 decomposition for 3x3 may not have rational entries.
    
    # Let me be safe and use schoolbook for 3x3 (rank 27) with Winograd 2x2 (rank 7)
    # Total: 7 * 27 = 189
    
    # Winograd 2x2 coefficients (verified):
    # The 7 bilinear forms for 2x2 matrix multiplication:
    
    # I'll use this known correct Winograd 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Winograd 7 products:
    # p1 = a*e
    # p2 = b*g
    # p3 = c*f
    # p4 = d*h
    # p5 = (a+c)*(e+g)
    # p6 = (a+b)*(f+h)
    # p7 = (c+d)*(g+h)
    #
    # C00 = p1 + p5 - p6 + p7 - p4  -- let me expand to verify
    # p1 + p5 - p6 + p7 - p4
    # = ae + (a+c)(e+g) - (a+b)(f+h) + (c+d)(g+h) - dh
    # = ae + ae + ag + ce + cg - af - ah - bf - bh + cg + ch + dg + dh - dh
    # = 2ae + ag + ce + 2cg - af - ah - bf - bh + ch + dg
    # This doesn't look like ae + bg. So this formulation is wrong.
    
    # Let me use a different, verified Winograd 2x2:
    # From literature, the 7 products are:
    # m1 = a
    # m2 = b
    # m3 = c
    # m4 = d
    # m5 = a + c
    # m6 = a + b
    # m7 = c + d
    #
    # n1 = e
    # n2 = g
    # n3 = f
    # n4 = h
    # n5 = e + g
    # n6 = f + h
    # n7 = g + h
    #
    # Products:
    # q1 = m1*n1 = a*e
    # q2 = m2*n2 = b*g
    # q3 = m3*n3 = c*f
    # q4 = m4*n4 = d*h
    # q5 = m5*n5 = (a+c)*(e+g)
    # q6 = m6*n6 = (a+b)*(f+h)
    # q7 = m7*n7 = (c+d)*(g+h)
    #
    # C00 = q1 + q5 - q6 + q7 - q4  -- same as before, let me check differently
    #
    # Actually, I think the correct reconstruction is:
    # C00 = q1 + q5 - q6 + q7 - q4 - q2 - q3 + ... this is getting messy.
    
    # Let me just use the straightforward schoolbook for 6x6 but try to find any improvement.
    # Given the complexity, let me stick with a clean schoolbook implementation.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
