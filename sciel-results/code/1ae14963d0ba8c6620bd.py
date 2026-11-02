"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT-based decomposition for 2x4 x 4x7 matrix multiplication.
    
    We use the Winograd Fourier Transform Technique over Z_2 x Z_7.
    
    Key idea: The multiplication C = A*B where A is 2x4 and B is 4x7.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We view j as an element of Z_4, but we'll use a different approach.
    
    Alternative approach using the structure:
    Since n=2 is small, we can use the fact that for each column k of B,
    we need to compute two linear combinations of the rows of B.
    
    Let's use a direct WFT construction over Z_2 x Z_7.
    
    Actually, let me try a cleaner approach:
    
    For 2x4 * 4x7, we can use the following decomposition:
    
    Think of A as a 2x4 matrix and B as a 4x7 matrix.
    
    Using the Winograd scheme for the 2x2 case and extending:
    
    For 2x2 * 2x2, the Winograd algorithm uses 5 multiplications:
    m0 = (A00 + A11) * (B00 + B11)
    m1 = (A00 + A10) * (B00 + B10)  
    m2 = (A10 + A11) * (B01 + B11)
    m3 = A00 * B00
    m4 = A11 * B11
    
    Then:
    C00 = m0 - m3 - m1 + m4... wait, let me get the standard Winograd right.
    
    Standard Winograd for 2x2:
    m0 = (A00 + A01) * (B00 + B10)
    m1 = (A00 + A01) * (B01 + B11)
    m2 = (A10 + A11) * (B00 + B10)
    m3 = (A10 + A11) * (B01 + B11)
    m4 = (A00 - A10 - A11) * (B00 - B01 + B11)
    
    C00 = m0 + m4
    C01 = m1 - m4
    C10 = m2 - m4
    C11 = m3 + m4
    
    Now for 2x4 * 4x7, let's split the middle dimension:
    A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    
    C = A_left * B_top + A_right * B_bottom
    
    For each 2x2 * 2x7 multiplication, we can use a Winograd-like scheme.
    
    For 2x2 * 2x7:
    A_sub = [[a00, a01], [a10, a11]]
    B_sub = [[b00, b01, ..., b06], [b10, b11, ..., b16]]
    
    C_sub[i,k] = a_i0 * b_0k + a_i1 * b_1k
    
    Using Winograd for 2x2 * 2x7:
    We can use 2*7 = 14 multiplications (schoolbook for this subcase)
    or potentially fewer with Winograd.
    
    Actually, for 2x2 * 2xp, the Winograd scheme uses 2p + 2 multiplications:
    For each column k:
    m0k = (a00 + a01) * b0k
    m1k = (a10 + a11) * b1k
    m2k = (a00 - a10 - a11) * (b0k - b1k)
    
    Wait, that's not quite right. Let me think again.
    
    For 2x2 * 2x2, Winograd uses 5 mults. For 2x2 * 2xp, it should use 2p + 2.
    
    Let me verify: For 2x2 * 2x2, 2*2+2 = 6, but Winograd uses 5. So the formula is different.
    
    Let me just use the direct Winograd extension:
    
    For 2x2 * 2xp:
    Let u0 = a00 + a01, u1 = a10 + a11, u2 = a00 - a10 - a11
    For each k in 0..p-1:
      v_k = b0k + b1k
      w_k = b0k - b1k
    
    Then:
    C0k = a00*b0k + a01*b1k
    C1k = a10*b0k + a11*b1k
    
    Using the Winograd identities:
    C0k = (u0*b0k + u0*b1k - u2*w_k)/2 ... this is getting complicated.
    
    Let me try a different approach. Since the task allows half-integer entries,
    I'll construct a decomposition that may use 1/2 factors.
    
    For 2x4 * 4x7, let's use the following WFT-based approach:
    
    Group G = Z_2 x Z_7, order 14.
    
    The multiplication tensor has rank at most n*p*(m) in schoolbook = 56.
    
    Let me try to achieve rank 28 (the lower bound max(8,28,14)=28).
    
    Actually, let me just implement a clean schoolbook decomposition first,
    then try to optimize.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 56
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
