"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _winograd_2x2():
    """
    Returns the rank-7 Winograd decomposition for 2x2 matrix multiplication.
    
    For A = [[a, b], [c, d]] and B = [[e, f], [g, h]]:
    C = A*B = [[ae+cg, af+dh], [ce+dg, cf+eh]]
    
    The 7 intermediate products are:
    m1 = (a+b)(e+h)
    m2 = (c+d)(e+f)
    m3 = a(f-h)
    m4 = d(g-f)
    m5 = (a-c)(e+g)
    m6 = (b+d)(g+h)
    m7 = (b+c)(e+f)  -- wait, let me use the standard form
    
    Actually, using the well-known rank-7 scheme:
    m1 = a*(e)
    m2 = b*(f)
    m3 = c*(g)
    m4 = d*(h)
    m5 = (a+b)*(e+f)
    m6 = (c+d)*(g+h)
    m7 = (a+c)*(e+g)
    
    C[0,0] = m1 + m3 - m5 + m7
    C[0,1] = m2 + m4 - m6 + m7  -- this isn't right either
    
    Let me use a verified rank-7 decomposition:
    m1 = (a+b)(e+h)
    m2 = (c+d)(e+f)
    m3 = a(f-h)
    m4 = d(g-f)
    m5 = (a-c)(e+g)
    m6 = (b+d)(g+h)
    m7 = (b+c)(f+e)
    
    C[0,0] = m1 - m2 + m5 - m3 - m7
    C[0,1] = m1 - m6 + m4 + m7
    C[1,0] = m2 + m5 + m3 + m7
    C[1,1] = m2 - m1 + m6 - m4 - m7
    
    Let me verify C[0,0]:
    m1 - m2 + m5 - m3 - m7
    = (a+b)(e+h) - (c+d)(e+f) + (a-c)(e+g) - a(f-h) - (b+c)(f+e)
    = ae+ah+be+bh - ce-cf-de-df + ae+ag-ce-cg - af+ah - bf-be-cf-ce
    = ae+ah+be+bh - ce-cf-de-df + ae+ag-ce-cg - af+ah - bf-be-cf-ce
    Collecting terms:
    ae: 1+1 = 2
    ah: 1+1 = 2
    be: 1-1 = 0
    bh: 1
    ce: -1-1-1 = -3
    cf: -1-1 = -2
    de: -1
    df: -1
    ag: 1
    cg: -1
    af: -1
    bf: -1
    This is getting messy. Let me use a different verified approach.
    """
    pass

def construct():
    """
    Tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    
    Uses Strassen's algorithm structure: split 4x4 into 2x2 blocks,
    use rank-7 Winograd for each 2x2*2x2 subproblem.
    
    For 4x4 * 4x8:
    - A is 4x4, split into 4 blocks of 2x2: A00, A01, A10, A11
    - B is 4x8, split into 2 blocks of 2x8: B_top (2x8), B_bottom (2x8)
    - C is 8x4, which we need to handle carefully
    
    Actually, let's think of it as:
    C = A * B where A is 4x4 and B is 4x8
    
    Split A into 2x2 blocks:
    A = [[A00, A01],
         [A10, A11]]
    
    Split B into 2x8 blocks (top 2 rows, bottom 2 rows):
    B = [[B0],     (2x8)
         [B1]]     (2x8)
    
    C = [[A00*B0 + A01*B1],   (2x8)
         [A10*B0 + A11*B1]]   (2x8)
    
    Each 2x2 * 2x8 multiplication can be done with rank 7*8 = 56?
    No, that's not right. A 2x2 * 2x8 is like 8 separate 2x2 * 2x1 multiplications.
    
    For 2x2 * 2x1: rank 7 (Winograd)
    For 2x2 * 2x8: we can do 8 independent 2x2*2x1, rank = 8*7 = 56
    
    Total: 4 subproblems (A00*B0, A01*B1, A10*B0, A11*B1)
    But A00*B0 and A10*B0 both use B0, so we can share?
    
    Actually, the total rank would be 4 * 56 = 224, which is worse than 128.
    
    Let me just use schoolbook for now but try to optimize.
    
    Schoolbook: rank = 4*4*8 = 128
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    rank = n * m * p  # 128
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 128
    V = [[0] * rank for _ in range(m * p)]  # 32 x 128
    W = [[0] * rank for _ in range(p * n)]  # 32 x 128
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
