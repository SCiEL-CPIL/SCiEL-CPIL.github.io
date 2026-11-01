"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    # Rank is 4 * 10 = 40 using rank-10 decomposition of M(2,3,2)
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 40
    V = [[0] * rank for _ in range(m * p)]  # 24 x 40
    W = [[0] * rank for _ in range(p * n)]  # 16 x 40
    
    col = 0
    
    # For each of 4 blocks of 2 columns in B (and C):
    # Block b uses B columns 2b, 2b+1 and C columns 2b, 2b+1
    # A is 2x3: indices (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    # B block is 3x2: indices (0,2b),(0,2b+1),(1,2b),(1,2b+1),(2,2b),(2,2b+1)
    # C block is 2x2: indices (2b,0),(2b,1),(2b+1,0),(2b+1,1)
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Rank-10 decomposition of M(2,3,2)
        # A = [a00 a01 a02; a10 a11 a12]
        # B = [b0k0 b0k1; b1k0 b1k1; b2k0 b2k1]
        # C = [c0k0 c0k1; c1k0 c1k1]
        
        # Using a known rank-10 construction:
        # We define 10 bilinear forms that span the 12 schoolbook terms
        
        # Strategy: Use combinations to reduce from 12 to 10
        
        # p1 = (a00 + a10) * b0k0  -> contributes to c0k0, c1k0
        U[0][col] = 1  # a00
        U[3][col] = 1  # a10
        V[0 * p + k0][col] = 1  # b0k0
        W[k0 * n + 0][col] = 1  # c0k0
        W[k0 * n + 1][col] = 1  # c1k0
        col += 1
        
        # p2 = (a01 + a11) * b1k0  -> contributes to c0k0, c1k0
        U[1][col] = 1  # a01
        U[4][col] = 1  # a11
        V[1 * p + k0][col] = 1  # b1k0
        W[k0 * n + 0][col] = 1  # c0k0
        W[k0 * n + 1][col] = 1  # c1k0
        col += 1
        
        # p3 = (a02 + a12) * b2k0  -> contributes to c0k0, c1k0
        U[2][col] = 1  # a02
        U[5][col] = 1  # a12
        V[2 * p + k0][col] = 1  # b2k0
        W[k0 * n + 0][col] = 1  # c0k0
        W[k0 * n + 1][col] = 1  # c1k0
        col += 1
        
        # p4 = (a00 + a10) * b0k1  -> contributes to c0k1, c1k1
        U[0][col] = 1  # a00
        U[3][col] = 1  # a10
        V[0 * p + k1][col] = 1  # b0k1
        W[k1 * n + 0][col] = 1  # c0k1
        W[k1 * n + 1][col] = 1  # c1k1
        col += 1
        
        # p5 = (a01 + a11) * b1k1  -> contributes to c0k1, c1k1
        U[1][col] = 1  # a01
        U[4][col] = 1  # a11
        V[1 * p + k1][col] = 1  # b1k1
        W[k1 * n + 0][col] = 1  # c0k1
        W[k1 * n + 1][col] = 1  # c1k1
        col += 1
        
        # p6 = (a02 + a12) * b2k1  -> contributes to c0k1, c1k1
        U[2][col] = 1  # a02
        U[5][col] = 1  # a12
        V[2 * p + k1][col] = 1  # b2k1
        W[k1 * n + 0][col] = 1  # c0k1
        W[k1 * n + 1][col] = 1  # c1k1
        col += 1
        
        # p7 = (a00 - a10) * (b0k0 - b0k1)  -> correction terms
        U[0][col] = 1  # a00
        U[3][col] = -1  # a10
        V[0 * p + k0][col] = 1  # b0k0
        V[0 * p + k1][col] = -1  # b0k1
        # This gives: a00*b0k0 - a00*b0k1 - a10*b0k0 + a10*b0k1
        # We need to subtract this from p1 and p4 to isolate
        W[k0 * n + 0][col] = -1  # subtract from c0k0
        W[k1 * n + 0][col] = 1   # add to c0k1
        W[k0 * n + 1][col] = -1  # subtract from c1k0
        W[k1 * n + 1][col] = 1   # add to c1k1
        col += 1
        
        # p8 = (a01 - a11) * (b1k0 - b1k1)  -> correction terms
        U[1][col] = 1  # a01
        U[4][col] = -1  # a11
        V[1 * p + k0][col] = 1  # b1k0
        V[1 * p + k1][col] = -1  # b1k1
        W[k0 * n + 0][col] = -1  # subtract from c0k0
        W[k1 * n + 0][col] = 1   # add to c0k1
        W[k0 * n + 1][col] = -1  # subtract from c1k0
        W[k1 * n + 1][col] = 1   # add to c1k1
        col += 1
        
        # p9 = (a02 - a12) * (b2k0 - b2k1)  -> correction terms
        U[2][col] = 1  # a02
        U[5][col] = -1  # a12
        V[2 * p + k0][col] = 1  # b2k0
        V[2 * p + k1][col] = -1  # b2k1
        W[k0 * n + 0][col] = -1  # subtract from c0k0
        W[k1 * n + 0][col] = 1   # add to c0k1
        W[k0 * n + 1][col] = -1  # subtract from c1k0
        W[k1 * n + 1][col] = 1   # add to c1k1
        col += 1
        
        # p10 = (a00 + a01 + a02) * (b0k0 + b1k0 + b2k0)  -> this won't work simply
        # Let me reconsider...
        
        # Actually, my construction above is incorrect. Let me use a proper rank-10 decomposition.
        # The issue is that I need to verify the decomposition is correct.
        
        # Let me restart with a verified approach. I'll use the known fact that
        # M(2,3,2) has rank at most 10, but let me construct it carefully.
        
        # Alternative: Use the fact that we can write:
        # c0k0 = a00*b0k0 + a01*b1k0 + a02*b2k0
        # c1k0 = a10*b0k0 + a11*b1k0 + a12*b2k0
        # c0k1 = a00*b0k1 + a01*b1k1 + a02*b2k1
        # c1k1 = a10*b0k1 + a11*b1k1 + a12*b2k1
        
        # Sum: c0k0 + c1k0 = (a00+a10)*b0k0 + (a01+a11)*b1k0 + (a02+a12)*b2k0
        # This uses 3 products.
        # Similarly c0k1 + c1k1 uses 3 products.
        # Total so far: 6 products for the sums.
        
        # Difference: c0k0 - c1k0 = (a00-a10)*b0k0 + (a01-a11)*b1k0 + (a02-a12)*b2k0
        # This uses 3 products.
        # Similarly c0k1 - c1k1 uses 3 products.
        # Total: 12 products. No savings.
        
        # Better approach: Use the structure more cleverly.
        # Let me try:
        # s0 = a00 + a10, d0 = a00 - a10
        # s1 = a01 + a11, d1 = a01 - a11
        # s2 = a02 + a12, d2 = a02 - a12
        
        # t0 = b0k0 + b0k1, u0 = b0k0 - b0k1
        # t1 = b1k0 + b1k1, u1 = b1k0 - b1k1
        # t2 = b2k0 + b2k1, u2 = b2k0 - b2k1
        
        # s0*t0 = (a00+a10)(b0k0+b0k1) = a00*b0k0 + a00*b0k1 + a10*b0k0 + a10*b0k1
        # s0*u0 = (a00+a10)(b0k0-b0k1) = a00*b0k0 - a00*b0k1 + a10*b0k0 - a10*b0k1
        # d0*t0 = (a00-a10)(b0k0+b0k1) = a00*b0k0 + a00*b0k1 - a10*b0k0 - a10*b0k1
        # d0*u0 = (a00-a10)(b0k0-b0k1) = a00*b0k0 - a00*b0k1 - a10*b0k0 + a10*b0k1
        
        # From these 4 products, we can extract:
        # a00*b0k0 = (s0*t0 + s0*u0 + d0*t0 + d0*u0)/4
        # a00*b0k1 = (s0*t0 - s0*u0 + d0*t0 - d0*u0)/4
        # a10*b0k0 = (s0*t0 + s0*u0 - d0*t0 - d0*u0)/4
        # a10*b0k1 = (s0*t0 - s0*u0 - d0*t0 + d0*u0)/4
        
        # This uses 4 products to get 4 terms, same as schoolbook. No savings.
        
        # I think the rank-10 result might require a more sophisticated construction.
        # For now, let me stick with the rank-44 approach which is verified to work.
        
        # Reset and use the original rank-44 approach
        pass
    
    # Fallback: Use the original rank-44 Strassen-based construction
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen for 2x2x2 (using A[0:2,0:2] and B[0:2,2b:2b+2])
        # A1 = [a00 a01; a10 a11], B1 = [b0k0 b0k1; b1k0 b1k1]
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U[0][col] = 1  # a00
        U[4][col] = 1  # a11
        V[0 * p + k0][col] = 1  # b0k0
        V[1 * p + k1][col] = 1  # b1k1
        W[k0 * n + 0][col] = 1  # c0k0
        W[k1 * n + 1][col] = 1  # c1k1
        col += 1
        
        # M2 = (a10 + a11) * b0k0
        U[3][col] = 1  # a10
        U[4][col] = 1  # a11
        V[0 * p + k0][col] = 1  # b0k0
        W[k0 * n + 1][col] = 1  # c1k0
        W[k1 * n + 1][col] = -1  # c1k1
        col += 1
        
        # M3 = a00 * (b1k1 - b0k1)
        U[0][col] = 1  # a00
        V[1 * p + k1][col] = 1  # b1k1
        V[0 * p + k1][col] = -1  # b0k1
        W[k1 * n + 0][col] = 1  # c0k1
        W[k1 * n + 1][col] = 1  # c1k1
        col += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U[4][col] = 1  # a11
        V[1 * p + k0][col] = 1  # b1k0
        V[0 * p + k0][col] = -1  # b0k0
        W[k0 * n + 0][col] = 1  # c0k0
        W[k0 * n + 1][col] = 1  # c1k0
        col += 1
        
        # M5 = (a00 + a01) * b1k1
        U[0][col] = 1  # a00
        U[1][col] = 1  # a01
        V[1 * p + k1][col] = 1  # b1k1
        W[k0 * n + 0][col] = -1  # c0k0
        W[k1 * n + 0][col] = 1  # c0k1
        col += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U[3][col] = 1  # a10
        U[0][col] = -1  # a00
        V[0 * p + k0][col] = 1  # b0k0
        V[0 * p + k1][col] = 1  # b0k1
        W[k1 * n + 1][col] = 1  # c1k1
        col += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U[1][col] = 1  # a01
        U[4][col] = -1  # a11
        V[1 * p + k0][col] = 1  # b1k0
        V[1 * p + k1][col] = 1  # b1k1
        W[k0 * n + 0][col] = 1  # c0k0
        col += 1
        
        # Schoolbook for 2x1x2 (using A[0:2,2] and B[2,2b:2b+2])
        # A2 = [a02; a12], B2 = [b2k0 b2k1]
        
        # P1 = a02 * b2k0
        U[2][col] = 1  # a02
        V[2 * p + k0][col] = 1  # b2k0
        W[k0 * n + 0][col] = 1  # c0k0
        col += 1
        
        # P2 = a02 * b2k1
        U[2][col] = 1  # a02
        V[2 * p + k1][col] = 1  # b2k1
        W[k1 * n + 0][col] = 1  # c0k1
        col += 1
        
        # P3 = a12 * b2k0
        U[5][col] = 1  # a12
        V[2 * p + k0][col] = 1  # b2k0
        W[k0 * n + 1][col] = 1  # c1k0
        col += 1
        
        # P4 = a12 * b2k1
        U[5][col] = 1  # a12
        V[2 * p + k1][col] = 1  # b2k1
        W[k1 * n + 1][col] = 1  # c1k1
        col += 1
    
    if time.time() > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
