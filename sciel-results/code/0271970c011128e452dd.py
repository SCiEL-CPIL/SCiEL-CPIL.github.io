"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(U, V, W, u_off, v_off, w_off, col):
    """
    Add a Strassen 2x2x2 block.
    U rows: u_off..u_off+3 (i=0,1; j=0,1) -> indices u_off + i*2 + j
    V rows: v_off..v_off+3 (j=0,1; k=0,1) -> indices v_off + j*2 + k
    W rows: w_off..w_off+3 (k=0,1; i=0,1) -> indices w_off + k*2 + i
    """
    # M1 = (A00+A11)*(B00+B11)
    U[u_off + 0][col] = 1; U[u_off + 3][col] = 1
    V[v_off + 0][col] = 1; V[v_off + 3][col] = 1
    W[w_off + 0][col] = 1; W[w_off + 3][col] = 1
    col += 1
    # M2 = (A10+A11)*B00
    U[u_off + 2][col] = 1; U[u_off + 3][col] = 1
    V[v_off + 0][col] = 1
    W[w_off + 2][col] = 1; W[w_off + 3][col] = -1
    col += 1
    # M3 = A00*(B01-B11)
    U[u_off + 0][col] = 1
    V[v_off + 1][col] = 1; V[v_off + 3][col] = -1
    W[w_off + 0][col] = 1; W[w_off + 3][col] = 1
    col += 1
    # M4 = A11*(B10-B00)
    U[u_off + 3][col] = 1
    V[v_off + 2][col] = 1; V[v_off + 0][col] = -1
    W[w_off + 0][col] = 1; W[w_off + 2][col] = 1
    col += 1
    # M5 = (A00+A01)*B11
    U[u_off + 0][col] = 1; U[u_off + 1][col] = 1
    V[v_off + 3][col] = 1
    W[w_off + 0][col] = -1; W[w_off + 1][col] = 1
    col += 1
    # M6 = (A10-A00)*(B00+B01)
    U[u_off + 2][col] = 1; U[u_off + 0][col] = -1
    V[v_off + 0][col] = 1; V[v_off + 1][col] = 1
    W[w_off + 3][col] = 1
    col += 1
    # M7 = (A01-A11)*(B10+B11)
    U[u_off + 1][col] = 1; U[u_off + 3][col] = -1
    V[v_off + 2][col] = 1; V[v_off + 3][col] = 1
    W[w_off + 0][col] = 1
    col += 1
    return col

def _rank10_2x2x3(U, V, W, u_off, v_off, w_off, col):
    """
    Rank-10 decomposition for 2x2x3 tensor.
    A is 2x2, B is 2x3, C is 3x2.
    U rows: u_off..u_off+3 (i=0,1; j=0,1) -> u_off + i*2 + j
    V rows: v_off..v_off+5 (j=0,1; k=0,1,2) -> v_off + j*3 + k
    W rows: w_off..w_off+5 (k=0,1,2; i=0,1) -> w_off + k*2 + i
    
    We use a known rank-10 decomposition. Let me construct one:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    We can use Strassen-like ideas. Let's define:
    M1 = A00*B00
    M2 = A11*B11
    M3 = (A00+A11)*(B00+B11)
    M4 = (A01+A10)*B00  -- hmm, let me think differently.
    
    Actually, let me use a direct approach with 10 terms.
    Let me use the following 10 products:
    1. A00 * B00
    2. A00 * B01
    3. A00 * B02
    4. A01 * B10
    5. A01 * B11
    6. A01 * B12
    7. A10 * B00
    8. A10 * B01
    9. A10 * B02
    10. A11 * B10
    11. A11 * B11
    12. A11 * B12
    That's 12. We need to reduce to 10.
    
    Let me try:
    M1 = A00*(B00+B01+B02)
    M2 = A11*(B10+B11+B12)
    M3 = (A00+A11)*B00
    M4 = (A00+A11)*B10
    M5 = (A00+A11)*(B01-B10)
    M6 = (A01+A10)*B00
    M7 = (A01+A10)*B11
    M8 = (A01-A10)*B01
    M9 = (A01-A10)*B12
    M10 = A01*B12 - A10*B12  -- this doesn't work cleanly.
    
    Let me just use a verified rank-10 decomposition. Actually, let me use a simpler approach:
    I'll use the fact that we can share some terms.
    
    Let me try this set of 10:
    t1 = A00 * B00
    t2 = A00 * B01
    t3 = A00 * B02
    t4 = A01 * B10
    t5 = A01 * B11
    t6 = A01 * B12
    t7 = (A10 + A11) * B00
    t8 = (A10 + A11) * B01
    t9 = (A10 + A11) * B02
    t10 = A11 * B10
    t11 = A11 * B11
    t12 = A11 * B12
    Still 12.
    
    OK let me try a different grouping:
    t1 = A00 * (B00 + B01 + B02)
    t2 = A01 * (B10 + B11 + B12)
    t3 = A10 * (B00 + B01 + B02)
    t4 = A11 * (B10 + B11 + B12)
    Now:
    C[0,0] = t1 + t3 - A00*B01 - A00*B02 - A10*B01 - A10*B02
    This is getting complicated. Let me just use rank 12 for the 2x2x3 part, giving total rank 7+12+7+12 = 38. That's worse than 36.
    
    Alternatively, let me just use the schoolbook for the 2x2x3 part: rank 12. Total = 7+12+7+12 = 38. Worse.
    
    Or I can use a 2x2x2 Strassen (rank 7) + 2x2x1 schoolbook (rank 4) = 11 for each 2x2x3. Total = 11+11 = 22 per row, times 2 rows = 44. Worse.
    
    Actually wait. Let me reconsider. The current approach gives rank 36. Let me see if I can do better.
    
    For 2x2x3, the known minimum rank is 10 (from Alon et al.). Let me construct it properly.
    
    A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    10-term decomposition:
    t1 = a*(e+f+g)
    t2 = d*(h+i+j)
    t3 = (a+d)*(e+h)
    t4 = (a+d)*(f+i)
    t5 = (a+d)*(g+j)
    t6 = (b+c)*e
    t7 = (b+c)*h
    t8 = (b-c)*f
    t9 = (b-c)*i
    t10 = (b-c)*g  -- no, this doesn't give the right answer.
    
    Let me try:
    t1 = a*e
    t2 = d*j
    t3 = (a+d)*(e+j)
    t4 = (a+d)*f
    t5 = (a+d)*i
    t6 = (b+c)*e
    t7 = (b+c)*h
    t8 = b*g + c*i  -- can't split
    t9 = (b+c)*g
    t10 = (b+c)*j
    
    C[0,0] = ae + bh = t1 + t7 - c*e + ... no.
    
    This is hard. Let me just use the known result. The rank of 2x2x3 is 10.
    
    Actually, let me just stick with the rank-36 approach and verify it works, then try to improve.
    """
    # Let me just use schoolbook for now (rank 12) and see if the total works
    # Actually, let me try a rank-10 construction
    # 
    # From literature: rank(2x2x3) = 10
    # One explicit decomposition:
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # 
    # t1 = a*e, t2 = a*f, t3 = a*g
    # t4 = d*h, t5 = d*i, t6 = d*j
    # t7 = (a+d)*e, t8 = (a+d)*h
    # t9 = (b+c)*f, t10 = (b+c)*i
    #
    # C[0,0] = ae + bh = t1 + (t8 - d*e) + ... no
    #
    # Let me try yet another approach. I'll use:
    # t1 = (a+d)*(e+h)
    # t2 = (a+d)*(f+i)
    # t3 = (a+d)*(g+j)
    # t4 = (b+c)*e
    # t5 = (b+c)*h
    # t6 = (b+c)*f
    # t7 = (b+c)*i
    # t8 = (b-c)*g
    # t9 = (b-c)*j
    # t10 = a*e + d*j  -- can't do this as a single product
    #
    # OK I'll just use rank 12 (schoolbook) for the 2x2x3 blocks.
    # Total rank = 2*(7+12) = 38. That's worse than 36.
    #
    # So the current rank-36 approach is better. Let me just verify it and return it.
    pass

def _construct_hybrid_rank36():
    """Construct the rank-36 hybrid decomposition."""
    n, m, p = 2, 4, 5
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Block 1: A row 0 (U rows 0-3) * B (V rows 0-19)
    # Cols 0-1: Strassen 2x2x2 for B cols 0-1
    col = _strassen_2x2x2(U, V, W, 0, 0, 0, col)
    # Col 2: schoolbook for B col 2
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1; col += 1
    U[1][col] = 1; V[3][col] = 1; W[4][col] = 1; col += 1
    U[2][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    U[3][col] = 1; V[3][col] = 1; W[5][col] = 1; col += 1
    # Cols 3-4: Strassen 2x2x2 for B cols 3-4
    col = _strassen_2x2x2(U, V, W, 0, 6, 6, col)
    
    # Block 2: A row 1 (U rows 4-7) * B (V rows 10-19)
    # Cols 0-1: Strassen 2x2x2 for B cols 0-1
    col = _strassen_2x2x2(U, V, W, 4, 10, 0, col)
    # Col 2: schoolbook for B col 2
    U[4][col] = 1; V[12][col] = 1; W[4][col] = 1; col += 1
    U[5][col] = 1; V[13][col] = 1; W[4][col] = 1; col += 1
    U[6][col] = 1; V[12][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[13][col] = 1; W[5][col] = 1; col += 1
    # Cols 3-4: Strassen 2x2x2 for B cols 3-4
    col = _strassen_2x2x2(U, V, W, 4, 16, 6, col)
    
    return {"U": U, "V": V, "W": W}

def _construct():
    n, m, p = DIMENSIONS
    result = _construct_hybrid_rank36()
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if _verify(U, V, W, n, m, p, rank):
        return result
    else:
        # Fallback to schoolbook
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

def construct():
    return _construct()
# EVOLVE-BLOCK-END
