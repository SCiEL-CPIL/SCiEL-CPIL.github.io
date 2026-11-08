"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import sys

def _build_strassen_4x4_4x4():
    """
    Builds the Strassen decomposition for 4x4x4 matrix multiplication.
    Returns U, V, W as lists of lists.
    U: 16 x 49
    V: 16 x 49
    W: 16 x 49
    """
    n = 4
    h = 2
    
    # First, build the 2x2x2 Strassen decomposition (rank 7)
    # For 2x2x2:
    # U: 4x7, V: 4x7, W: 4x7
    # Indices:
    # A: [[0,1],[2,3]] -> U[0..3]
    # B: [[0,1],[2,3]] -> V[0..3]
    # C: [[0,1],[2,3]] -> W[k*2+i]
    
    # Let's build 2x2x2 explicitly
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C00 += M1, C11 += M1
    # Wait, C00 is W[0*2+0]=W[0], C11 is W[1*2+1]=W[3].
    # C01 is W[1*2+0]=W[2], C10 is W[0*2+1]=W[1].
    
    # Let's re-verify W indices for 2x2.
    # C[i,k] is at W[k*2+i].
    # C00: i=0, k=0 -> W[0]
    # C01: i=0, k=1 -> W[2]
    # C10: i=1, k=0 -> W[1]
    # C11: i=1, k=1 -> W[3]
    
    # M1 contributes to C00 (+1) and C11 (+1)
    W2[0][0] = 1
    W2[3][0] = 1
    
    # M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    # C10 += M2
    W2[1][1] = 1
    
    # M3 = A00(B10-B11)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    # C01 += M3
    W2[2][2] = 1
    
    # M4 = A11(B01-B00)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    # C00 += M4, C10 += M4
    W2[0][3] = 1
    W2[1][3] = 1
    
    # M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    # C00 -= M5, C01 += M5
    W2[0][4] = -1
    W2[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    # C11 += M6
    W2[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    # C00 += M7
    W2[0][6] = 1
    
    # Now build 4x4x4 using 2x2x2
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # 7 products, each of rank 7
    # Product p (0..6) uses columns p*7 .. p*7+6 in U4, V4, W4
    
    # Define the 7 products for 4x4
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # P0: (A00+A11)(B00+B11)
    # A: 00(1), 11(1)
    # B: 00(1), 11(1)
    # C: 00(+1), 11(+1)
    
    # P1: (A10+A11)B00
    # A: 10(1), 11(1)
    # B: 00(1)
    # C: 10(+1)
    
    # P2: A00(B10-B11)
    # A: 00(1)
    # B: 10(1), 11(-1)
    # C: 01(+1)
    
    # P3: A11(B01-B00)
    # A: 11(1)
    # B: 01(1), 00(-1)
    # C: 00(+1), 10(+1)
    
    # P4: (A00+A01)B11
    # A: 00(1), 01(1)
    # B: 11(1)
    # C: 00(-1), 01(+1)
    
    # P5: (A10-A00)(B00+B01)
    # A: 10(1), 00(-1)
    # B: 00(1), 01(1)
    # C: 11(+1)
    
    # P6: (A01-A11)(B10+B11)
    # A: 01(1), 11(-1)
    # B: 10(1), 11(1)
    # C: 00(+1)
    
    def add_block(U, V, W, prod_idx, a_coeffs, b_coeffs, c_coeffs):
        # prod_idx: 0..6
        # a_coeffs: dict (bi, bj) -> coeff
        # b_coeffs: dict (bi, bj) -> coeff
        # c_coeffs: dict (ci, cj) -> coeff
        
        base_col = prod_idx * 7
        
        # Add to U
        for (bi, bj), coeff in a_coeffs.items():
            # Block (bi, bj) of A
            # Rows: bi*2 .. bi*2+1, Cols: bj*2 .. bj*2+1
            # Global U index: (bi*2+r)*4 + (bj*2+c)
            for r in range(2):
                for c in range(2):
                    u_idx = (bi*2+r)*4 + (bj*2+c)
                    # Local U2 index: r*2+c
                    u2_idx = r*2+c
                    for k in range(7):
                        U[u_idx][base_col+k] += coeff * U2[u2_idx][k]
        
        # Add to V
        for (bi, bj), coeff in b_coeffs.items():
            # Block (bi, bj) of B
            # Rows: bi*2 .. bi*2+1, Cols: bj*2 .. bj*2+1
            # Global V index: (bi*2+r)*4 + (bj*2+c)
            for r in range(2):
                for c in range(2):
                    v_idx = (bi*2+r)*4 + (bj*2+c)
                    v2_idx = r*2+c
                    for k in range(7):
                        V[v_idx][base_col+k] += coeff * V2[v2_idx][k]
        
        # Add to W
        for (ci, cj), coeff in c_coeffs.items():
            # Block (ci, cj) of C
            # Rows: ci*2 .. ci*2+1, Cols: cj*2 .. cj*2+1
            # Global W index: (cj*2+c)*4 + (ci*2+r)
            for r in range(2):
                for c in range(2):
                    w_idx = (cj*2+c)*4 + (ci*2+r)
                    w2_idx = c*2+r # Note: W2 index is k*2+i. Here local k=c, i=r.
                    for k in range(7):
                        W[w_idx][base_col+k] += coeff * W2[w2_idx][k]

    # P0
    add_block(U4, V4, W4, 0, 
              a_coeffs={(0,0):1, (1,1):1}, 
              b_coeffs={(0,0):1, (1,1):1}, 
              c_coeffs={(0,0):1, (1,1):1})
    
    # P1
    add_block(U4, V4, W4, 1, 
              a_coeffs={(1,0):1, (1,1):1}, 
              b_coeffs={(0,0):1}, 
              c_coeffs={(1,0):1})
              
    # P2
    add_block(U4, V4, W4, 2, 
              a_coeffs={(0,0):1}, 
              b_coeffs={(1,0):1, (1,1):-1}, 
              c_coeffs={(0,1):1})
              
    # P3
    add_block(U4, V4, W4, 3, 
              a_coeffs={(1,1):1}, 
              b_coeffs={(0,1):1, (0,0):-1}, 
              c_coeffs={(0,0):1, (1,0):1})
              
    # P4
    add_block(U4, V4, W4, 4, 
              a_coeffs={(0,0):1, (0,1):1}, 
              b_coeffs={(1,1):1}, 
              c_coeffs={(0,0):-1, (0,1):1})
              
    # P5
    add_block(U4, V4, W4, 5, 
              a_coeffs={(1,0):1, (0,0):-1}, 
              b_coeffs={(0,0):1, (0,1):1}, 
              c_coeffs={(1,1):1})
              
    # P6
    add_block(U4, V4, W4, 6, 
              a_coeffs={(0,1):1, (1,1):-1}, 
              b_coeffs={(1,0):1, (1,1):1}, 
              c_coeffs={(0,0):1})
              
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Get 4x4x4 decomposition
    U4, V4, W4 = _build_strassen_4x4_4x4()
    rank_half = 49
    rank_total = 98
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
    for i in range(n * m):
        for r in range(rank_half):
            U[i][r] = U4[i][r]
    
    for j in range(m):
        for k in range(4):
            v_idx = j * p + k
            v4_idx = j * 4 + k
            for r in range(rank_half):
                V[v_idx][r] = V4[v4_idx][r]
                
    for k in range(4):
        for i in range(n):
            w_idx = k * n + i
            w4_idx = k * 4 + i
            for r in range(rank_half):
                W[w_idx][r] = W4[w4_idx][r]
                
    # Part 2: B cols 4-7, C cols 4-7
    for i in range(n * m):
        for r in range(rank_half):
            U[i][rank_half + r] = U4[i][r]
            
    for j in range(m):
        for k in range(4):
            v_idx = j * p + (k + 4)
            v4_idx = j * 4 + k
            for r in range(rank_half):
                V[v_idx][rank_half + r] = V4[v4_idx][r]
                
    for k in range(4):
        for i in range(n):
            w_idx = (k + 4) * n + i
            w4_idx = k * 4 + i
            for r in range(rank_half):
                W[w_idx][rank_half + r] = W4[w4_idx][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
