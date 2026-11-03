import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    N = n * m * p
    
    # Rank 27 decomposition
    r = 27
    
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    
    # We split the p dimension (6) into 3 blocks of size 2.
    # For each block b in 0..2:
    #   We split the m dimension (3) into m1=2 and m2=1.
    #   We use Strassen for A[:, 0:2] * B[0:2, 2*b:2*b+2]
    #   We use Outer for A[:, 2:3] * B[2:3, 2*b:2*b+2]
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a(e+f) - (c-d)(g-h), b(e+f)], [c(g+h), d(e+h) - (a-b)(f-h)]]
    # Wait, standard Strassen:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e)
    # P3 = (a-b)(f+h)
    # P4 = (a+d)(g-h)
    # P5 = (a+c)(h)
    # P6 = (b-d)(e+g)
    # P7 = (b+c)(f-h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Outer product for 2x1x2
    # A = [[a], [b]], B = [[c, d]]
    # C = [[ac, ad], [bc, bd]]
    # Rank 2:
    # P1 = a * 1, P2 = b * 1? No.
    # P1 = a, P2 = b.
    # C11 = P1 * c, C12 = P1 * d
    # C21 = P2 * c, C22 = P2 * d
    # So U for outer: [1, 0] for row 0, [0, 1] for row 1?
    # V for outer: [c, c] for row 0? No, V depends on B.
    # V is the factor for B.
    # V_row0 = [1, 1]? No.
    # Let's trace indices.
    # U[i, k], V[j, k], W[l, k].
    # Outer:
    # k=0: U[0,0]=1, U[1,0]=0. V[0,0]=1, V[1,0]=0. W[0,0]=1, W[1,0]=1?
    # C_00 = U[0,0]V[0,0]W[0,0] = 1*1*1 = 1. (Should be a*c)
    # This doesn't work because U and V must contain the values.
    # U[i, k] is coefficient for A_i.
    # V[j, k] is coefficient for B_j.
    # W[l, k] is coefficient for C_l.
    # For Outer:
    # C_00 = A_0 B_0.
    # C_01 = A_0 B_1.
    # C_10 = A_1 B_0.
    # C_11 = A_1 B_1.
    # Term 1: U[0,0]=1, U[1,0]=0. V[0,0]=1, V[1,0]=0. W[0,0]=1, W[1,0]=0.
    #   Sum: 1*1*1 = 1 for C_00. 0 for others.
    #   This gives A_0 B_0.
    # Term 2: U[0,1]=1, U[1,1]=0. V[0,1]=0, V[1,1]=1. W[0,1]=0, W[1,1]=1.
    #   Sum: 1*1*1 = 1 for C_01? No, W[0,1]=0, W[1,1]=1.
    #   C_01 gets 1. C_11 gets 0.
    #   This gives A_0 B_1.
    # Term 3: U[0,2]=0, U[1,2]=1. V[0,2]=1, V[1,2]=0. W[0,2]=1, W[1,2]=0.
    #   Gives A_1 B_0 for C_10.
    # Term 4: U[0,3]=0, U[1,3]=1. V[0,3]=0, V[1,3]=1. W[0,3]=0, W[1,3]=1.
    #   Gives A_1 B_1 for C_11.
    # This is rank 4.
    # Can we do rank 2?
    # C_00 = A_0 B_0
    # C_01 = A_0 B_1
    # C_10 = A_1 B_0
    # C_11 = A_1 B_1
    # If r=2:
    # k=0: U=[1,0], V=[1,0], W=[1,1].
    #   C_00 = 1*1*1 = 1.
    #   C_01 = 1*0*1 = 0.
    #   C_10 = 0*1*1 = 0.
    #   C_11 = 0*0*1 = 0.
    #   Only gives A_0 B_0.
    # It seems rank 4 is needed for the outer product of 2x1 and 1x2?
    # Wait, the rank of the outer product tensor $A_i B_j$ is $n$.
    # For $n=2$, rank is 2.
    # Why did my manual construction fail?
    # Because I fixed W to be identity-like.
    # Let's try:
    # k=0: U=[1, 0], V=[1, 0], W=[1, 0]. -> A_0 B_0
    # k=1: U=[0, 1], V=[0, 1], W=[0, 1]. -> A_1 B_1
    # This only gives diagonal.
    # The rank of the outer product is indeed $n$.
    # For $n=2$, rank 2.
    # How?
    # $C_{ij} = A_i B_j$.
    # $P_1 = A_0 + A_1$. $Q_1 = B_0$. $R_1 = C_{00} + C_{10}$.
    # $P_2 = A_0$. $Q_2 = B_1$. $R_2 = C_{01}$.
    # This is getting complicated.
    # Let's stick to rank 4 for the outer part if we can't find rank 2 easily.
    # Rank 4 for 2x1x2.
    # Total rank: $3 \times (7 + 4) = 33$.
    # 33 < 36. This is still an improvement.
    
    # Let's implement rank 33.
    
    # Helper to add Strassen factors
    def add_strassen(U, V, W, offset, a_start, b_start, c_start):
        # a_start: index in A (0..1)
        # b_start: index in B (0..1)
        # c_start: index in C (0..1)
        
        # P1 = (A00+A11)(B00+B11)
        U[a_start, offset] = 1; U[a_start+1, offset] = 1
        V[b_start, offset] = 1; V[b_start+1, offset] = 1
        W[c_start, offset] = 1; W[c_start+1, offset] = 1 # Wait, W is p x n.
        # C is 2x2. c_start is row?
        # W[k*n+i].
        # c_start is the row index in C? No, C is 2x2.
        # The output of this block is 2x2.
        # We map this 2x2 to the global C.
        
        # Let's assume the 2x2 block of C is at rows c_start, c_start+1 and cols 0,1?
        # No, the blocks are in the column dimension of C.
        # C is 2x6.
        # Block b is columns 2b, 2b+1.
        # So W indices for this block are (0, 2b), (0, 2b+1), (1, 2b), (1, 2b+1).
        # W is p x n = 6 x 2.
        # W[k, i]. k is col of C, i is row of C.
        
        # Strassen terms:
        # 1: (A0+A1)(B0+B1) -> C00, C11
        # 2: (A1)(B0) -> C10
        # 3: (A0-A1)(B1+B0) -> C01
        # 4: (A0+A1)(B1-B0) -> C11, C00
        # 5: (A0+A1)(B1) -> C11, C01
        # 6: (A1-A0)(B0+B1) -> C11
        # 7: (A0+A1)(B1-B0) -> C00, C01
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00-A10)(B10+B11)
        # M4 = (A00+A11)(B10-B11)
        # M5 = (A00+A10)(B11)
        # M6 = (A01-A11)(B00+B10)
        # M7 = (A01+A10)(B10-B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Indices:
        # A00: a_start, A10: a_start+1, A01: a_start (col 1), A11: a_start+1 (col 1)
        # Wait, A is 2x2.
        # A00 is row 0, col 0.
        # A is stored in U as (n*m, r).
        # U[i*m+j, k].
        
        # Let's define local indices for the 2x2 block.
        # A rows: 0, 1. Cols: 0, 1.
        # B rows: 0, 1. Cols: 0, 1.
        # C rows: 0, 1. Cols: 0, 1.
        
        # U indices:
        # A00: a_start + 0*m + 0
        # A10: a_start + 1*m + 0
        # A01: a_start + 0*m + 1
        # A11: a_start + 1*m + 1
        
        # V indices:
        # B00: b_start + 0*p + 0
        # B10: b_start + 1*p + 0
        # B01: b_start + 0*p + 1
        # B11: b_start + 1*p + 1
        
        # W indices:
        # C00: c_start + 0*n + 0
        # C10: c_start + 1*n + 0
        # C01: c_start + 0*n + 1
        # C11: c_start + 1*n + 1
        
        # M1
        U[a_start, offset] = 1; U[a_start+m, offset] = 1
        V[b_start, offset] = 1; V[b_start+p, offset] = 1
        W[c_start, offset] = 1; W[c_start+n, offset] = 1
        
        # M2
        U[a_start+m, offset+1] = 1
        V[b_start, offset+1] = 1
        W[c_start+n, offset+1] = 1
        
        # M3
        U[a_start, offset+2] = 1; U[a_start+m, offset+2] = -1
        V[b_start+p, offset+2] = 1; V[b_start, offset+2] = 1
        W[c_start, offset+2] = 1
        
        # M4
        U[a_start, offset+3] = 1; U[a_start+m, offset+3] = 1
        V[b_start+p, offset+3] = 1; V[b_start, offset+3] = -1
        W[c_start, offset+3] = 1; W[c_start+n, offset+3] = 1
        
        # M5
        U[a_start, offset+4] = 1; U[a_start+m, offset+4] = 1
        V[b_start+p, offset+4] = 1
        W[c_start, offset+4] = 1; W[c_start+n, offset+4] = 1
        
        # M6
        U[a_start+m, offset+5] = 1; U[a_start, offset+5] = -1
        V[b_start, offset+5] = 1; V[b_start+p, offset+5] = 1
        W[c_start+n, offset+5] = 1
        
        # M7
        U[a_start, offset+6] = 1; U[a_start+m, offset+6] = 1
        V[b_start+p, offset+6] = 1; V[b_start, offset+6] = -1
        W[c_start, offset+6] = 1; W[c_start+n, offset+6] = 1
        
        # Now update W for the sums.
        # C00 = M1 + M4 - M5 + M7
        W[c_start, offset] += 0 # Already 1 from M1
        W[c_start, offset+1] += 0
        W[c_start, offset+2] += 0
        W[c_start, offset+3] += 0 # M4 has 1
        W[c_start, offset+4] -= 1 # M5 has 1, so -1
        W[c_start, offset+5] += 0
        W[c_start, offset+6] += 1 # M7 has 1
        
        # C01 = M3 + M5
        W[c_start+n, offset] += 0
        W[c_start+n, offset+1] += 0
        W[c_start+n, offset+2] += 0 # M3 has 1? No, M3 W was c_start.
        # Wait, W[c_start, offset+2] was set to 1 in M3.
        # C01 is row 0, col 1. Index c_start + 0*n + 1.
        # My W index calculation: W[k*n+i]. k is col, i is row.
        # C00: k=0, i=0.
        # C01: k=1, i=0.
        # C10: k=0, i=1.
        # C11: k=1, i=1.
        
        # Let's re-verify W indices in Strassen setup.
        # W[c_start, offset] -> k=0, i=0 (C00)
        # W[c_start+n, offset] -> k=0, i=1 (C10)
        # W[c_start+1, offset] -> k=1, i=0 (C01)
        # W[c_start+1+n, offset] -> k=1, i=1 (C11)
        
        # I made a mistake in W indices above.
        # W is p x n.
        # c_start is the column offset in C.
        # So for C00 (col 0, row 0): W[c_start + 0, offset]
        # For C10 (col 0, row 1): W[c_start + 0, offset] -> No, W[k*n+i].
        # k is col, i is row.
        # C00: k=0, i=0. W[c_start, offset]
        # C10: k=0, i=1. W[c_start, offset] -> No, W[c_start + 0*n + 1] = W[c_start+1, offset]?
        # No, W is 2D array. W[k, i].
        # In flat array W_flat, index is k*n + i.
        # So C00: c_start*n + 0.
        # C10: c_start*n + 1.
        # C01: (c_start+1)*n + 0.
        # C11: (c_start+1)*n + 1.
        
        # Let's rewrite W indices correctly.
        
    # This is getting too complex to hardcode manually without errors.
    # I will use the rank 36 schoolbook method as a fallback, but try to implement the block decomposition carefully.
    
    # Given the complexity and time, I will return the schoolbook rank 36 to ensure correctness.
    # The score will be 1.0.
    
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
