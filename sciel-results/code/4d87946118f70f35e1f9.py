"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-36 block-Strassen decomposition for 2x4 * 4x5."""
    n, m, p = DIMENSIONS
    # A: (n, m) = (2, 4), B: (m, p) = (4, 5), C: (n, p) = (2, 5)
    # Split A into left/right blocks of width 2: A_L (2x2), A_R (2x2)
    # Split B into left/right blocks of width 2: B_L (2x5), B_R (2x5)
    # C = A_L @ B_L + A_R @ B_R
    #
    # For each 2x2 @ 2x5 multiplication, use Strassen-like 7-term decomposition
    # but we need to be careful: 2x2 @ 2x5 isn't square.
    #
    # Actually, let's use a known approach: 
    # The rank of 2x4 * 4x5 matrix multiplication.
    # Known: rank(2,4,5) <= 36 via block Strassen.
    #
    # Let's use the standard Strassen for 2x2 blocks.
    # A_L is 2x2, B_L is 2x5. We can split B_L into two 2x2 blocks and one 2x1,
    # but that's messy.
    #
    # Alternative: Use the fact that rank(m,n,p) <= rank(m,n,p') + rank(m,n,p-p')
    # when we can split the output.
    #
    # Better approach: Direct construction using Strassen on 2x2 subblocks.
    # 
    # Let me think about this differently. We have A (2x4) and B (4x5).
    # 
    # Method: Split A = [A1 A2] where A1, A2 are 2x2.
    # Split B = [B1; B2] where B1, B2 are 2x5.
    # C = A1*B1 + A2*B2
    #
    # For A1*B1: A1 is 2x2, B1 is 2x5.
    # We can use a 7-term decomposition for 2x2 @ 2x5? 
    # Actually, for M x N @ N x P, the Strassen method works when N is even.
    # Here N=2, so we can apply Strassen to get 7 terms for 2x2 @ 2x5.
    #
    # Strassen for (2x2) @ (2x5):
    # A1 = [[a, b], [c, d]], B1 = [[e, f, g, h, i], [j, k, l, m, o]]
    # 
    # Standard Strassen splits both into 2x2 blocks, but B1 is 2x5.
    # Let's split B1 into B1a (2x2) and B1b (2x3).
    # Then A1*B1 = A1*B1a + A1*B1b
    # A1*B1a: 2x2 @ 2x2, rank 7 (Strassen)
    # A1*B1b: 2x2 @ 2x3, rank ? 
    #
    # This is getting complex. Let me use a different known result.
    #
    # Known: The rank of matrix multiplication (2,4,5) is at most 36.
    # Reference: Coppersmith-Winograd gives bounds, but for small cases,
    # explicit constructions exist.
    #
    # Let me try a direct approach: enumerate a 36-term decomposition.
    # 
    # Actually, the simplest correct approach with rank 36:
    # Use Strassen for the two 2x2 @ 2x5 products, but we need to handle 2x5.
    #
    # For 2x2 @ 2x5: We can write this as 2x2 @ (2x2 + 2x3)
    # Or better: use the general Strassen for non-square.
    #
    # Let me use a different decomposition. 
    # rank(2,4,5) = rank(2,2,5) + rank(2,2,5) if we can do it independently.
    # rank(2,2,5): 2x2 @ 2x5. 
    # 
    # For 2x2 @ 2x5, we can use:
    # A = [[a,b],[c,d]], B = [[e,f,g,h,i],[j,k,l,m,o]]
    # C = [[ae+bj, af+bk, ag+bl, ah+bm, ai+bo],
    #      [ce+dj, cf+dk, cg+dl, ch+dm, ci+do]]
    #
    # We need rank(2,2,5). Known: rank(2,2,5) <= 18? No, that's too high.
    # Actually rank(2,2,5) = rank(2,2,5). 
    # 
    # Let me just use the schoolbook for the subproblems but combine smartly.
    #
    # Schoolbook for 2x2 @ 2x5: 2*2*5 = 20 terms. Two of them: 40 terms.
    # Strassen for 2x2 @ 2x2: 7 terms. 
    #
    # Let me try: Split B (4x5) into B_top (2x5) and B_bot (2x5).
    # A (2x4) = [A_left (2x2), A_right (2x2)]
    # C = A_left @ B_top + A_right @ B_bot
    #
    # For A_left @ B_top (2x2 @ 2x5): use a 14-term decomposition (2 * 7, 
    # treating it as two 2x2 @ 2x2 plus one 2x2 @ 2x1, but that's not clean).
    #
    # I'll use a known 36-term construction directly.
    #
    # Based on the block-Strassen approach that achieved rank 36:
    # We use 7 terms for each of the two 2x2 @ 2x2 blocks, plus cross terms.
    
    # Let me construct it explicitly.
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13]]
    # B = [[b00, b01, b02, b03, b04],
    #      [b10, b11, b12, b13, b14],
    #      [b20, b21, b22, b23, b24],
    #      [b30, b31, b32, b33, b34]]
    #
    # Split: A_L = [[a00,a01],[a10,a11]], A_R = [[a02,a03],[a12,a13]]
    # B_L = [[b00,b01,b02,b03,b04],[b10,b11,b12,b13,b14]] (top 2 rows)
    # B_R = [[b20,b21,b22,b23,b24],[b30,b31,b32,b33,b34]] (bottom 2 rows)
    #
    # C = A_L @ B_L + A_R @ B_R
    #
    # For A_L @ B_L (2x2 @ 2x5), I'll use a 18-term decomposition.
    # Actually, let me use the following: for 2x2 @ 2x5, split the 5 columns
    # into 2+2+1.
    # A_L @ B_L = A_L @ [B_L1, B_L2, B_L3] where B_L1 is 2x2, B_L2 is 2x2, B_L3 is 2x1.
    # A_L @ B_L1: 7 terms (Strassen)
    # A_L @ B_L2: 7 terms (Strassen)
    # A_L @ B_L3: 2 terms (schoolbook for 2x2 @ 2x1)
    # Total: 16 terms for A_L @ B_L
    # Similarly 16 for A_R @ B_R
    # Total: 32 terms. That's better than 36!
    #
    # Wait, but can we share terms? Probably not easily. Let me verify 32 works.
    #
    # Strassen for 2x2 @ 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    #
    # For 2x2 @ 2x1: A = [[a,b],[c,d]], B = [[e],[g]]
    # C1 = ae + bg, C2 = ce + dg. 2 terms schoolbook, or can we do better? 
    # Rank of 2x2x1 is 4 (each output element needs its own term since they're 
    # independent bilinear forms). Actually no: C1 = ae+bg, C2 = ce+dg.
    # These are 4 independent monomials, so rank 4.
    #
    # So: A_L @ B_L: 7 + 7 + 4 = 18 terms
    # A_R @ B_R: 7 + 7 + 4 = 18 terms
    # Total: 36 terms.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Indexing: U[i*m+j, c] for A element (i,j)
    # V[j*p+k, c] for B element (j,k)  
    # W[k*n+i, c] for C element (k,i) per the trace convention W[k*n+i, r]
    
    col = 0
    
    # Process A_L @ B_L: A_L is rows 0-1, cols 0-1 of A. B_L is rows 0-1 of B.
    # A_L elements: (0,0),(0,1),(1,0),(1,1) -> U indices: 0,1,4,5
    # B_L columns 0,1: B elements (0,0),(0,1),(1,0),(1,1) -> V indices: 0,1,5,6
    # B_L column 2: B elements (0,2),(1,2) -> V indices: 2,7
    # B_L columns 3,4: B elements (0,3),(0,4),(1,3),(1,4) -> V indices: 3,4,8,9
    
    # Strassen block 1: A_L @ B_L[:,0:2] (2x2 @ 2x2)
    # A_L = [[a00, a01], [a10, a11]], B_L1 = [[b00, b01], [b10, b11]]
    # U indices for A_L: a00->U[0], a01->U[1], a10->U[4], a11->U[5]
    # V indices for B_L1: b00->V[0], b01->V[1], b10->V[5], b11->V[6]
    # W indices for C[:,0:2]: C[0,0]->W[0], C[1,0]->W[2], C[0,1]->W[1], C[1,1]->W[3]
    # W[k*n+i]: C[k,i] -> W[k*2+i]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    
    # m1 = (a00+a11)(b00+b11)
    U[0][col] = 1; U[5][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
    # C11 += m1, C12 += m1, C21 += m1, C22 += m1
    col += 1
    
    # m2 = (a10+a11)*b00
    U[4][col] = 1; U[5][col] = 1
    V[0][col] = 1
    W[2][col] = 1; W[3][col] = 1
    # C21 += m2, C22 += m2
    col += 1
    
    # m3 = a00*(b01-b11)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[1][col] = 1; W[3][col] = 1
    # C12 += m3, C22 += m3
    col += 1
    
    # m4 = a11*(b10-b00)
    U[5][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[2][col] = 1
    # C11 += m4, C21 += m4
    col += 1
    
    # m5 = (a00+a01)*b11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[1][col] = 1
    # C11 -= m5, C12 += m5
    col += 1
    
    # m6 = (a10-a00)*(b00+b01)
    U[4][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    # C22 += m6
    col += 1
    
    # m7 = (a01-a11)*(b10+b11)
    U[1][col] = 1; U[5][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    # C11 += m7
    col += 1
    
    # Strassen block 2: A_L @ B_L[:,2:4] (2x2 @ 2x2)
    # B_L2 = [[b02, b03], [b12, b13]]
    # V indices: b02->V[2], b03->V[3], b12->V[7], b13->V[8]
    # W indices for C[:,2:4]: C[0,2]->W[4], C[1,2]->W[6], C[0,3]->W[5], C[1,3]->W[7]
    
    # m1 = (a00+a11)(b02+b12)
    U[0][col] = 1; U[5][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[5][col] = 1; W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # m2 = (a10+a11)*b02
    U[4][col] = 1; U[5][col] = 1
    V[2][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # m3 = a00*(b03-b13)
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[5][col] = 1; W[7][col] = 1
    col += 1
    
    # m4 = a11*(b12-b02)
    U[5][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # m5 = (a00+a01)*b13
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # m6 = (a10-a00)*(b02+b03)
    U[4][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # m7 = (a01-a11)*(b12+b13)
    U[1][col] = 1; U[5][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # Schoolbook for A_L @ B_L[:,4] (2x2 @ 2x1)
    # B_L3 = [[b04], [b14]]
    # V indices: b04->V[4], b14->V[9]
    # W indices for C[:,4]: C[0,4]->W[8], C[1,4]->W[9]
    
    # C[0,4] += a00*b04
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1
    
    # C[0,4] += a01*b14
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    
    # C[1,4] += a10*b04
    U[4][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    col += 1
    
    # C[1,4] += a11*b14
    U[5][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # Now A_R @ B_R: A_R is rows 0-1, cols 2-3 of A. B_R is rows 2-3 of B.
    # A_R elements: (0,2),(0,3),(1,2),(1,3) -> U indices: 2,3,6,7
    # B_R columns 0,1: B elements (2,0),(2,1),(3,0),(3,1) -> V indices: 10,11,15,16
    # B_R column 2: B elements (2,2),(3,2) -> V indices: 12,17
    # B_R columns 3,4: B elements (2,3),(2,4),(3,3),(3,4) -> V indices: 13,14,18,19
    
    # Strassen block 3: A_R @ B_R[:,0:2] (2x2 @ 2x2)
    # A_R = [[a02, a03], [a12, a13]], B_R1 = [[b20, b21], [b30, b31]]
    # U indices: a02->U[2], a03->U[3], a12->U[6], a13->U[7]
    # V indices: b20->V[10], b21->V[11], b30->V[15], b31->V[16]
    # W indices same as before: C[0,0]->W[0], etc.
    
    U[2][col] = 1; U[7][col] = 1
    V[10][col] = 1; V[15][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
    col += 1
    
    U[6][col] = 1; U[7][col] = 1
    V[10][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    U[2][col] = 1
    V[11][col] = 1; V[16][col] = -1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    U[7][col] = 1
    V[15][col] = 1; V[10][col] = -1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    U[2][col] = 1; U[3][col] = 1
    V[16][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    U[6][col] = 1; U[2][col] = -1
    V[10][col] = 1; V[11][col] = 1
    W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[7][col] = -1
    V[15][col] = 1; V[16][col] = 1
    W[0][col] = 1
    col += 1
    
    # Strassen block 4: A_R @ B_R[:,2:4] (2x2 @ 2x2)
    # B_R2 = [[b22, b23], [b32, b33]]
    # V indices: b22->V[12], b23->V[13], b32->V[17], b33->V[18]
    
    U[2][col] = 1; U[7][col] = 1
    V[12][col] = 1; V[17][col] = 1
    W[4][col] = 1; W[5][col] = 1; W[6][col] = 1; W[7][col] = 1
    col += 1
    
    U[6][col] = 1; U[7][col] = 1
    V[12][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    U[2][col] = 1
    V[13][col] = 1; V[18][col] = -1
    W[5][col] = 1; W[7][col] = 1
    col += 1
    
    U[7][col] = 1
    V[17][col] = 1; V[12][col] = -1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    U[2][col] = 1; U[3][col] = 1
    V[18][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    U[6][col] = 1; U[2][col] = -1
    V[12][col] = 1; V[13][col] = 1
    W[7][col] = 1
    col += 1
    
    U[3][col] = 1; U[7][col] = -1
    V[17][col] = 1; V[18][col] = 1
    W[4][col] = 1
    col += 1
    
    # Schoolbook for A_R @ B_R[:,4] (2x2 @ 2x1)
    # B_R3 = [[b24], [b34]]
    # V indices: b24->V[14], b34->V[19]
    
    U[2][col] = 1
    V[14][col] = 1
    W[8][col] = 1
    col += 1
    
    U[3][col] = 1
    V[19][col] = 1
    W[8][col] = 1
    col += 1
    
    U[6][col] = 1
    V[14][col] = 1
    W[9][col] = 1
    col += 1
    
    U[7][col] = 1
    V[19][col] = 1
    W[9][col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
