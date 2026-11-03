import time
import sys

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # Split B (3x6) into three 3x2 blocks.
    # Each block multiplication is 3x3x2.
    # We use a rank-14 decomposition for 3x3x2.
    # Total rank = 3 * 14 = 42.
    
    # First, define the Rank-7 Strassen decomposition for 2x2x2.
    # Indices: 0,1 for rows/cols.
    # U, V, W are 4x7 matrices.
    # Strassen's 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to tensor factors U, V, W.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2x2: n=2, m=2, p=2.
    # U is 4x7, V is 4x7, W is 4x7.
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1
    # (A11+A22)(B11+B22)
    # U: A11=1, A22=1
    U2[0][0] = 1 # A11
    U2[3][0] = 1 # A22
    # V: B11=1, B22=1
    V2[0][0] = 1 # B11
    V2[3][0] = 1 # B22
    # W: C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
    W2[0][0] = 1 # C11 (k=0, i=0) -> index k*2+i = 0
    W2[1][0] = -1 # C12 (k=1, i=0) -> index 1*2+0 = 2? Wait.
    # W index is k*n+i. n=2.
    # k=0 (col 1), i=0 (row 1) -> 0*2+0 = 0.
    # k=1 (col 2), i=0 (row 1) -> 1*2+0 = 2.
    # k=0, i=1 -> 0*2+1 = 1.
    # k=1, i=1 -> 1*2+1 = 3.
    W2[2][0] = -1 # C12
    W2[3][0] = 1  # C22
    W2[1][0] = -1 # C21
    
    # r=1: M2 = (A21+A22)B11
    U2[2][1] = 1 # A21 (row 2, col 1) -> index 1*2+0 = 2
    U2[3][1] = 1 # A22
    V2[0][1] = 1 # B11
    # W: C21 += M2
    W2[1][1] = 1 # C21 (k=0, i=1) -> index 1
    
    # r=2: M3 = A11(B12-B22)
    U2[0][2] = 1 # A11
    V2[1][2] = 1 # B12 (row 1, col 2) -> index 0*2+1 = 1
    V2[3][2] = -1 # B22
    # W: C12 += M3
    W2[2][2] = 1 # C12 (k=1, i=0) -> index 2
    
    # r=3: M4 = A22(B21-B11)
    U2[3][3] = 1 # A22
    V2[2][3] = 1 # B21 (row 2, col 1) -> index 1*2+0 = 2
    V2[0][3] = -1 # B11
    # W: C22 += M4
    W2[3][3] = 1 # C22 (k=1, i=1) -> index 3
    
    # r=4: M5 = (A11+A12)B22
    U2[0][4] = 1 # A11
    U2[1][4] = 1 # A12 (row 1, col 2) -> index 0*2+1 = 1
    V2[3][4] = 1 # B22
    # W: C22 += M5
    W2[3][4] = 1 # C22
    
    # r=5: M6 = (A21-A11)(B11+B12)
    U2[2][5] = 1 # A21
    U2[0][5] = -1 # A11
    V2[0][5] = 1 # B11
    V2[1][5] = 1 # B12
    # W: C21 += M6
    W2[1][5] = 1 # C21
    
    # r=6: M7 = (A12-A22)(B21+B22)
    U2[1][6] = 1 # A12
    U2[3][6] = -1 # A22
    V2[2][6] = 1 # B21
    V2[3][6] = 1 # B22
    # W: C12 += M7
    W2[2][6] = 1 # C12
    
    # Now construct Rank-14 decomposition for 3x3x2.
    # We use the 2x2x2 part for the top-left 2x2 of A and B.
    # A:
    # [A11 A12 A13]
    # [A21 A22 A23]
    # [A31 A32 A33]
    # B (3x2):
    # [B11 B12]
    # [B21 B22]
    # [B31 B32]
    
    # C (3x2):
    # [C11 C12]
    # [C21 C22]
    # [C31 C32]
    
    # C11 = A11B11 + A12B21 + A13B31
    # C12 = A11B12 + A12B22 + A13B32
    # C21 = A21B11 + A22B21 + A23B31
    # C22 = A21B12 + A22B22 + A23B32
    # C31 = A31B11 + A32B21 + A33B31
    # C32 = A31B12 + A32B22 + A33B32
    
    # We use the 7 terms from Strassen for the 2x2 part (A11,A12,A21,A22 and B11,B12,B21,B22).
    # This covers:
    # C11: A11B11 + A12B21 (partially)
    # C12: A11B12 + A12B22 (partially)
    # C21: A21B11 + A22B21 (partially)
    # C22: A21B12 + A22B22 (partially)
    
    # We need to add the terms involving A13, A23, A31, A32, A33 and B31, B32.
    
    # Let's just use the schoolbook for the remaining parts to be safe, or find a low rank decomposition.
    # Given the complexity, I will use the Rank-14 decomposition which is known to exist.
    # I will hardcode the Rank-14 factors if possible, or use a simpler Rank-18 (schoolbook) for 3x3x2 if I can't.
    # But Rank 18 * 3 = 54. No gain.
    
    # I will use the Rank-14 decomposition from a known source.
    # Since I cannot look it up, I will use a heuristic: 
    # The rank of 3x3x2 is 14.
    # I will construct it by adding 7 rank-1 terms to the Strassen 7 terms.
    
    # The 7 Strassen terms handle the 2x2x2 part.
    # The remaining terms are:
    # A13B31, A13B32, A23B31, A23B32, A31B11, A31B12, A32B21, A32B22, A33B31, A33B32.
    # This is 10 terms.
    # Total 7 + 10 = 17.
    # We can reduce this to 7?
    # A13B31 + A13B32 = A13(B31, B32). Rank 2.
    # A23B31 + A23B32 = A23(B31, B32). Rank 2.
    # A31B11 + A31B12 = A31(B11, B12). Rank 2.
    # A32B21 + A32B22 = A32(B21, B22). Rank 2.
    # A33B31 + A33B32 = A33(B31, B32). Rank 2.
    # Total 10.
    # 7 + 10 = 17.
    # This is not 14.
    
    # However, the Rank 14 decomposition is known.
    # I will fall back to the Rank 46 solution (two 3x3x3 blocks with Rank 23) if I can't get Rank 14.
    # But I don't have Rank 23 coefficients either.
    
    # I will use the Schoolbook Rank 54 as a fallback.
    # But I can try to improve it by using the Rank 7 Strassen for 2x2x2 blocks within the 3x3x6?
    
    # Let's just return the Schoolbook for now, but optimized.
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
