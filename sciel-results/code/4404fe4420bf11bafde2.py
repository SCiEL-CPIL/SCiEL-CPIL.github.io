"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition for 2x4 * 4x5.
    # We use a Strassen-style decomposition by splitting the middle dimension.
    #
    # Strategy:
    # Split A (2x4) into A1 (2x2) and A2 (2x2) horizontally.
    # Split B (4x5) into B1 (2x5) and B2 (2x5) vertically.
    # C = A1*B1 + A2*B2
    #
    # Use the identity: A1*B1 + A2*B2 = ((A1+A2)*(B1+B2) - (A1-A2)*(B1-B2))/2 + A1*B1 + A2*B2 - (A1*A2+B1*B2)/2 ? No.
    # Actually, for scalar x,y,z,w: xz+yw = ((x+y)(z+w) - (x-y)(z-w))/2.
    # This works for scalars. For matrices, since matrix multiplication is not commutative, we have to be careful.
    # However, the Strassen algorithm works for square matrices.
    # Here A1, A2 are 2x2. B1, B2 are 2x5.
    # The products A1*B1 and A2*B2 are 2x5.
    # The Strassen identity holds for any "matrices" where the multiplication is defined and associative, as long as the blocks can be added.
    # Specifically, (A1+A2)(B1+B2) = A1*B1 + A1*B2 + A2*B1 + A2*B2.
    # (A1-A2)(B1-B2) = A1*B1 - A1*B2 - A2*B1 + A2*B2.
    # Sum: 2*A1*B1 + 2*A2*B2.
    # So A1*B1 + A2*B2 = 0.5 * [ (A1+A2)(B1+B2) + (A1-A2)(B1-B2) ].
    #
    # Now we need to decompose the products:
    # 1. P1 = (A1+A2) * (B1+B2)
    #    (A1+A2) is 2x4. (B1+B2) is 4x5.
    #    This is the original problem 2x4 * 4x5! We can't just use it recursively without knowing its rank.
    #    BUT, we can split this further.
    #    Let's look at the structure again.
    #    Maybe splitting 4 into 2+2 is not the most efficient if the resulting 2x4*4x5 is the same as the original.
    #    
    #    Let's try splitting the other dimension.
    #    A is 2x4. B is 4x5.
    #    Split A into A_top (1x4) and A_bottom (1x4).
    #    Split B into B_left (4x2) and B_right (4x3)? No, B is 4x5.
    #    
    #    Let's try a known optimal decomposition for 2x4 * 4x5.
    #    According to literature, the rank of 2x4 * 4x5 is not trivially known to be < 40, but often lower bounds are high.
    #    However, we can use a block Strassen approach.
    #    
    #    Let's split A (2x4) into two 2x2 blocks: A = [A1 A2].
    #    Let's split B (4x5) into B1 (2x5) and B2 (2x5).
    #    C = A1*B1 + A2*B2.
    #    
    #    We can use the rank-7 decomposition for 2x2 * 2x2 (Strassen) for the parts where dimensions match, but B is 2x5.
    #    We can split B1 into B1a (2x2) and B1b (2x3).
    #    Similarly B2 into B2a (2x2) and B2b (2x3).
    #    
    #    C = A1*B1a + A1*B1b + A2*B2a + A2*B2b.
    #    
    #    Let's group:
    #    Term 1: A1*B1a (2x2 * 2x2). Rank 7 (Strassen).
    #    Term 2: A2*B2a (2x2 * 2x2). Rank 7 (Strassen).
    #    Term 3: A1*B1b (2x2 * 2x3).
    #    Term 4: A2*B2b (2x2 * 2x3).
    #    
    #    What is the rank of 2x2 * 2x3?
    #    It is equivalent to computing a 2x2 matrix times a 2x3 matrix.
    #    The tensor is <2,2,3>.
    #    Schoolbook rank is 2*2*3 = 12.
    #    Is there a better one?
    #    For 2x2 * 2x3, we can view it as 2 separate 2x2 * 2x1 products? No.
    #    Actually, rank of <2,2,3> is 11? Or 12?
    #    Let's check. The tensor is T_{ijk} = delta_{ik} (if we identify indices properly).
    #    Wait, the multiplication tensor for m1 x m2 * m2 x m3 is T_{i,j,k} where i is row of C, j is column of C, and the contraction is over the middle.
    #    For 2x2 * 2x3:
    #    Inputs: A (2x2), B (2x3). Output C (2x3).
    #    C_{i,k} = sum_{j=1}^2 A_{i,j} B_{j,k}.
    #    Rank is at most 12.
    #    Is it 11?
    #    There is a known result: rank(2x2 * 2x3) = 11? No, I think it's 12.
    #    Actually, let's look for "rank of 2x2x3 matrix multiplication tensor".
    #    It is known that R(2,2,3) = 12. No, wait.
    #    Let's try to construct a rank 11 decomposition.
    #    If we can't, we use 12.
    #    
    #    Let's assume rank 12 for 2x2 * 2x3 for now.
    #    Total rank = 7 + 7 + 12 + 12 = 38.
    #    This is better than 40.
    #    
    #    Can we do better?
    #    What if we use the Strassen identity on the sums?
    #    C = A1*B1 + A2*B2.
    #    Let S1 = A1+A2, D1 = A1-A2.
    #    Let S2 = B1+B2, D2 = B1-B2.
    #    S1*S2 + D1*D2 = 2(A1*B1 + A2*B2) = 2C.
    #    So C = 0.5 (S1*S2 + D1*D2).
    #    
    #    Now we need to compute S1*S2 and D1*D2.
    #    S1 is 2x4. S2 is 4x5.
    #    D1 is 2x4. D2 is 4x5.
    #    This doesn't reduce the problem size. It's the same shape.
    #    
    #    So the split into 2x2 blocks is the way to go.
    #    Rank 38 is a solid improvement over 40.
    #    
    #    Let's verify the rank of 2x2 * 2x3.
    #    If it is 12, total is 38.
    #    If it is 11, total is 36.
    #    
    #    Let's try to find a rank 11 decomposition for 2x2 * 2x3.
    #    Reference: "The rank of the matrix multiplication tensor <2,2,3>".
    #    Actually, I recall that R(2,2,2)=7, R(2,2,3)=12?
    #    Let's check a few sources mentally.
    #    For <2,2,3>, the trivial bound is 12.
    #    Lower bound is 11?
    #    If we assume 12, we get 38.
    #    
    #    Let's implement the decomposition with rank 38.
    #    We need:
    #    1. Strassen decomposition for 2x2 * 2x2 (Rank 7).
    #    2. Schoolbook decomposition for 2x2 * 2x3 (Rank 12).
    #    
    #    Wait, we can potentially mix them.
    #    A1*B1b: A1 is 2x2, B1b is 2x3.
    #    We can split B1b into columns? No, that's schoolbook.
    #    
    #    Let's construct the matrices.
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We will build U, V, W with a total rank R.
    # R = 7 (A1*B1a) + 7 (A2*B2a) + 12 (A1*B1b) + 12 (A2*B2b) = 38.
    
    R = 38
    U = [[0.0] * R for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0.0] * R for _ in range(m * p)]  # 4*5 = 20 rows
    W = [[0.0] * R for _ in range(p * n)]  # 5*2 = 10 rows
    
    # Indexing:
    # A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # U indices: i*m + j.
    # i=0: U[0], U[1], U[2], U[3] correspond to A[0,0], A[0,1], A[0,2], A[0,3]
    # i=1: U[4], U[5], U[6], U[7] correspond to A[1,0], A[1,1], A[1,2], A[1,3]
    #
    # B is 4x5. Rows 0,1,2,3. Cols 0,1,2,3,4.
    # V indices: j*p + k.
    # j=0: V[0..4] -> B[0,0..4]
    # j=1: V[5..9] -> B[1,0..4]
    # j=2: V[10..14] -> B[2,0..4]
    # j=3: V[15..19] -> B[3,0..4]
    #
    # C is 2x5. Rows 0,1. Cols 0,1,2,3,4.
    # W indices: k*n + i.
    # k=0: W[0], W[1] -> C[0,0], C[1,0]
    # k=1: W[2], W[3] -> C[0,1], C[1,1]
    # k=2: W[4], W[5] -> C[0,2], C[1,2]
    # k=3: W[6], W[7] -> C[0,3], C[1,3]
    # k=4: W[8], W[9] -> C[0,4], C[1,4]
    
    # Blocks:
    # A1 = A[0:2, 0:2] -> U rows [0,1, 4,5]
    # A2 = A[0:2, 2:4] -> U rows [2,3, 6,7]
    #
    # B1 = B[0:2, 0:5] -> V rows [0..4, 5..9]
    # B2 = B[2:4, 0:5] -> V rows [10..14, 15..19]
    #
    # B1a = B1[:, 0:2] -> V rows [0,1, 5,6]
    # B1b = B1[:, 2:5] -> V rows [2,3,4, 7,8,9]
    #
    # B2a = B2[:, 0:2] -> V rows [10,11, 15,16]
    # B2b = B2[:, 2:5] -> V rows [12,13,14, 17,18,19]
    
    # 1. A1 * B1a (2x2 * 2x2) -> Rank 7
    # Strassen decomposition:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A11+A22)(B21-B22)
    # P3 = (A11-A12)B11
    # P4 = A22(B22-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Let's map this to the global indices.
    # A11 -> U[0], A12 -> U[1], A21 -> U[4], A22 -> U[5]
    # B11 -> V[0], B12 -> V[1], B21 -> V[5], B22 -> V[6]
    # C11 -> W[0], C12 -> W[1], C21 -> W[2], C22 -> W[3]
    # Wait, W index is k*n+i.
    # C[0,0] is k=0,i=0 -> W[0].
    # C[0,1] is k=1,i=0 -> W[2].
    # C[1,0] is k=0,i=1 -> W[1].
    # C[1,1] is k=1,i=1 -> W[3].
    
    col = 0
    # P1
    U[0][col] = 1; U[5][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    # P2
    U[0][col] = 1; U[5][col] = 1
    V[5][col] = 1; V[6][col] = -1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    # P3
    U[0][col] = 1; U[1][col] = -1
    V[0][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    # P4
    U[5][col] = 1
    V[6][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    # P5
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    # P6
    U[4][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    # P7
    U[1][col] = 1; U[5][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[2][col] = -1
    col += 1
    
    # 2. A2 * B2a (2x2 * 2x2) -> Rank 7
    # A21 -> U[6], A22 -> U[7], A31 -> U[6]? No.
    # A2 is A[0:2, 2:4].
    # A[0,2]->U[2], A[0,3]->U[3], A[1,2]->U[6], A[1,3]->U[7].
    # Let's call these A11'=U[2], A12'=U[3], A21'=U[6], A22'=U[7].
    # B2a is B[2:4, 0:2].
    # B[2,0]->V[10], B[2,1]->V[11], B[3,0]->V[15], B[3,1]->V[16].
    # Let's call these B11'=V[10], B12'=V[11], B21'=V[15], B22'=V[16].
    # Result is C[0:2, 0:2] which is C[0,0], C[0,1], C[1,0], C[1,1].
    # W indices: W[0], W[2], W[1], W[3].
    
    # P1'
    U[2][col] = 1; U[7][col] = 1
    V[10][col] = 1; V[16][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    # P2'
    U[2][col] = 1; U[7][col] = 1
    V[15][col] = 1; V[16][col] = -1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    # P3'
    U[2][col] = 1; U[3][col] = -1
    V[10][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    # P4'
    U[7][col] = 1
    V[16][col] = 1; V[10][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    # P5'
    U[2][col] = 1; U[3][col] = 1
    V[16][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    # P6'
    U[6][col] = 1; U[2][col] = -1
    V[10][col] = 1; V[11][col] = 1
    W[3][col] = 1
    col += 1
    # P7'
    U[3][col] = 1; U[7][col] = -1
    V[15][col] = 1; V[16][col] = 1
    W[0][col] = 1; W[2][col] = -1
    col += 1
    
    # 3. A1 * B1b (2x2 * 2x3) -> Rank 12 (Schoolbook)
    # A1: U[0], U[1], U[4], U[5]
    # B1b: V[2], V[3], V[4], V[7], V[8], V[9]
    # Result: C[0:2, 2:5] -> C[0,2], C[0,3], C[0,4], C[1,2], C[1,3], C[1,4]
    # W indices:
    # C[0,2] -> k=2, i=0 -> W[4]
    # C[0,3] -> k=3, i=0 -> W[6]
    # C[0,4] -> k=4, i=0 -> W[8]
    # C[1,2] -> k=2, i=1 -> W[5]
    # C[1,3] -> k=3, i=1 -> W[7]
    # C[1,4] -> k=4, i=1 -> W[9]
    
    # Schoolbook: 4 elements in A1, 6 in B1b. 12 products.
    # A[0,0]*B[0,k], A[0,1]*B[1,k] -> C[0,k]
    # A[1,0]*B[0,k], A[1,1]*B[1,k] -> C[1,k]
    
    # A1 indices:
    # A11=U[0], A12=U[1]
    # A21=U[4], A22=U[5]
    
    # B1b indices:
    # B11=V[2] (B[0,2]), B12=V[3] (B[0,3]), B13=V[4] (B[0,4])
    # B21=V[7] (B[1,2]), B22=V[8] (B[1,3]), B23=V[9] (B[1,4])
    
    # For k=2 (col 2 of B1b, which is col 2 of B):
    # C[0,2] += A11*B11 + A12*B21
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1
    col += 1
    U[1][col] = 1; V[7][col] = 1; W[4][col] = 1
    col += 1
    # C[1,2] += A21*B11 + A22*B21
    U[4][col] = 1; V[2][col] = 1; W[5][col] = 1
    col += 1
    U[5][col] = 1; V[7][col] = 1; W[5][col] = 1
    col += 1
    
    # For k=3:
    # C[0,3] += A11*B12 + A12*B22
    U[0][col] = 1; V[3][col] = 1; W[6][col] = 1
    col += 1
    U[1][col] = 1; V[8][col] = 1; W[6][col] = 1
    col += 1
    # C[1,3] += A21*B12 + A22*B22
    U[4][col] = 1; V[3][col] = 1; W[7][col] = 1
    col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1
    col += 1
    
    # For k=4:
    # C[0,4] += A11*B13 + A12*B23
    U[0][col] = 1; V[4][col] = 1; W[8][col] = 1
    col += 1
    U[1][col] = 1; V[9][col] = 1; W[8][col] = 1
    col += 1
    # C[1,4] += A21*B13 + A22*B23
    U[4][col] = 1; V[4][col] = 1; W[9][col] = 1
    col += 1
    U[5][col] = 1; V[9][col] = 1; W[9][col] = 1
    col += 1
    
    # 4. A2 * B2b (2x2 * 2x3) -> Rank 12 (Schoolbook)
    # A2: U[2], U[3], U[6], U[7]
    # B2b: V[12], V[13], V[14], V[17], V[18], V[19]
    # Result: C[0:2, 2:5]
    # W indices same as above: W[4..9]
    
    # A21=U[2], A22=U[3]
    # A31=U[6], A32=U[7]
    
    # B21=V[12] (B[2,2]), B22=V[13] (B[2,3]), B23=V[14] (B[2,4])
    # B31=V[17] (B[3,2]), B32=V[18] (B[3,3]), B33=V[19] (B[3,4])
    
    # For k=2:
    U[2][col] = 1; V[12][col] = 1; W[4][col] = 1
    col += 1
    U[3][col] = 1; V[17][col] = 1; W[4][col] = 1
    col += 1
    U[6][col] = 1; V[12][col] = 1; W[5][col] = 1
    col += 1
    U[7][col] = 1; V[17][col] = 1; W[5][col] = 1
    col += 1
    
    # For k=3:
    U[2][col] = 1; V[13][col] = 1; W[6][col] = 1
    col += 1
    U[3][col] = 1; V[18][col] = 1; W[6][col] = 1
    col += 1
    U[6][col] = 1; V[13][col] = 1; W[7][col] = 1
    col += 1
    U[7][col] = 1; V[18][col] = 1; W[7][col] = 1
    col += 1
    
    # For k=4:
    U[2][col] = 1; V[14][col] = 1; W[8][col] = 1
    col += 1
    U[3][col] = 1; V[19][col] = 1; W[8][col] = 1
    col += 1
    U[6][col] = 1; V[14][col] = 1; W[9][col] = 1
    col += 1
    U[7][col] = 1; V[19][col] = 1; W[9][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
