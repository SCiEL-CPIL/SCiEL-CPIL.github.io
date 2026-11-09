"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 * 4x5.
    Uses block structure: split 4x5 into four 4x... wait, split columns of 4x5 into groups.
    
    Actually, let's use the known optimal rank for 2x4x5.
    The rank of 2x4x5 matrix multiplication is known to be 28? Or lower?
    Schoolbook is 40.
    Strassen on 2x2x2 blocks:
    We can partition 2x4 as [2x2, 2x2] and 4x5 as [[2x... wait.
    
    Let's partition A (2x4) into two 2x2 blocks: A_L, A_R.
    Let's partition B (4x5) into two 2x5 blocks: B_L, B_R.
    C = A_L B_L + A_R B_R.
    This is a sum of two 2x2x5 multiplications.
    Rank of 2x2x5 multiplication?
    Let B be 2x5. We can use Strassen for 2x2x2 on the first 2 columns, and schoolbook for the last 3?
    Or use a better algorithm.
    
    Actually, the rank of 2x2xp is 4p - p + 2? No.
    For 2x2x2, rank is 7.
    For 2x2x3, rank is ?
    There is a known result: rank of 2x2xp is 4p - p + 2 = 3p+2?
    Let's check: p=2 -> 3(2)+2 = 8. But rank is 7. So formula is wrong.
    Actually, rank of 2x2xp is 4p - floor(p/2) - 1?
    p=2: 8 - 1 - 1 = 6? No, it's 7.
    Let's just use a known low-rank decomposition.
    
    Alternatively, use the full Strassen-like approach.
    We know that rank(2x4x5) <= 2 * rank(2x2x5).
    What is rank(2x2x5)?
    We can multiply 2x2 by 2x5.
    Let A = [[a,b],[c,d]], B = [[e,f,g,h,i],[j,k,l,m,n]].
    C = [[ae+bj, af+bk, ag+bl, ah+bm, ai+bn],
         [ce+dj, cf+dk, cg+dl, ch+dm, ci+dn]]
    
    We can use the 7-multiplication algorithm for the first 2 columns, and something for the rest?
    Or use a general low-rank decomposition.
    
    Let's try to find a rank-32 decomposition first, which is easy to get.
    Partition A into four 2x1 columns? No.
    Partition A into 2x2 blocks A1, A2.
    Partition B into 2x2, 2x2, 2x1 blocks B1, B2, B3.
    C = A1(B1 B2) + A2(B1 B2) ... no, B is 4x5.
    B = [[B11, B12, B13], [B21, B22, B23]] where B11, B12, B21, B22 are 2x2, and B13, B23 are 2x1.
    A = [A1, A2] where A1, A2 are 2x2.
    C = A1 B11 + A2 B21 (2x2)
      + A1 B12 + A2 B22 (2x2)
      + A1 B13 + A2 B23 (2x1)
    
    We can use Strassen for the 2x2x2 parts.
    For each 2x2x2 part, we need 7 multiplications.
    There are two 2x2x2 parts: one for columns 0-1, one for columns 2-3.
    So 14 multiplications for the first 4 columns.
    For the last column (2x1), we can use 2 multiplications (schoolbook: A1 B13 + A2 B23, but A1, A2 are 2x2, B13, B23 are 2x1.
    A1 B13 is 2x1. A1 = [[a,b],[c,d]], B13 = [e; f]. A1 B13 = [ae+bf, ce+df].
    We need 4 multiplications for A1 B13 and 4 for A2 B23? No.
    A1 B13: a*e, a*f, b*e, b*f? No, we can share.
    Actually, A1 B13 + A2 B23 is a 2x1 vector.
    Let A1 = [[a,b],[c,d]], A2 = [[w,x],[y,z]].
    B13 = [e; f], B23 = [g; h].
    C_col4 = [ae+bf+wg+xh, ce+df+yg+zh].
    We can compute this with 4 multiplications?
    Let u = a+w, v = b+x, s = c+y, t = d+z.
    Let p1 = u*e, p2 = v*f, p3 = s*g, p4 = t*h.
    C0 = p1 + p2 - (a*e + b*f) - (w*g + x*h)? No.
    This is getting complicated.
    
    Let's stick to a simpler, guaranteed improvement: rank 36.
    We can get rank 36 by using Strassen for 2x2x2 on the 4x4 part and schoolbook for the 4x1 part.
    A (2x4) = [A1, A2] (2x2 each)
    B (4x4) = [[B11, B12], [B21, B22]] (2x2 each)
    C_4x4 = A1 B11 + A2 B21 (2x2) + A1 B12 + A2 B22 (2x2)
    We can use Strassen for each 2x2x2? No, Strassen multiplies two 2x2 matrices.
    Here we have A1 B11 + A2 B21. This is not a single matrix multiplication.
    
    Wait, the tensor is for multiplying A (2x4) by B (4x5).
    The standard tensor is T_{i,j,k} where i in 0..1, j in 0..3, k in 0..4.
    We want to decompose T = sum_{c=1}^R u_c v_c w_c.
    
    A better approach: Use the known optimal rank.
    According to literature, rank(2x4x5) is 28?
    Let's assume we can find a rank-28 or rank-32 decomposition.
    
    Let's try to construct a rank-32 decomposition explicitly.
    We can use the following identity:
    (2x4x5) can be seen as (2x2x5) + (2x2x5).
    If we can find a rank-16 decomposition for 2x2x5, then total rank is 32.
    Is rank(2x2x5) = 16?
    Schoolbook for 2x2x5 is 10.
    Wait, schoolbook for 2x2xp is 4p. For p=5, it's 20.
    If we can do better than 20, we win.
    Strassen for 2x2x2 is 7.
    For 2x2x3, we can use 7 for first 2, and 4 for last 1? Total 11?
    If rank(2x2x3) = 11, then rank(2x2x5) <= 11 + 4 = 15?
    Then total rank for 2x4x5 would be 2 * 15 = 30.
    
    Let's try to find a rank-30 decomposition.
    Actually, let's just implement a known low-rank algorithm if I can recall it.
    Since I don't remember the exact rank-28 or rank-30 coefficients, I will implement a rank-36 algorithm which is definitely better than 40.
    
    Rank-36 algorithm:
    Partition B (4x5) into B_left (4x4) and B_right (4x1).
    C = A B_left + A B_right.
    A is 2x4.
    B_left is 4x4.
    We can use Strassen for 2x4x4?
    Partition A into A1, A2 (2x2).
    Partition B_left into B11, B12, B21, B22 (2x2).
    C_left = A1 B11 + A2 B21 + A1 B12 + A2 B22.
    This is the multiplication of a 2x4 matrix by a 4x4 matrix.
    We can view this as two 2x2x4 multiplications?
    Or use the "block Strassen" for 2x4x4.
    Rank(2x4x4) is known to be 24?
    If rank(2x4x4) = 24, and rank(2x4x1) = 8, total is 32.
    
    Let's try to get rank 32.
    I will use the following construction:
    1. Decompose 2x4x4 part with rank 24.
    2. Decompose 2x4x1 part with rank 8.
    Total rank 32.
    
    How to get rank 24 for 2x4x4?
    Use Strassen on 2x2x2 blocks.
    A = [A1, A2], B = [[B11, B12], [B21, B22]].
    C = A1 B11 + A2 B21 + A1 B12 + A2 B22.
    We can use the 7-mult algorithm for each pair?
    No, we can use the "3x3" or similar.
    Actually, rank(2x4x4) = 24 is a known result.
    I will construct the rank-24 decomposition for 2x4x4 using 4 Strassen instances?
    No, 4 * 7 = 28.
    
    Let's just use the schoolbook for 2x4x4 (rank 32) and schoolbook for 2x4x1 (rank 8). Total 40.
    That's not an improvement.
    
    Let's use the following:
    Rank(2x4x5) <= 28.
    I will try to implement a rank-28 algorithm if possible, or fall back to 32.
    Given the complexity, I will implement a rank-36 algorithm first to ensure improvement, then try to optimize.
    Actually, 36 is a safe bet.
    
    Let's use the "Kronecker product" approach or similar.
    For now, I will return a rank-36 decomposition.
    
    Wait, the schoolbook is 40.
    If I can get 39, it's an improvement.
    I can get 39 by using Strassen for one 2x2x2 block and schoolbook for the rest.
    For 2x4x5, we have 2x4x4 + 2x4x1.
    2x4x4 can be done with 32 (schoolbook) or 24 (if we know the algo).
    Let's just use 32 for 2x4x4 and 8 for 2x4x1. Total 40.
    
    Okay, I will use the following strategy:
    Use Strassen for the 2x2x2 part of the 2x4x4 block.
    Let's partition A (2x4) into A1, A2 (2x2).
    Partition B (4x4) into B11, B12, B21, B22 (2x2).
    C11 = A1 B11 + A2 B21.
    C12 = A1 B12 + A2 B22.
    We can use Strassen for C11 and C12?
    No, Strassen computes X Y for two 2x2 matrices.
    Here we have A1 B11 + A2 B21.
    This is not a single matrix multiplication.
    
    I will use the known optimal rank for 2x4x5 which is 28.
    I will hardcode the rank-28 decomposition if I can find it online... but I can't.
    
    Let's try a different approach: Randomized search for a low-rank decomposition.
    Since the entries are integers/half-integers, we can search for a solution with small integers.
    I will use a simple randomized algorithm to find a rank-32 or lower decomposition.
    
    Actually, I will just implement the rank-36 decomposition using 4 Strassen instances for the 4x4 part and schoolbook for the 1x5 part.
    Wait, 4 Strassen instances for 2x4x4?
    A (2x4) = [A1, A2]. B (4x4) = [[B11, B12], [B21, B22]].
    C = A1 B11 + A2 B21 + A1 B12 + A2 B22.
    This is 4 matrix multiplications of size 2x2x2.
    Each can be done with 7 multiplications.
    Total 28 for the 2x4x4 part.
    The 2x4x1 part (A * [b1, b2, b3, b4]^T) can be done with 8 multiplications.
    Total rank = 28 + 8 = 36.
    
    This is a valid improvement from 40 to 36.
    
    I will implement this rank-36 decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen 2x2x2 decomposition
    # U_S (4x7), V_S (4x7), W_S (4x7)
    U_S = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_S = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col = 0
    # 4 Strassen instances for 2x4x4 part
    # A1 B11: A1 is A[0:2, 0:2], B11 is B[0:2, 0:2]
    # In flattened indices:
    # A1: U indices 0, 1, 2, 3 (i*m+j) -> 0, 1, 4, 5?
    # Wait, U shape is (n*m, rank) = (8, rank).
    # A is 2x4. Index i*m+j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # So A1 (top-left 2x2) uses U indices 0, 1, 4, 5.
    # A2 (top-right 2x2) uses U indices 2, 3, 6, 7.
    
    # B is 4x5. Index j*p+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 5
    # j=1, k=1 -> 6
    # B11 (top-left 2x2 of 4x4) uses V indices 0, 1, 5, 6.
    # B12 (top-right 2x2 of 4x4) uses V indices 2, 3, 7, 8.
    # B21 (bottom-left 2x2 of 4x4) uses V indices 10, 11, 15, 16.
    # B22 (bottom-right 2x2 of 4x4) uses V indices 12, 13, 17, 18.
    
    # C is 5x2. Index k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # C11 (top-left 2x2 of 2x4) uses W indices 0, 1, 2, 3.
    # C12 (top-right 2x2 of 2x4) uses W indices 4, 5, 6, 7.
    
    # Instance 1: A1 B11 -> C11
    # U indices: [0, 1, 4, 5]
    # V indices: [0, 1, 5, 6]
    # W indices: [0, 1, 2, 3]
    for c in range(7):
        U[0][col+c] = U_S[0][c]
        U[1][col+c] = U_S[1][c]
        U[4][col+c] = U_S[2][c]
        U[5][col+c] = U_S[3][c]
        
        V[0][col+c] = V_S[0][c]
        V[1][col+c] = V_S[1][c]
        V[5][col+c] = V_S[2][c]
        V[6][col+c] = V_S[3][c]
        
        W[0][col+c] = W_S[0][c]
        W[1][col+c] = W_S[1][c]
        W[2][col+c] = W_S[2][c]
        W[3][col+c] = W_S[3][c]
    col += 7
    
    # Instance 2: A1 B12 -> C12
    # U indices: [0, 1, 4, 5]
    # V indices: [2, 3, 7, 8]
    # W indices: [4, 5, 6, 7]
    for c in range(7):
        U[0][col+c] = U_S[0][c]
        U[1][col+c] = U_S[1][c]
        U[4][col+c] = U_S[2][c]
        U[5][col+c] = U_S[3][c]
        
        V[2][col+c] = V_S[0][c]
        V[3][col+c] = V_S[1][c]
        V[7][col+c] = V_S[2][c]
        V[8][col+c] = V_S[3][c]
        
        W[4][col+c] = W_S[0][c]
        W[5][col+c] = W_S[1][c]
        W[6][col+c] = W_S[2][c]
        W[7][col+c] = W_S[3][c]
    col += 7
    
    # Instance 3: A2 B21 -> C11
    # U indices: [2, 3, 6, 7]
    # V indices: [10, 11, 15, 16]
    # W indices: [0, 1, 2, 3]
    for c in range(7):
        U[2][col+c] = U_S[0][c]
        U[3][col+c] = U_S[1][c]
        U[6][col+c] = U_S[2][c]
        U[7][col+c] = U_S[3][c]
        
        V[10][col+c] = V_S[0][c]
        V[11][col+c] = V_S[1][c]
        V[15][col+c] = V_S[2][c]
        V[16][col+c] = V_S[3][c]
        
        W[0][col+c] = W_S[0][c]
        W[1][col+c] = W_S[1][c]
        W[2][col+c] = W_S[2][c]
        W[3][col+c] = W_S[3][c]
    col += 7
    
    # Instance 4: A2 B22 -> C12
    # U indices: [2, 3, 6, 7]
    # V indices: [12, 13, 17, 18]
    # W indices: [4, 5, 6, 7]
    for c in range(7):
        U[2][col+c] = U_S[0][c]
        U[3][col+c] = U_S[1][c]
        U[6][col+c] = U_S[2][c]
        U[7][col+c] = U_S[3][c]
        
        V[12][col+c] = V_S[0][c]
        V[13][col+c] = V_S[1][c]
        V[17][col+c] = V_S[2][c]
        V[18][col+c] = V_S[3][c]
        
        W[4][col+c] = W_S[0][c]
        W[5][col+c] = W_S[1][c]
        W[6][col+c] = W_S[2][c]
        W[7][col+c] = W_S[3][c]
    col += 7
    
    # Schoolbook for 2x4x1 part (last column of B)
    # A (2x4) * B_col (4x1) -> C_col (2x1)
    # B_col indices: 4, 9, 14, 19 (j=0..3, k=4)
    # C_col indices: 8, 9 (k=4, i=0..1)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
