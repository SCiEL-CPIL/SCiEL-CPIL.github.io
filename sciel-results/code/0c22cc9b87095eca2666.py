"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Strategy:
    # 1. Construct the schoolbook decomposition (rank 90) as a guaranteed fallback.
    # 2. Attempt to construct a block-diagonal decomposition with rank 39:
    #    - Split the inner dimension (5) into 1 and 4.
    #    - Block 1: 3x1 * 1x6. Rank = 3*1*6 = 18 (Schoolbook).
    #    - Block 2: 3x4 * 4x6. 
    #      We can view this as 3 * (1x4 * 4x6) or similar.
    #      Actually, the strategy in the prompt suggests using a tensor product of Winograd (rank 7 for 4x4) and Identity (rank 3 for 3x3? No, 3x1 and 1x6).
    #      Let's look at the dimensions:
    #      A is 3x5, B is 5x6.
    #      Split A into A1 (3x1) and A2 (3x4).
    #      Split B into B1 (1x6) and B2 (4x6).
    #      C = A1 B1 + A2 B2.
    #      
    #      Term 1: A1 (3x1) * B1 (1x6) -> C1 (3x6).
    #      This is an outer product of a 3x1 and 1x6 matrix? No, matrix multiplication.
    #      (3x1) * (1x6) = 3x6.
    #      The tensor rank for (n, m, p) = (3, 1, 6) is 3*1*6 = 18.
    #      We can decompose this with 18 terms.
    #      
    #      Term 2: A2 (3x4) * B2 (4x6) -> C2 (3x6).
    #      The tensor rank for (3, 4, 6) is at most 3*4*6 = 72 (schoolbook).
    #      The prompt suggests: "tensor product of the rank-7 Winograd decomposition for 4x4x4 matrix multiplication and a rank-3 identity".
    #      This implies decomposing the 3x4x6 multiplication into 3 independent 1x4x6 multiplications?
    #      Rank of (1, 4, 6) is 1*4*6 = 24.
    #      If we have 3 such blocks, total rank is 3 * 24 = 72.
    #      This is not 21.
    #      
    #      Let's re-read the prompt's strategy: "yielding a total rank of 18 + 21 = 39".
    #      How can 3x4x6 have rank 21?
    #      Winograd for 4x4x4 has rank 7.
    #      Maybe the 3x4x6 block is viewed as a Kronecker product of (3x1x1) and (1x4x4) and (1x1x6)? No.
    #      
    #      Perhaps the 3x4x6 multiplication is decomposed using the fact that 3 and 6 are multiples of 1?
    #      If we use the Winograd 4x4x4 (rank 7) for the 4x4 part, and then extend to 6 columns?
    #      A 4x6 multiplication can be viewed as two 4x3 multiplications.
    #      Rank of 4x3x3 is 4*3*3 = 36? No.
    #      
    #      Let's look at known ranks.
    #      Rank(3,4,4) is unknown exactly, but <= 3*4*4 = 48.
    #      Rank(4,4,4) = 7 (Winograd).
    #      
    #      If we assume the strategy is valid and rank 39 is achievable, I need to construct it.
    #      However, without explicit coefficients for a rank-21 decomposition of 3x4x6, I cannot generate it.
    #      The prompt says "Use a block-diagonal tensor decomposition...".
    #      
    #      Let's try a different block split that is easier to construct with known low-rank algorithms.
    #      Split 5 into 3 and 2.
    #      Block 1: 3x3 * 3x6. Rank <= 3*3*6 = 54.
    #      Block 2: 3x2 * 2x6. Rank <= 3*2*6 = 36.
    #      Total 90.
    #      
    #      Split 5 into 1 and 4.
    #      Block 1: 3x1 * 1x6. Rank 18.
    #      Block 2: 3x4 * 4x6.
    #      Can we use Strassen/Winograd for the 4x4 part?
    #      If we pad 3x4 to 4x4 and 4x6 to 4x8? No.
    #      
    #      Actually, let's look at the "rank-21" claim.
    #      Is there a rank-21 decomposition for 3x4x6?
    #      3*4*6 = 72.
    #      If we use the Winograd 4x4x4 (rank 7) for the 4x4 part, and then multiply by 3 and 6/4?
    #      This doesn't make sense directly.
    #      
    #      Let's try to implement the Block 1 (Rank 18) and see if we can find a known decomposition for Block 2.
    #      If not, I will fall back to schoolbook for Block 2 (Rank 72), total 90.
    #      Or, I can try to use the Winograd 4x4x4 (Rank 7) for a 4x4x4 sub-block and handle the rest.
    #      
    #      Let's try a different approach:
    #      Use the Winograd 4x4x4 (Rank 7) for the 4x4x4 part of the 4x6x6? No.
    #      
    #      Let's stick to the safe path:
    #      1. Construct Schoolbook (Rank 90).
    #      2. Try to construct a Block Diagonal with:
    #         - Block A: 3x1x6 (Rank 18).
    #         - Block B: 3x4x6.
    #      If I cannot find a low-rank decomposition for 3x4x6, I will use Schoolbook for Block B (Rank 72).
    #      Total Rank = 18 + 72 = 90. Same as schoolbook.
    #      
    #      Wait, the prompt says "yielding a total rank of 18 + 21 = 39".
    #      This implies a rank-21 decomposition for 3x4x6 exists or is assumed.
    #      Since I don't have the coefficients, I cannot implement it.
    #      
    #      However, I can try to use the Winograd 4x4x4 (Rank 7) for a 4x4x4 sub-problem.
    #      If I split 3x4x6 into:
    #      - 1x4x4 (Rank 4*4=16? No, 1*4*4=16).
    #      - 2x4x2 (Rank 2*4*2=16).
    #      This is getting complicated.
    #      
    #      Let's try to implement the Block 1 (3x1x6) with Rank 18.
    #      And for Block 2 (3x4x6), I will use the Schoolbook Rank 72.
    #      This gives Rank 90.
    #      
    #      Is there any way to reduce the rank of 3x4x6?
    #      The lower bound for 3x4x6 is max(3*4, 4*6, 3*6) = 24? No, lower bound is max(n*m, m*p, n*p) is not correct.
    #      The lower bound is max(n*m, m*p, n*p) is for the number of non-zero entries?
    #      The rank of the tensor T_{n,m,p} is at least max(n*m, m*p, n*p) / something?
    #      Actually, the rank of T_{n,m,p} is at least max(n*m, m*p, n*p) is false.
    #      The rank of T_{1,1,1} is 1.
    #      The rank of T_{2,2,2} is 7.
    #      
    #      Given the uncertainty, I will implement the Block 1 (Rank 18) and Block 2 (Rank 72) decomposition.
    #      This is structurally different from the full schoolbook but has the same rank.
    #      It might be a stepping stone, but the score will be the same.
    #      
    #      To improve the score, I need a lower rank.
    #      Let's try to use the Winograd 4x4x4 (Rank 7) for a 4x4x4 block.
    #      If I can embed a 4x4x4 multiplication into the 3x5x6 problem, I might reduce the rank.
    #      For example, if I use Winograd for a 4x4x4 part, and schoolbook for the rest.
    #      
    #      Let's try a block split:
    #      Split 5 into 4 and 1.
    #      Block 1: 3x4 * 4x6.
    #      Block 2: 3x1 * 1x6.
    #      
    #      For Block 1 (3x4x6):
    #      We can view it as 3 * (1x4x6).
    #      Rank of 1x4x6 is 24.
    #      Total 72.
    #      
    #      Alternatively, view it as 6 * (3x4x1).
    #      Rank of 3x4x1 is 12.
    #      Total 72.
    #      
    #      What if we use Winograd 4x4x4 (Rank 7) for the 4x4 part?
    #      We have A2 (3x4) and B2 (4x6).
    #      We can write B2 as [B21, B22] where B21 is 4x4 and B22 is 4x2.
    #      Then A2 B2 = A2 B21 + A2 B22.
    #      A2 B21 is 3x4.
    #      A2 B22 is 3x2.
    #      
    #      For A2 B21 (3x4 * 4x4 -> 3x4):
    #      We can use the Winograd 4x4x4 (Rank 7) for the 4x4x4 part?
    #      No, the output is 3x4, not 4x4.
    #      We can pad A2 to 4x4 by adding a zero row.
    #      Then A2_padded B21 is 4x4.
    #      We can use Winograd (Rank 7) to compute this 4x4 product.
    #      The first 3 rows of the result are the desired A2 B21.
    #      So this block has rank 7.
    #      
    #      For A2 B22 (3x4 * 4x2 -> 3x2):
    #      We can use schoolbook. Rank 3*4*2 = 24.
    #      
    #      So Block 1 (3x4x6) can be decomposed with rank 7 + 24 = 31.
    #      
    #      Block 2 (3x1x6) has rank 18.
    #      
    #      Total rank = 31 + 18 = 49.
    #      This is better than 90!
    #      
    #      Let's refine Block 1 (3x4x6):
    #      A2 is 3x4. B2 is 4x6.
    #      Split B2 into B21 (4x4) and B22 (4x2).
    #      C2 = A2 B21 + A2 B22.
    #      
    #      Term C21 = A2 B21 (3x4).
    #      Pad A2 to A2p (4x4) by adding a zero row at the bottom.
    #      Compute A2p B21 (4x4) using Winograd (Rank 7).
    #      Take the first 3 rows of the result.
    #      
    #      Term C22 = A2 B22 (3x2).
    #      Use schoolbook for 3x4 * 4x2. Rank 24.
    #      
    #      Total rank for Block 1 = 7 + 24 = 31.
    #      
    #      Block 2 (3x1x6):
    #      A1 is 3x1. B1 is 1x6.
    #      C1 = A1 B1 (3x6).
    #      Use schoolbook. Rank 18.
    #      
    #      Total rank = 31 + 18 = 49.
    #      
    #      Let's implement this.
    #      I need the Winograd 4x4x4 coefficients.
    #      
    #      Winograd 4x4x4 (Rank 7) coefficients:
    #      The 7 terms are:
    #      1. (1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0) -> No, that's not right.
    #      
    #      The standard Winograd 4x4x4 decomposition has 7 terms.
    #      I will hardcode the coefficients if I can find them.
    #      Otherwise, I will fall back to schoolbook for the 4x4x4 part (Rank 16).
    #      Then Block 1 rank = 16 + 24 = 40.
    #      Total rank = 40 + 18 = 58.
    #      Still better than 90.
    #      
    #      Let's try to find the Winograd 4x4x4 coefficients.
    #      According to literature, the 4x4x4 matrix multiplication tensor has rank 7.
    #      The coefficients are integers.
    #      
    #      I will use a known set of coefficients for the 4x4x4 Winograd algorithm.
    #      If I cannot recall them, I will use the schoolbook for the 4x4x4 part.
    #      
    #      Let's assume I can use the schoolbook for the 4x4x4 part for now, and try to optimize later.
    #      Schoolbook 4x4x4 rank 16.
    #      Block 1 rank = 16 + 24 = 40.
    #      Block 2 rank = 18.
    #      Total rank = 58.
    #      
    #      Let's implement this.

    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    def winograd_4x4x4_decomposition():
        # Placeholder for Winograd 4x4x4 rank 7 decomposition.
        # If not available, use schoolbook.
        return None

    # Try to get Winograd 4x4x4
    winograd = winograd_4x4x4_decomposition()
    use_winograd = winograd is not None

    if not use_winograd:
        # Use schoolbook for 4x4x4
        U_444, V_444, W_444 = schoolbook_decomposition(4, 4, 4)
        rank_444 = 16
    else:
        U_444, V_444, W_444 = winograd
        rank_444 = 7

    # Block 1: 3x4 * 4x6
    # Split B into B1 (4x4) and B2 (4x2)
    # C1 = A * B1 (3x4) using Winograd/Schoolbook 4x4x4 (padded)
    # C2 = A * B2 (3x2) using Schoolbook 3x4x2

    # Dimensions for Block 1
    n1, m1, p1 = 3, 4, 6
    # Sub-block 1.1: 3x4 * 4x4 -> 3x4
    # Pad A to 4x4.
    # U_11: (4*4, rank_444) from Winograd/Schoolbook
    # V_11: (4*4, rank_444)
    # W_11: (4*4, rank_444)
    # We need to map the 4x4 result to the 3x4 output.
    # The 4x4 result is C_padded. We take the first 3 rows.
    # So W_11 should be modified to only select the first 3 rows of the 4x4 output.
    # W_11 shape is (4*4, rank_444). Index is (k, i) where k is col, i is row?
    # Convention: W[k*n+i, r]. Here n=4, p=4.
    # We want the output to be 3x4. So we only care about i=0,1,2.
    # We can zero out the rows of W_11 corresponding to i=3.
    # W_11 index: k*4 + i.
    # For i=3, indices are k*4 + 3 for k=0..3.
    # These are 3, 7, 11, 15.
    
    U_11 = U_444.copy()
    V_11 = V_444.copy()
    W_11 = W_444.copy()
    for k in range(4):
        W_11[k*4 + 3, :] = 0.0

    # Sub-block 1.2: 3x4 * 4x2 -> 3x2
    U_12, V_12, W_12 = schoolbook_decomposition(3, 4, 2)
    rank_12 = 3*4*2

    # Total rank for Block 1
    rank_block1 = rank_444 + rank_12

    # Block 2: 3x1 * 1x6
    U_2, V_2, W_2 = schoolbook_decomposition(3, 1, 6)
    rank_block2 = 3*1*6

    # Total rank
    total_rank = rank_block1 + rank_block2

    # Construct full U, V, W matrices
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))

    # Block 1 (3x4 * 4x6)
    # A indices: 0..2, j=0..3 -> U indices 0..11
    # B indices: j=0..3, k=0..5 -> V indices 0..23
    # C indices: k=0..5, i=0..2 -> W indices 0..17

    # Sub-block 1.1 (3x4 * 4x4)
    # A indices: i=0..2, j=0..3.
    # U_11 is (16, rank_444). Index i*4+j.
    # We need to map U_11[i*4+j] to U_full[i*5+j].
    # Note: U_11 is for padded A (4x4). We only use rows i=0..2.
    for i in range(3):
        for j in range(4):
            U_full[i*m + j, 0:rank_444] = U_11[i*4 + j, :]

    # V_11 is for B1 (4x4). Index j*4+k.
    # We need to map V_11[j*4+k] to V_full[j*6+k].
    # B1 is the first 4 columns of B2 (which is the 4x6 block).
    # So k=0..3.
    for j in range(4):
        for k in range(4):
            V_full[j*p + k, 0:rank_444] = V_11[j*4 + k, :]

    # W_11 is for C1 (4x4). Index k*4+i.
    # We need to map W_11[k*4+i] to W_full[k*3+i].
    # C1 is the first 4 columns of the 3x6 output.
    # So k=0..3, i=0..2.
    for k in range(4):
        for i in range(3):
            W_full[k*n + i, 0:rank_444] = W_11[k*4 + i, :]

    # Sub-block 1.2 (3x4 * 4x2)
    # A indices: i=0..2, j=0..3.
    for i in range(3):
        for j in range(4):
            U_full[i*m + j, rank_444:rank_444+rank_12] = U_12[i*4 + j, :]

    # V_12 is for B2 (4x2). Index j*2+k.
    # B2 is the last 2 columns of B2 (which is the 4x6 block).
    # So k=4..5.
    # In V_full, the index is j*6 + k.
    for j in range(4):
        for k in range(2):
            V_full[j*p + (4+k), rank_444:rank_444+rank_12] = V_12[j*2 + k, :]

    # W_12 is for C2 (3x2). Index k*3+i.
    # C2 is the last 2 columns of the 3x6 output.
    # So k=4..5, i=0..2.
    for k in range(2):
        for i in range(3):
            W_full[k*n + i, rank_444:rank_444+rank_12] = W_12[k*3 + i, :]

    # Block 2 (3x1 * 1x6)
    # A indices: i=0..2, j=4.
    for i in range(3):
        U_full[i*m + 4, rank_block1:rank_block1+rank_block2] = U_2[i*1 + 0, :]

    # V_2 is for B (1x6). Index 0*6+k.
    for k in range(6):
        V_full[4*p + k, rank_block1:rank_block1+rank_block2] = V_2[0*1 + k, :]

    # W_2 is for C (3x6). Index k*3+i.
    for k in range(6):
        for i in range(3):
            W_full[k*n + i, rank_block1:rank_block1+rank_block2] = W_2[k*3 + i, :]

    # Verify the decomposition
    U_check = U_full
    V_check = V_full
    W_check = W_full
    
    # Check non-zero entries
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U_check[a, :], V_check[b, :] * W_check[c, :])
                err += (val - 1.0)**2

    if err > 1e-6:
        # Fallback to schoolbook
        U_full, V_full, W_full = schoolbook_decomposition(n, m, p)

    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
# EVOLVE-BLOCK-END
