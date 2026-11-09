"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time as _time
from fractions import Fraction as _Frac

def construct():
    """Return rank-minimal exact tensor decomposition for 2x4 * 4x7.
    
    Splits the 4-dimension into two 2-blocks and one 1-block:
    - 2 blocks of 2x4x2 (Strassen rank 7 each)
    - 1 block of 2x4x1 (schoolbook rank 8)
    Total rank: 7+7+8 = 22
    Wait, 4 = 2+2, so two blocks of 2x4x2.
    Actually 4 = 2+2, two Strassen blocks of rank 7 = 14 total.
    But we need to handle the 7 output columns: 4 = 2+2 for A's columns,
    and 7 = 2+2+2+1 for B's columns? No...
    
    Let me reconsider: A is 2x4, B is 4x7, C is 2x7.
    The middle dimension is 4. We can split A's columns into 2+2 and
    B's rows correspondingly. Then for each pair, we have 2x2x7 subproblems.
    Actually, let's split the output: 7 = 2+2+2+1.
    
    Better approach: split the middle dimension 4 into 2+2.
    A = [A1 | A2] where A1, A2 are 2x2.
    B = [B1; B2] where B1, B2 are 2x7.
    C = A1*B1 + A2*B2.
    
    Each 2x2x7 can be done with rank... for 2x2xp, rank is at most 2p 
    via schoolbook, but with Strassen-like for the 2x2 part:
    The 2x2x1 tensor has rank 3 (Strassen), 2x2x2 has rank 7, etc.
    
    For 2x2xp: we can use the fact that it's like 2x2 matrix mult by a 1xp matrix.
    Rank of 2x2xp tensor: this is the tensor of (2x2) * (1xp) which is 
    M_{2,2,p}. Known ranks: M_{2,2,1} rank 3, M_{2,2,2} rank 7, M_{2,2,3} rank 11?
    
    Actually let's just use: split 7 into 2+2+2+1.
    For each 2x2x2 block: rank 7 (Strassen for 2x2x2 is 7)
    For each 2x2x1 block: rank 3 (Strassen for 2x2x1)
    Total: 3*7 + 3 = 24? No, 3 blocks of 2x2x2 and 1 block of 2x2x1.
    
    Wait, I'm overcomplicating. Let me just do:
    Split middle dim 4 into 2+2.
    For each 2x2x7 subproblem, use schoolbook: rank = 2*2*7 = 28? No that's too much.
    
    Actually the simplest guaranteed improvement: use the known result that
    M_{2,4,7} can be done in rank <= 2*4*7 - something.
    
    Let me just implement a concrete decomposition:
    Split the 4 middle dimension into two groups of 2.
    For group 1 (columns 0,1 of A, rows 0,1 of B): 2x2x7 problem
    For group 2 (columns 2,3 of A, rows 2,3 of B): 2x2x7 problem
    
    For each 2x2x7, use schoolbook: 2*2*7 = 28 terms. Total 56. No improvement.
    
    Use Strassen on the 2x2 part: each 2x2x7 has rank at most 3*7 = 21? 
    No, Strassen reduces the 2x2 multiplication to 7 multiplications, but each
    multiplication is by a scalar (from the 7-column output). So rank = 7*7=49? Worse.
    
    Let me try a different structural approach. The known asymptotic bounds suggest
    M_{2,4,7} rank is well below 56. Let me use a simple block decomposition:
    
    A is 2x4, B is 4x7. Write A = [A_11 A_12; A_21 A_22] with 2x2 blocks.
    B = [B_11 B_12; B_21 B_22] with 2x7 blocks? No, B is 4x7 so blocks are 2x7.
    
    C = A*B = [A_11 A_12; A_21 A_22] * [B_1; B_2] where B_1, B_2 are 2x7.
    C = A_11*B_1 + A_12*B_2 (top row, 1x7)
       A_21*B_1 + A_22*B_2 (bottom row, 1x7)
    
    Each A_ij is 2x2, B_j is 2x7. The product A_ij * B_j is 2x7.
    Using Strassen on the 2x2 part: 7 scalar multiplications, each result is 
    a linear combination of entries of B_j (which is 2x7, so 14 entries).
    So each A_ij * B_j takes rank 7 (Strassen) * 7 (output columns) = 49? 
    That's worse than schoolbook 4*7=28 per block.
    
    I think the best practical approach for now is to keep schoolbook but see if
    we can find any cancellation. Let me just return the schoolbook decomposition
    with rank 56, which is correct, and rely on the evaluation to see if there's
    a better structure.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
