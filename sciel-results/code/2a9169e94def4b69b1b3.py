"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    We use a block decomposition approach:
    - Partition A (3x5) into A1 (3x3) and A2 (3x2)
    - Partition B (5x7) into B1 (3x7) and B2 (2x7)
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (3x3 * 3x7):
      Partition B1 into B1a (3x3) and B1b (3x4)
      A1*B1a: 3x3x3 -> use Strassen rank-23
      A1*B1b: 3x3x4 -> use schoolbook rank 36
      
    For A2*B2 (3x2 * 2x7):
      Partition B2 into B2a (2x3) and B2b (2x4)
      A2*B2a: 3x2x3 -> use Laderman rank-7
      A2*B2b: 3x2x4 -> use schoolbook rank 24
      
    Total: 23 + 36 + 7 + 24 = 90
    
    Actually, let me reconsider. Let me try to get better decompositions.
    
    For 3x3x3: The known minimum rank is 23 (Strassen-like).
    For 3x2x3: The known minimum rank is 7 (Laderman-like for rectangular).
    For 3x3x4: We can partition into 3x3x3 + 3x3x1, giving 23 + 9 = 32.
    For 3x2x4: We can partition into 3x2x3 + 3x2x1, giving 7 + 6 = 13.
    
    Total: 23 + 32 + 7 + 13 = 75
    
    Even better:
    For 3x3x4: partition into 3x3x2 + 3x3x2, each 3x3x2 has rank... 
    Actually 3x3x2 can be seen as two 3x3x1 columns, each rank 9, total 18.
    Or use a different decomposition.
    
    Let me think about this differently. Let me use the following approach:
    
    A (3x5) * B (5x7) = C (3x7)
    
    I'll partition the middle dimension (5) into 3+2:
    A = [A1 (3x3) | A2 (3x2)]
    B = [B1 (3x7); B2 (2x7)]
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (3x3 * 3x7):
      I'll partition the output columns into 3+4:
      B1 = [B1a (3x3) | B1b (3x4)]
      A1*B1a: 3x3x3 -> rank 23
      A1*B1b: 3x3x4 -> I'll partition B1b into 2+2:
        B1b = [B1b1 (3x2) | B1b2 (3x2)]
        A1*B1b1: 3x3x2 -> rank ? 
        A1*B1b2: 3x3x2 -> rank ?
      
      For 3x3x2: This is multiplying a 3x3 by a 3x2 to get 3x2.
      Schoolbook: 3*3*2 = 18.
      Can we do better? We can think of it as 3 separate 1x3x2 multiplications... no.
      Actually, 3x3x2 tensor: we can partition rows of output into 1+1+1, each 1x3x2 has rank 6 (schoolbook). Total 18.
      Or partition columns of A: 3x3x2 = (1x3x2) + (1x3x2) + (1x3x2) = 6+6+6 = 18.
      Or use Strassen-like: pad to 4x4x2? That might not help.
      
      Let me just use schoolbook for 3x3x2: rank 18.
      So A1*B1b: 18 + 18 = 36.
      
    For A2*B2 (3x2 * 2x7):
      Partition output columns into 3+4:
      B2 = [B2a (2x3) | B2b (2x4)]
      A2*B2a: 3x2x3 -> rank 7 (Laderman)
      A2*B2b: 3x2x4 -> partition into 2+2:
        B2b = [B2b1 (2x2) | B2b2 (2x2)]
        A2*B2b1: 3x2x2 -> rank ?
        A2*B2b2: 3x2x2 -> rank ?
      
      For 3x2x2: multiplying 3x2 by 2x2 to get 3x2.
      Schoolbook: 3*2*2 = 12.
      Can we do better? Partition rows: 1x2x2 + 1x2x2 + 1x2x2 = 4+4+4 = 12.
      Or use Strassen for 2x2x2 (rank 7) for each row? No, each row is 1x2x2 which is just 4 terms.
      Actually 1x2x2: C[i,k] = sum_j A[i,j]*B[j,k]. For fixed i, this is a 2x2 matrix product (1x2 * 2x2 = 1x2... wait).
      Actually A is 3x2, B is 2x2, C is 3x2.
      C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
      For each i (3 values) and k (2 values), that's 6 output entries, each requiring 2 multiplications = 12 total.
      Schoolbook rank = 12.
      
      Hmm, can we use Strassen here? Strassen is for 2x2x2. 
      For 3x2x2, we have 3 rows of A, each doing a 1x2 * 2x2 multiplication.
      Each 1x2 * 2x2 = 1x2 has rank 4 (schoolbook). 3 rows = 12.
      
      Alternatively, think of it as 2x2x2 applied 3 times with different row selections... 
      Actually, let me just use schoolbook: rank 12 each, total 24.
      
    So total: 23 + 36 + 7 + 24 = 90.
    
    Wait, I made an error. Let me recompute:
    A1*B1a: 3x3x3 -> 23
    A1*B1b1: 3x3x2 -> 18
    A1*B1b2: 3x3x2 -> 18
    A2*B2a: 3x2x3 -> 7
    A2*B2b1: 3x2x2 -> 12
    A2*B2b2: 3x2x2 -> 12
    Total: 23 + 18 + 18 + 7 + 12 + 12 = 90
    
    This is better than 105 (schoolbook). Let me implement this.
    
    Actually, let me try an even better partition. What if I partition the middle dimension differently?
    
    A (3x5) * B (5x7)
    Partition middle into 2+3:
    A = [A1 (3x2) | A2 (3x3)]
    B = [B1 (2x7); B2 (3x7)]
    
    C = A1*B1 + A2*B2
    
    A1*B1: 3x2 * 2x7 = 3x7
      Partition B1 columns into 3+4:
      B1 = [B1a (2x3) | B1b (2x4)]
      A1*B1a: 3x2x3 -> 7
      A1*B1b: 3x2x4 -> partition into 2+2:
        A1*B1b1: 3x2x2 -> 12
        A1*B1b2: 3x2x2 -> 12
      Subtotal: 7 + 12 + 12 = 31
    
    A2*B2: 3x3 * 3x7 = 3x7
      Partition B2 columns into 3+4:
      B2 = [B2a (3x3) | B2b (3x4)]
      A2*B2a: 3x3x3 -> 23
      A2*B2b: 3x3x4 -> partition into 2+2:
        A2*B2b1: 3x3x2 -> 18
        A2*B2b2: 3x3x2 -> 18
      Subtotal: 23 + 18 + 18 = 59
    
    Total: 31 + 59 = 90. Same.
    
    Let me try yet another approach. What about using Strassen for 4x4x4 and extracting?
    Pad 3x5 to 4x6, pad 5x7 to 6x8? That gets complicated.
    
    Let me stick with the 90-rank approach and implement it carefully.
    
    Actually, I realize I should double-check the rank-7 decomposition for 3x2x3.
    
    For 3x2x3: A is 3x2, B is 2x3, C is 3x3.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Schoolbook: 3*2*3 = 18 terms.
    
    The Laderman algorithm gives rank 7 for 3x3x3... wait, no. Laderman is for 3x3x3 with rank 23? 
    No, Strassen is 23 for 3x3x3. Laderman improved it? Let me recall.
    
    Actually, the minimum rank for 3x3x3 is 23 (this is a well-known result).
    For 2x2x2, it's 7 (Strassen).
    For 3x2x3, I'm not sure of the exact minimum rank. Let me just use schoolbook (18) to be safe,
    or try to find a better decomposition.
    
    Actually, for 3x2x3, we can think of it as:
    C = A * B where A is 3x2, B is 2x3.
    This is equivalent to computing 3 independent 1x2 * 2x3 products.
    Each 1x2 * 2x3 = 1x3 has rank 6 (schoolbook: 2*3=6).
    Total: 3*6 = 18.
    
    Can we do better? We can use the fact that all three rows share the same B.
    C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    C[2,:] = A[2,0]*B[0,:] + A[2,1]*B[1,:]
    
    This is like a bilinear map. The tensor is sum over i,j,k of e_i ⊗ e_j ⊗ e_k * A[i,j]*B[j,k].
    
    I'll just use schoolbook for all sub-blocks to ensure correctness, then the total is 105.
    But I want to do better. Let me use Strassen for 3x3x3 (rank 23) and schoolbook for the rest.
    
    Total with Strassen for 3x3x3 blocks:
    23 + 18 + 18 + 18 + 12 + 12 = 101. Still better than 105.
    
    Let me implement this. I'll define the Strassen 3x3x3 decomposition explicitly.
    
    The Strassen 3x3x3 decomposition (rank 23):
    I'll use the standard one. Let me define it.
    
    For 3x3 matrices X, Y, Z = X*Y:
    Using Strassen's algorithm generalized to 3x3.
    
    Actually, let me just use the schoolbook decomposition for everything to ensure correctness,
    and get rank 105. Then I can try to improve.
    
    For now, let me implement the schoolbook with the block structure to get rank 105,
    and then I'll try to replace some blocks with better decompositions.
    
    Actually, let me just implement the full schoolbook (rank 105) correctly first.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition: rank = n*m*p = 105
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
