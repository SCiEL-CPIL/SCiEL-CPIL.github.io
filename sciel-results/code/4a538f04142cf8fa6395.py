"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a low-rank tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Strategy: Partition the 3x4 matrix into two 3x2 blocks (columns 0-1 and 2-3)
    and the 4x6 matrix into two 2x6 blocks (rows 0-1 and 2-3).
    For each block pair, we need to multiply a 3x2 matrix by a 2x6 matrix.
    
    For a 3x2 * 2x6 multiplication, the tensor is T[i, j, k] where i in 0..2, j in 0..1, k in 0..5.
    We can use a rank-8 decomposition for this sub-problem (since 3*2*6=36, but we can do better).
    
    Actually, let's think about this differently. The full tensor is T[i, j, k] for i in 0..2, j in 0..3, k in 0..5.
    
    A known approach: use the rank-7 algorithm for 2x2x2 tensor (Strassen) as a building block.
    
    Let me try a direct construction. The tensor T has dimensions 3 x 4 x 6.
    T[i, j, k] = 1 if the output position (i,k) depends on input (A[i,j] * B[j,k]).
    
    In the standard embedding:
    U has shape (n*m, rank) = (12, rank)
    V has shape (m*p, rank) = (24, rank)  
    W has shape (p*n, rank) = (18, rank)
    
    The decomposition requires: for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Let's try to find a rank-24 decomposition using block structure.
    
    Split j into two groups: j in {0,1} and j in {2,3}.
    
    For group 1 (j=0,1): We have a 3x2 x 2x6 sub-tensor.
    For group 2 (j=2,3): We have a 3x2 x 2x6 sub-tensor.
    
    Each 3x2 x 2x6 sub-tensor can be decomposed. The tensor for 3x2*2x6 has rank at most 3*2*6=36,
    but we can do better.
    
    Actually, for multiplying a 3x2 matrix by a 2x6 matrix, the result is 3x6.
    The tensor is T[i, j, k] where i in 0..2, j in 0..1, k in 0..5.
    
    A simple rank-12 decomposition: for each (i,k) pair, use one rank-1 term with U[i,0]=1, V[1,k]=1, W[k,0]=1... no that's not right.
    
    Let me use the standard schoolbook for each block: 3*2*6 = 36 per block, total 72. That's worse than 72.
    
    Let me try rank-24 total. That would be rank-12 per block.
    
    For a 3x2 * 2x6 multiplication, can we do rank 12?
    The tensor T[i,j,k] for i in 0..2, j in 0..1, k in 0..5.
    
    One approach: use rank-3 for the 3x2 part and rank-4 for the 2x6 part... 
    
    Actually, let's just try to find a good decomposition numerically or use a known construction.
    
    Let me try the following: use the identity that a 3x4*4x6 multiplication can be done with rank 24.
    
    Here's a construction:
    - For each of the 3 rows of A and 6 columns of B, we need to compute the dot product of a 4-vector.
    - We can use rank-4 decompositions for each (i,k) pair, but that's 18*4=72.
    
    Let me try a different approach. Use the fact that the tensor is a Kronecker-like product.
    
    Actually, let me just implement a search for a low-rank decomposition using random initialization and gradient-free optimization, or use a known result.
    
    For now, let me try rank 24 with a structured approach.
    """
    n, m, p = 3, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try to construct a rank-24 decomposition
    # U: (12, 24), V: (24, 24), W: (18, 24)
    
    # The tensor T[i,j,k] = delta(i*m+j, ...) 
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let's use a block structure.
    # Split the 4 columns of A into two groups of 2: group0 = cols 0,1; group1 = cols 2,3
    # Split the 4 rows of B into two groups of 2: group0 = rows 0,1; group1 = rows 2,3
    
    # For group0-group0: A[:,0:2] * B[0:2,:] gives contribution to output
    # For group0-group1: A[:,0:2] * B[2:4,:] gives contribution to output
    # For group1-group0: A[:,2:4] * B[0:2,:] gives contribution to output
    # For group1-group1: A[:,2:4] * B[2:4,:] gives contribution to output
    
    # Each sub-problem is 3x2 * 2x6.
    
    # For a 3x2 * 2x6 multiplication, let's use rank 6 per sub-problem (total 24).
    # The tensor for 3x2*2x6: T[i,j,k] for i in 0..2, j in 0..1, k in 0..5
    # We need U_sub (3*2=6, rank), V_sub (2*6=12, rank), W_sub (6*3=18, rank)
    # such that sum_c U_sub[i*2+j, c] * V_sub[j*6+k, c] * W_sub[k*3+i, c] = 1
    
    # Rank 6 for 3x2*2x6: Let's try.
    # For each (i,k) pair, we need to compute sum over j in {0,1} of A[i,j]*B[j,k].
    # This is a 2-term dot product. A rank-2 decomposition for each (i,k) would use 18*2=36 terms.
    # But we can share.
    
    # Actually, let's just use rank 12 per sub-problem, total 48. Or rank 6 per sub-problem, total 24.
    
    # Let me try to construct rank 6 for 3x2*2x6.
    # U_sub: (6, 6), V_sub: (12, 6), W_sub: (18, 6)
    
    # For c in 0..5:
    # We want U_sub[i*2+j, c] * V_sub[j*6+k, c] * W_sub[k*3+i, c] to sum to 1.
    
    # Let's try: for each c, assign it to a specific (i, j, k) triple? No, that's rank 36.
    
    # Let me just use schoolbook rank 36 per sub-problem, total 72. That's the same as the original.
    
    # Let me try a different strategy: use known low-rank algorithms.
    # For 3x4*4x6, the rank is known to be at most 24 (by partitioning into 2x2 blocks and using Strassen).
    
    # Let me implement the rank-24 construction.
    
    # Partition A into 2 blocks of 3x2 (by columns), B into 2 blocks of 2x6 (by rows).
    # A = [A0 A1] where A0, A1 are 3x2
    # B = [B0; B1] where B0, B1 are 2x6
    # A*B = A0*B0 + A1*B1
    
    # Each term A0*B0 is 3x2 * 2x6.
    # For 3x2 * 2x6, we can use rank 12 (schoolbook is 3*2*6=36, but we can do 12).
    
    # Actually, 3x2*2x6: the output is 3x6. Each output element (i,k) is a dot product of length 2.
    # We can compute all 18 output elements with 12 multiplications? 
    # Let's see: we have 2 "inner" dimensions. For each (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # We can share computations.
    
    # Rank 12 for 3x2*2x6:
    # For each i in 0..2, k in 0..5:
    #   Use 2 rank-1 terms: one for j=0, one for j=1.
    #   That's 18*2 = 36. Not 12.
    
    # Hmm, let me just use rank 36 per sub-problem, total 72. Same as original.
    
    # Let me try to actually find a lower rank decomposition using optimization.
    
    # For now, let's return the schoolbook decomposition as a fallback.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
