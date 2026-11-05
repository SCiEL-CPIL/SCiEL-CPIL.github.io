"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    We use the Strassen algorithm for 2x2 blocks as a building block, but since our dimensions
    are 3, 4, 8, we need a different approach.
    
    For 3x4*4x8:
    - Input A is 3x4
    - Input B is 4x8
    - Output C is 3x8
    
    The tensor has size 3*4*8 = 96.
    
    Let's think about this more carefully. The matrix multiplication tensor for (n,m,p) has
    rank at most n*m*p (schoolbook). We want to find the minimal rank.
    
    For 3x4*4x8, let's try to use a block decomposition approach.
    
    Actually, let me think about known results. The rank of matrix multiplication tensor M_{n,m,p}
    is known for small cases. For 3x4*4x8, this is not a standard case I'm immediately familiar with.
    
    Let me try a different approach: use the fact that 4 = 2*2 and 8 = 2*4, so we can think of
    B as having 4 rows split into two groups of 2, and 8 columns split into four groups of 2.
    
    Actually, let me just implement a careful construction.
    
    One approach: use the identity that if we can decompose 2x2*2x2 with rank 7 (Strassen),
    we can potentially use this as a sub-block.
    
    Let me think about the structure:
    A is 3x4, B is 4x8.
    
    We can partition A into rows: row 0, row 1, row 2 (each of length 4)
    We can partition B into columns: cols 0-1, 2-3, 4-5, 6-7 (each pair of length 2)
    
    But this doesn't immediately help.
    
    Let me try a more direct approach. The schoolbook rank is 3*4*8 = 96.
    
    Can we do better? Let's think about using the Strassen-like idea.
    
    Actually, for the specific case of 3x4 * 4x8, let me try to construct a decomposition
    with rank less than 96.
    
    One idea: use the fact that 3 = 2+1 and 4 = 2+2, 8 = 4+4.
    
    Let me try to use a block decomposition where I split the 4-dimensional middle index
    into two parts of size 2.
    
    A = [A1 | A2] where A1, A2 are 3x2
    B = [B1; B2] where B1, B2 are 2x8
    
    Then C = A1*B1 + A2*B2
    
    This is a sum of two 3x2 * 2x8 multiplications. Each 3x2*2x8 has rank at most 3*2*8 = 48.
    So total rank would be at most 96, which is the same as schoolbook.
    
    But maybe I can share some structure between the two products.
    
    Let me try another approach: use the known rank bounds.
    
    Actually, let me just implement the schoolbook for now and see if I can find a better one.
    
    Wait, let me reconsider. The task says "rank-minimal exact tensor decomposition". 
    For 3x4*4x8, the minimal rank might be known or can be found by search.
    
    Let me try a construction based on the following idea:
    
    Use the fact that we can write the 3x4 matrix as having 3 rows, each of length 4.
    The 4x8 matrix has 4 rows, each of length 8.
    
    One approach is to use the "winograd" style or "fast matrix multiplication" ideas.
    
    Let me try to use a decomposition with rank 3*4*8 = 96 first, then try to optimize.
    
    Actually, I realize I should try to find a smarter construction. Let me think about
    the tensor structure more carefully.
    
    The matrix multiplication tensor M_{3,4,8} has components T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    where i in {0,1,2}, j in {0,1,2,3}, k in {0,...,7}.
    
    A rank-r decomposition is a set of r triples (u_r, v_r, w_r) such that
    T = sum_r u_r ⊗ v_r ⊗ w_r
    
    where u_r is in R^3, v_r is in R^4, w_r is in R^8.
    
    In the interface, U has shape (n*m, rank) = (12, rank), V has shape (m*p, rank) = (32, rank),
    W has shape (p*n, rank) = (24, rank).
    
    The convention is W[k*n+i, r], so W is indexed by (k, i) where k is the column index of B
    (which becomes the column index of C) and i is the row index of A (which becomes the row index of C).
    
    Let me implement the schoolbook decomposition first, then try to improve.
    
    For now, let me just return the schoolbook decomposition with rank 96.
    """
    n, m, p = DIMENSIONS
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
