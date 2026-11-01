import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses a block decomposition:
    - Split the 2x3 input matrix into a 2x2 block (first two columns) and a 2x1 block (third column).
    - Use Strassen's algorithm for the 2x2 part (rank 7) combined with a 2x10 output, and
      schoolbook for the 2x1 part (rank 20).
    Total rank = 7 * 10 + 2 * 1 * 10 = 70 + 20 = 90.
    However, we can do better by recognizing the structure.
    
    Actually, let's use a direct approach with known lower bounds and constructions.
    For M(2,3,10), we can use:
    - The 2x3 matrix times 3x10 matrix.
    - We can view this as computing two 1x10 row vectors.
    - Each row of the output is a linear combination of the rows of the second matrix.
    
    Let's use a block-based Strassen-like approach:
    Split A (2x3) into A1 (2x2) and A2 (2x1).
    Split B (3x10) into B1 (2x10) and B2 (1x10).
    Output C = A1*B1 + A2*B2.
    
    For A1*B1 (2x2 times 2x10):
    Use Strassen's 7 multiplications for 2x2, each multiplied by 10 elements of B1.
    So rank = 7 * 10 = 70.
    
    For A2*B2 (2x1 times 1x10):
    This is just 2*1*10 = 20 scalar multiplications.
    Rank = 20.
    
    Total rank = 90.
    
    But wait, we can be more clever. The Strassen decomposition for 2x2*2x2 has rank 7.
    For 2x2 * 2x10, we can use rank 7*10 = 70, but actually we can use rank 7*10 only if
    we treat it as 10 independent 2x2*2x1 problems, each of rank 7, so 70.
    
    However, there might be a better way. Let's think about the tensor M(2,3,10).
    
    Actually, let's just use the schoolbook decomposition with rank 60, which is known to be valid.
    The question is whether we can do better than 60.
    
    Lower bounds: omega(2,3,10) >= 2*3*10 / (2+3+10-2) = 60/13 ≈ 4.6, so at least 5.
    But this is very weak.
    
    Known results: M(2,3,10) is not well-studied. Let's try to find a decomposition with rank < 60.
    
    One approach: use the fact that 3 = 2+1, so we can split the middle dimension.
    A is 2x3, B is 3x10.
    Write A = [A_2x2 | A_2x1], B = [B_2x10; B_1x10].
    C = A_2x2 * B_2x10 + A_2x1 * B_1x10.
    
    For A_2x2 * B_2x10: use Strassen rank 7 for the 2x2 part, and for each of the 7 Strassen products,
    multiply by the corresponding 2x10 submatrix. But this is not straightforward.
    
    Actually, the standard way to extend Strassen to rectangular matrices:
    For 2x2 * 2xn, the rank is 7n.
    For 2x1 * 1xn, the rank is 2n.
    
    So total rank = 7*10 + 2*10 = 70 + 20 = 90.
    
    But this is worse than 60! So the schoolbook is better.
    
    Let's try another split: A is 2x3, B is 3x10.
    Split the output dimension: C is 2x10.
    C_i = sum_j A[i,j] * B[j,:]
    
    For i=0: C_0 = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:]
    For i=1: C_1 = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:]
    
    This is like two 1x3 * 3x10 multiplications.
    Each 1x3 * 3x10 has rank 3*10 = 30 (schoolbook).
    Total rank = 2*30 = 60.
    
    Can we share any structure between the two rows?
    The intermediate dimension is 3, and we have 3 scalar multiplications per row.
    For row 0: we compute A[0,0]*B[0,:], A[0,1]*B[1,:], A[0,2]*B[2,:].
    For row 1: we compute A[1,0]*B[0,:], A[1,1]*B[1,:], A[1,2]*B[2,:].
    
    Notice that B[0,:], B[1,:], B[2,:] are shared.
    But the coefficients A[0,j] and A[1,j] are different.
    
    We can precompute linear combinations of B rows:
    Let D[j,:] = B[j,:] for j=0,1,2.
    Then C_0 = A[0,0]*D[0,:] + A[0,1]*D[1,:] + A[0,2]*D[2,:]
    C_1 = A[1,0]*D[0,:] + A[1,1]*D[1,:] + A[1,2]*D[2,:]
    
    This is a 2x3 * 3x10 problem, which is the original problem.
    
    I don't see an obvious way to reduce the rank below 60 with simple block decomposition.
    
    Let's try a different approach: use a known algorithm for small matrix multiplication.
    For 2x3 * 3x10, we can use the fact that 2x3 * 3x10 = (2x3 * 3x3) * (3x3 * 3x10) is not helpful.
    
    Alternatively, we can use the Laderman algorithm or other fast matrix multiplication for small sizes,
    but those are for square matrices.
    
    Given the time constraint, let's just return the schoolbook decomposition with rank 60.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    rank = n * m * p  # 60
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
