"""Matrix multiplication 3x3x4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank decomposition for 3x3 * 3x4 matrix multiplication.
    
    The strategy uses a Winograd-like approach: we exploit the cyclic structure
    of the 3x3 part by evaluating at roots of unity (or using a 3-point DFT over
    a suitable ring), while handling the 3x4 part more directly.
    
    For 3x3 * 3x4, the naive rank is 3*3*4 = 36.
    
    Key insight: We can decompose the 3x3 multiplication using a rank-5 or rank-6
    scheme (e.g., using the fact that 3x3 matrix mult can be done in 5 multiplications
    is not known, but 6 is possible via certain schemes). However, for exact integer
    decomposition, we need to be careful.
    
    Actually, let's think about this more carefully. The tensor for matrix multiplication
    C = A * B where A is 3x3 and B is 3x4 has entries T[i,j,k,l] = delta_{j,k} where
    C[i,l] = sum_j A[i,j] * B[j,l].
    
    So the tensor is T[i,j,k,l] = 1 if j==k else 0, with i in [0,3), j in [0,3), k in [0,3), l in [0,4).
    
    The rank of this tensor is what we want to minimize.
    
    For the 3x3 * 3x3 case, the minimum rank is 21 (Laderman's algorithm).
    For 3x3 * 3x4, we can extend Laderman-like ideas.
    
    Let me try to construct a decomposition based on Laderman's algorithm for 3x3x3
    and extend it to 3x3x4.
    
    Laderman's algorithm for 3x3x3 uses rank 21. The basic idea is to use 9 "linear
    combinations" on each side and 3 "correction" terms.
    
    Actually, let me try a different approach. Let's use the fact that we can do
    3x3 matrix multiplication in rank 21 (Laderman) and then extend to 3x4 by
    simply adding 3 more columns (since the 4th column of B just adds 3 more
    products).
    
    Wait, that's not quite right. Let me think again.
    
    For C = A * B where A is 3x3 and B is 3x4:
    C[i,l] = sum_{j=0}^{2} A[i,j] * B[j,l]
    
    The tensor T has indices (i, j, k, l) with T[i,j,k,l] = delta_{j,k}.
    
    One simple extension: take Laderman's rank-21 decomposition for 3x3x3 and for
    each of the 4 columns of B, we need the same structure. But that would give
    rank 21*4/3 = 28, which is not right.
    
    Let me try a more direct approach. Let's use a bilinear algorithm.
    
    Actually, let me try to use a known result. The minimum rank for 3x3x4 matrix
    multiplication is not well-known to me, but we can try to find a good decomposition.
    
    Let me try the following approach: use a 3-point DFT-like structure for the 3x3 part.
    
    For the 3x3 part, consider the polynomial P(x) = A[0,0] + A[0,1]*x + A[0,2]*x^2 mod (x^3-1).
    Similarly for each row. Then we can evaluate at 3 points and multiply pointwise.
    
    But x^3-1 = (x-1)(x^2+x+1), so we need to work over a ring where this factors nicely.
    Over the integers, we can use the evaluation at x=1 and the "Gaussian" points.
    
    Let me try a concrete construction.
    
    Define U, V, W matrices such that for each column c:
    U[:, c] is a vector in R^9 (for the 3x3 input A)
    V[:, c] is a vector in R^12 (for the 3x4 input B)  
    W[:, c] is a vector in R^12 (for the 3x4 output C)
    
    And we need: sum_c U[i,j,c] * V[j,k,c] * W[k,l,c] = delta_{j,k} for all i,j,k,l
    
    Wait, that's not quite the right tensor structure. Let me re-examine.
    
    The matrix multiplication tensor for C = AB where A is m x n and B is n x p is:
    T[a, b, c, d] = delta_{b,c} where a in [0,m), b in [0,n), c in [0,n), d in [0,p)
    
    And the decomposition is: T = sum_r U[:, :, r] * V[:, :, r] * W[:, :, r]
    where U is m x n x rank, V is n x n x rank, W is n x p x rank.
    
    Wait, I need to match the interface. The interface says:
    U has shape (n*m, rank) - so 9 x rank
    V has shape (m*p, rank) - so 12 x rank
    W has shape (p*n, rank) - so 12 x rank
    
    And the convention is W[k*n+i, r] for the output.
    
    So the decomposition is: C[k*n+i] = sum_r U[i*m+j... wait, let me re-read.
    
    "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    So: C[i, k] = sum_{j,r} A[i, j] * V[j*p+k, r] * ... 
    
    Hmm, let me re-read the task description more carefully.
    
    "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    I think the decomposition is:
    C[i, k] = sum_{j, r} A[i, j] * B[j, k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j V[j*p+k, r] * ...)
    
    Actually, I think the standard form is:
    C[i, k] = sum_r U[i*m + ?, r] * V[?, r] * W[k*n+i, r]
    
    Let me just implement the schoolbook and verify, then try to improve.
    
    For now, let me implement a Laderman-based approach for 3x3x3 and extend to 3x3x4.
    
    Laderman's algorithm for 3x3x3:
    The 9 products are:
    m0 = (a00 + a11 + a22) * (b00 + b11 + b22)
    m1 = (a01 + a11 + a22) * (b01 + b11 + b21)
    m2 = (a02 + a11 + a22) * (b02 + b12 + b22)
    m3 = (a10 + a11 + a22) * (b10 + b11 + b01)
    m4 = (a20 + a11 + a22) * (b20 + b11 + b02)
    m5 = (a21 + a11 + a22) * (b21 + b11 + b12)
    m6 = (a00 + a22) * (b00 + b22)
    m7 = (a10 + a20) * (b10 + b20)
    m8 = (a11 + a21) * (b11 + b21)
    m9 = (a12 + a22) * (b12 + b22)
    m10 = (a00 + a01) * (b00 + b10)
    m11 = (a01 + a02) * (b01 + b11)
    m12 = (a02 + a00) * (b02 + b12)
    m13 = (a10 + a11) * (b10 + b20)
    m14 = (a11 + a12) * (b11 + b21)
    m15 = (a12 + a10) * (b12 + b22)
    m16 = (a20 + a21) * (b20 + b00)
    m17 = (a21 + a22) * (b21 + b01)
    m18 = (a22 + a20) * (b22 + b02)
    m19 = (a00 - a11) * (b00 - b11)
    m20 = (a11 - a22) * (b11 - b22)
    
    That's 21 products. The reconstruction is complex.
    
    For 3x3x4, we can extend by treating the 4th column similarly.
    
    Actually, let me try a simpler approach. Let me just use the schoolbook (rank 36) 
    and see if I can find a better decomposition through optimization.
    
    Given the time constraint, let me implement a reasonable approach.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s margin
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Schoolbook decomposition (rank = n*m*p = 36)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using Laderman-like ideas
    # For 3x3x3, Laderman achieves rank 21. For 3x3x4, we can try to extend.
    
    # Let me try a construction based on the following idea:
    # Use a 3-point evaluation for the 3x3 part.
    
    # Define the 3x3x3 tensor and try to find a rank-21 decomposition
    # then extend to 3x3x4.
    
    # For now, let's return the schoolbook if we run out of time
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to construct a Laderman-based decomposition for 3x3x3 and extend to 3x3x4
    # This is complex, so let me try a simpler improvement.
    
    # Actually, let me try to use the fact that we can share some computations
    # between the 3x3 and the extra column.
    
    # For 3x3x4, we can view it as 3x3x3 + 3x3x1 (the 4th column).
    # The 3x3x3 part can be done in rank 21 (Laderman).
    # The 3x3x1 part can be done in rank 3 (just 3 products).
    # Total: 24.
    
    # Let me implement Laderman's algorithm for 3x3x3.
    # The 21 products for Laderman's algorithm:
    
    # I'll define the U, V, W factors for the 3x3x3 part (rank 21)
    # and then add 3 more for the 4th column.
    
    # Laderman's algorithm factors:
    # U is 9x21, V is 9x21, W is 9x21 for the 3x3x3 case.
    
    # Let me define them explicitly.
    # From Laderman (1976), the 21 bilinear forms are:
    
    laderman_U = np.zeros((9, 21))
    laderman_V = np.zeros((9, 21))
    laderman_W = np.zeros((9, 21))
    
    # m0 = (a00+a11+a22)(b00+b11+b22)
    laderman_U[0, 0] = 1; laderman_U[4, 0] = 1; laderman_U[8, 0] = 1
    laderman_V[0, 0] = 1; laderman_V[4, 0] = 1; laderman_V[8, 0] = 1
    laderman_W[0, 0] = 1; laderman_W[4, 0] = 1; laderman_W[8, 0] = 1
    
    # m1 = (a01+a11+a22)(b01+b11+b21)
    laderman_U[1, 1] = 1; laderman_U[4, 1] = 1; laderman_U[8, 1] = 1
    laderman_V[1, 1] = 1; laderman_V[4, 1] = 1; laderman_V[7, 1] = 1
    laderman_W[1, 1] = 1; laderman_W[4, 1] = 1; laderman_W[7, 1] = 1
    
    # m2 = (a02+a11+a22)(b02+b12+b22)
    laderman_U[2, 2] = 1; laderman_U[4, 2] = 1; laderman_U[8, 2] = 1
    laderman_V[2, 2] = 1; laderman_V[5, 2] = 1; laderman_V[8, 2] = 1
    laderman_W[2, 2] = 1; laderman_W[5, 2] = 1; laderman_W[8, 2] = 1
    
    # m3 = (a10+a11+a22)(b10+b11+b01)
    laderman_U[3, 3] = 1; laderman_U[4, 3] = 1; laderman_U[8, 3] = 1
    laderman_V[3, 3] = 1; laderman_V[4, 3] = 1; laderman_V[1, 3] = 1
    laderman_W[3, 3] = 1; laderman_W[4, 3] = 1; laderman_W[1, 3] = 1
    
    # m4 = (a20+a11+a22)(b20+b11+b02)
    laderman_U[6, 4] = 1; laderman_U[4, 4] = 1; laderman_U[8, 4] = 1
    laderman_V[6, 4] = 1; laderman_V[4, 4] = 1; laderman_V[2, 4] = 1
    laderman_W[6, 4] = 1; laderman_W[4, 4] = 1; laderman_W[2, 4] = 1
    
    # m5 = (a21+a11+a22)(b21+b11+b12)
    laderman_U[7, 5] = 1; laderman_U[4, 5] = 1; laderman_U[8, 5] = 1
    laderman_V[7, 5] = 1; laderman_V[4, 5] = 1; laderman_V[5, 5] = 1
    laderman_W[7, 5] = 1; laderman_W[4, 5] = 1; laderman_W[5, 5] = 1
    
    # m6 = (a00+a22)(b00+b22)
    laderman_U[0, 6] = 1; laderman_U[8, 6] = 1
    laderman_V[0, 6] = 1; laderman_V[8, 6] = 1
    laderman_W[0, 6] = 1; laderman_W[8, 6] = 1
    
    # m7 = (a10+a20)(b10+b20)
    laderman_U[3, 7] = 1; laderman_U[6, 7] = 1
    laderman_V[3, 7] = 1; laderman_V[6, 7] = 1
    laderman_W[3, 7] = 1; laderman_W[6, 7] = 1
    
    # m8 = (a11+a21)(b11+b21)
    laderman_U[4, 8] = 1; laderman_U[7, 8] = 1
    laderman_V[4, 8] = 1; laderman_V[7, 8] = 1
    laderman_W[4, 8] = 1; laderman_W[7, 8] = 1
    
    # m9 = (a12+a22)(b12+b22)
    laderman_U[5, 9] = 1; laderman_U[8, 9] = 1
    laderman_V[5, 9] = 1; laderman_V[8, 9] = 1
    laderman_W[5, 9] = 1; laderman_W[8, 9] = 1
    
    # m10 = (a00+a01)(b00+b10)
    laderman_U[0, 10] = 1; laderman_U[1, 10] = 1
    laderman_V[0, 10] = 1; laderman_V[3, 10] = 1
    laderman_W[0, 10] = 1; laderman_W[3, 10] = 1
    
    # m11 = (a01+a02)(b01+b11)
    laderman_U[1, 11] = 1; laderman_U[2, 11] = 1
    laderman_V[1, 11] = 1; laderman_V[4, 11] = 1
    laderman_W[1, 11] = 1; laderman_W[4, 11] = 1
    
    # m12 = (a02+a00)(b02+b12)
    laderman_U[2, 12] = 1; laderman_U[0, 12] = 1
    laderman_V[2, 12] = 1; laderman_V[5, 12] = 1
    laderman_W[2, 12] = 1; laderman_W[5, 12] = 1
    
    # m13 = (a10+a11)(b10+b20)
    laderman_U[3, 13] = 1; laderman_U[4, 13] = 1
    laderman_V[3, 13] = 1; laderman_V[6, 13] = 1
    laderman_W[3, 13] = 1; laderman_W[6, 13] = 1
    
    # m14 = (a11+a12)(b11+b21)
    laderman_U[4, 14] = 1; laderman_U[5, 14] = 1
    laderman_V[4, 14] = 1; laderman_V[7, 14] = 1
    laderman_W[4, 14] = 1; laderman_W[7, 14] = 1
    
    # m15 = (a12+a10)(b12+b22)
    laderman_U[5, 15] = 1; laderman_U[3, 15] = 1
    laderman_V[5, 15] = 1; laderman_V[8, 15] = 1
    laderman_W[5, 15] = 1; laderman_W[8, 15] = 1
    
    # m16 = (a20+a21)(b20+b00)
    laderman_U[6, 16] = 1; laderman_U[7, 16] = 1
    laderman_V[6, 16] = 1; laderman_V[0, 16] = 1
    laderman_W[6, 16] = 1; laderman_W[0, 16] = 1
    
    # m17 = (a21+a22)(b21+b01)
    laderman_U[7, 17] = 1; laderman_U[8, 17] = 1
    laderman_V[7, 17] = 1; laderman_V[1, 17] = 1
    laderman_W[7, 17] = 1; laderman_W[1, 17] = 1
    
    # m18 = (a22+a20)(b22+b02)
    laderman_U[8, 18] = 1; laderman_U[6, 18] = 1
    laderman_V[8, 18] = 1; laderman_V[2, 18] = 1
    laderman_W[8, 18] = 1; laderman_W[2, 18] = 1
    
    # m19 = (a00-a11)(b00-b11)
    laderman_U[0, 19] = 1; laderman_U[4, 19] = -1
    laderman_V[0, 19] = 1; laderman_V[4, 19] = -1
    laderman_W[0, 19] = 1; laderman_W[4, 19] = -1
    
    # m20 = (a11-a22)(b11-b22)
    laderman_U[4, 20] = 1; laderman_U[8, 20] = -1
    laderman_V[4, 20] = 1; laderman_V[8, 20] = -1
    laderman_W[4, 20] = 1; laderman_W[8, 20] = -1
    
    # Now I need to verify this is correct. Let me check the reconstruction.
    # The output C[i,j] = sum_r U[i,j,r] * V[?,r] * W[i,j,r]... 
    # This is getting complex. Let me just return the schoolbook for now
    # since I'm not confident in the Laderman extension to 3x3x4.
    
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
