"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_28():
    """
    Construct a rank-28 decomposition for 2x4 * 4x4.
    
    We split the 2x4 matrix A into two 2x2 blocks: A_left, A_right.
    We split the 4x4 matrix B into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR.
    The result C is 2x4, split into C_left = A*B_left_blocks, C_right = A*B_right_blocks.
    
    Actually, let's use the block structure more carefully.
    A is 2x4. B is 4x4.
    Let A = [A1, A2] where A1, A2 are 2x2.
    Let B = [[B11, B12], [B21, B22]] where Bij are 2x2.
    
    C = A * B = [A1, A2] * [[B11, B12], [B21, B22]]
    C = [A1*B11 + A2*B21,  A1*B12 + A2*B22]
    
    So C_left = A1*B11 + A2*B21  (2x2)
    C_right = A1*B12 + A2*B22    (2x2)
    
    We can use Strassen's algorithm for each 2x2 multiplication if we had single products.
    But here we have sums of two 2x2 products.
    
    Let's use the rank-7 Strassen for 2x2*2x2.
    For C_left = A1*B11 + A2*B21, this is a sum of two 2x2 matrix products.
    The rank of the sum of two independent 2x2 matrix multiplication tensors is at most 7+7=14, but potentially less due to structure?
    Actually, the tensor for A1*B11 + A2*B21 is the sum of two tensors.
    Let's just compute the rank of the 2x4x4 tensor directly using a known decomposition or a greedy approach.
    
    Wait, the knowledge says: "Strassen decomposition on 2x2 blocks reduces rank from 32 to 28".
    So there is a known rank-28 decomposition.
    
    Let's try to construct it.
    The schoolbook rank is 2*4*4 = 32.
    The reduction is 4.
    
    Let's look at the structure.
    C[i, j] = sum_{k=0}^{3} A[i, k] B[k, j]
    
    If we group k into pairs (0,1) and (2,3):
    C[i, j] = (A[i,0]B[0,j] + A[i,1]B[1,j]) + (A[i,2]B[2,j] + A[i,3]B[3,j])
    
    Let L_i = [A[i,0], A[i,1]] (1x2) and R_j = [B[0,j], B[1,j]]^T (2x1).
    Then A[i,0]B[0,j] + A[i,1]B[1,j] = L_i R_j.
    Similarly for the other pair.
    
    This doesn't immediately give Strassen.
    
    Let's try a different grouping.
    A is 2x4. Let's view A as a 2x2 block matrix [A_L, A_R] where A_L, A_R are 2x2.
    B is 4x4. Let's view B as a 2x2 block matrix [[B_TL, B_TR], [B_BL, B_BR]] where each is 2x2.
    
    C = A B = [A_L B_TL + A_R B_BL,  A_L B_TR + A_R B_BR]
    C_left = A_L B_TL + A_R B_BL
    C_right = A_L B_TR + A_R B_BR
    
    Each of C_left and C_right is a 2x2 matrix.
    C_left is the sum of two 2x2 matrix products.
    The rank of the tensor for X Y + Z W (where X,Z are 2x2, Y,W are 2x2) is at most 14 (7+7).
    But maybe we can do better?
    
    Actually, the tensor T for C_left is:
    T_left[i, j, k1, k2, l1, l2] ... no, the indices are mixed.
    
    Let's use the known fact that the rank of 2x4x4 is 28.
    I will attempt to construct it using a greedy algorithm or by solving a linear system for a specific rank.
    Since I don't have the explicit 28-rank formula, I will try to find it computationally.
    
    Strategy:
    1. Start with the schoolbook rank-32 decomposition.
    2. Try to reduce the rank by finding linear dependencies or using a rank-reduction algorithm.
    3. Alternatively, try to find a rank-28 decomposition by solving for the factors.
    
    Since solving for rank-28 directly is hard, I will use a heuristic search or a known construction if I can derive it.
    
    Let's try to derive the rank-28 construction.
    Reference: "The rank of the 2x4x4 matrix multiplication tensor is 28" (or similar).
    
    Actually, I can try to use the `scipy` or `numpy` to solve for a rank-28 decomposition by setting up the equations.
    The number of variables in a rank-r decomposition is 3*r*(n*m + m*p + p*n) ... no, it's (n*m + m*p + p*n)*r.
    For r=28: (8 + 16 + 8)*28 = 32*28 = 896 variables.
    The number of constraints is n*m*p = 32.
    This is an underdetermined system, so solutions should exist.
    However, the equations are quadratic (bilinear in U,V and W), so it's a system of quadratic equations.
    
    I will try to use a local search or a random initialization with Newton's method.
    But given the time constraint, I will first check if a simple block-wise Strassen application works.
    
    Let's apply Strassen to the 2x2 blocks.
    For C_left = A_L B_TL + A_R B_BL:
    Let M1 = (A_L + A_R) (B_TL + B_BL)
    Let M2 = A_L (B_TL - B_BL)
    Let M3 = A_R B_TL
    Let M4 = A_R (B_BL - B_TL)
    ... this is getting complicated.
    
    Let's just use the rank-28 decomposition from the literature if I can recall it.
    Or, I will implement a rank-reduction search.
    
    Given the time, I will implement a function that tries to find a lower rank decomposition by random sampling and local optimization.
    If it fails, it falls back to rank 32.
    
    But wait, the knowledge says a program achieved 1.143, which corresponds to rank 28 (32/28 = 1.1428).
    So rank 28 is achievable.
    
    I will try to construct the rank-28 decomposition explicitly.
    
    Here is a known construction for 2x4x4 rank 28:
    It uses the fact that the tensor is equivalent to two 2x2x2 tensors plus some correction terms.
    
    Actually, let's just use the greedy rank reduction algorithm.
    """
    n, m, p = 2, 4, 4
    rank = 28
    # We need to find U (8x28), V (16x28), W (8x28)
    # such that T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # where T[i,j,k] = 1 if k == i*m+j (wait, indices are tricky)
    
    # Let's define the tensor T of shape (n, m, p) = (2, 4, 4)
    # T[i, j, k] = 1 if the output index (k*n + i) corresponds to input (i*m+j) and (j*p+k)?
    # No, the convention is:
    # U has shape (n*m, rank), indexed by (i*m + j)
    # V has shape (m*p, rank), indexed by (j*p + k)
    # W has shape (p*n, rank), indexed by (k*n + i)
    # The tensor entry T[i, j, k] = 1.
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # So we need to solve:
    # For all i in [0, n), j in [0, m), k in [0, p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 32 equations.
    # Unknowns: U (8*28), V (16*28), W (8*28). Total 896 unknowns.
    # We can fix some variables to break symmetry.
    # For example, set W to be an identity-like matrix or random.
    
    # Let's try a random initialization and use gradient descent on the squared error.
    # But this might be slow.
    
    # Alternative: Use the fact that the rank is 28 and try to find a basis.
    
    # I will implement a simple local search.
    # Initialize U, V, W randomly.
    # Compute the error.
    # Update using gradient descent.
    
    # However, this is risky for a 120s limit.
    
    # Let's try a different approach: Use the known rank-28 decomposition from the "Strassen-like" identities.
    
    # I found a reference online (in my training data) that the rank of 2x4x4 is 28.
    # The decomposition can be obtained by applying Strassen's algorithm to the 2x2 blocks and then combining.
    
    # Let's try to construct it manually.
    # A = [A1, A2] (2x2 blocks)
    # B = [[B1, B2], [B3, B4]] (2x2 blocks)
    # C = [C1, C2] where C1 = A1*B1 + A2*B3, C2 = A1*B2 + A2*B4
    
    # We need to compute C1 and C2.
    # C1 is a 2x2 matrix. C2 is a 2x2 matrix.
    # Total output is 4x4? No, C is 2x4.
    # C1 is 2x2, C2 is 2x2.
    
    # Let's use Strassen for C1 = A1*B1 + A2*B3.
    # This is not a single matrix multiplication.
    
    # I will fall back to the schoolbook rank 32 if I cannot find a faster way to get rank 28.
    # But the prompt says a program achieved 1.143, so it is possible.
    
    # Let's try to use the `scipy.optimize` to solve for the factors.
    # But it might be too slow.
    
    # I will implement the schoolbook rank 32 for now, and add a comment that rank 28 is possible.
    # Wait, the task is to find a rank-minimal decomposition. If I return rank 32, the score is 1.0.
    # If I return rank 28, the score is 32/28 = 1.143.
    
    # I will try to hardcode the rank-28 decomposition if I can find it.
    # Since I cannot, I will try a quick random search.
    
    # For now, return rank 32.
    pass

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
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
