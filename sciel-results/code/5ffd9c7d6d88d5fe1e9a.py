"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Output factor follows W[k*n+i, r] convention.
    
    The tensor T has entries T[i, j, k] = 1 if (i,j,k) is a valid triple,
    where i in [0,3), j in [0,4), k in [0,5).
    
    We seek rank-r factor matrices U (n*m x r), V (m*p x r), W (p*n x r)
    such that for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Strategy: Use Strassen's 7-multiplication algorithm for 2x2 blocks where applicable,
    combined with schoolbook for the remaining parts.
    
    For 3x4x5, we can partition:
    - The 3x4 matrix A into blocks
    - The 4x5 matrix B into blocks
    
    One approach: Use a 2x2x2 Strassen decomposition on overlapping 2x2 blocks.
    
    Actually, let's think about this more carefully.
    
    The matrix multiplication tensor for (n,m,p) has dimension n*m*p.
    The schoolbook rank is n*m*p = 60.
    
    For 2x2x2, Strassen gives rank 7 instead of 8.
    
    For 3x4x5, we can try to use Strassen on parts of the computation.
    
    Let's try a different approach: use the fact that we can decompose the problem
    into smaller subproblems and apply Strassen where 2x2x2 subproblems exist.
    
    Partition A (3x4) and B (4x5):
    - We can view this as computing C = A * B where C is 3x5.
    
    One strategy: Split the 4 middle dimension into 2+2.
    A = [A1 | A2] where A1 is 3x2, A2 is 3x2
    B = [B1; B2] where B1 is 2x5, B2 is 2x5
    C = A1*B1 + A2*B2
    
    Each A_i*B_i is a 3x2 * 2x5 = 3x5 product.
    
    For a 3x2 * 2x5 product, the schoolbook rank is 3*2*5 = 30.
    Can we do better?
    
    Alternatively, split differently. Let's try to use Strassen on 2x2 blocks.
    
    Actually, let me just implement a search for a low-rank decomposition using
    the known Strassen algorithm extended to our case, or use a direct algebraic approach.
    
    Given the complexity, let me try to construct a rank-42 or lower decomposition
    by combining Strassen-style tricks.
    
    For now, let's return the schoolbook decomposition as a safe baseline,
    but try to improve it with Strassen on 2x2 sub-blocks.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook baseline
    rank_sb = n * m * p  # 60
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to improve using Strassen on 2x2x2 sub-blocks
    # Strategy: Partition the 4 middle dimension as 2+2
    # A (3x4) = [A_left (3x2) | A_right (3x2)]
    # B (4x5) = [B_top (2x5); B_bottom (2x5)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # Each term is 3x2 @ 2x5. 
    # For 3x2 @ 2x5, we can further use Strassen if we split 3 into 2+1 and 5 into 2+2+1.
    
    # Let's try a direct approach: use known low-rank decompositions.
    # For 2x2x2, Strassen has rank 7.
    
    # Let's construct a decomposition using Strassen on 2x2 blocks where possible.
    
    # Partition indices:
    # i: 0,1,2 (n=3) -> split as (0,1) and (2)
    # j: 0,1,2,3 (m=4) -> split as (0,1) and (2,3)  
    # k: 0,1,2,3,4 (p=5) -> split as (0,1), (2,3), (4)
    
    # This gives us blocks. Let's use Strassen for the 2x2x2 parts.
    
    # Strassen 2x2x2 decomposition (rank 7):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our problem, we need to handle the non-2x2x2 parts with schoolbook.
    
    # Let me just return the schoolbook for now to ensure correctness,
    # and note that further optimization is needed.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
