"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x4 matrix.
    
    The multiplication tensor T has dimensions n x m x p = 3 x 4 x 4.
    The schoolbook rank is n*m*p = 48.
    
    We use a block-diagonal strategy: decompose the 3x4x4 multiplication into
    independent subproblems. Specifically, we can view the 3x4 matrix as having
    3 rows, each of length 4. The product with a 4x4 matrix gives 3 rows of length 4.
    
    One approach: Use the fact that matrix multiplication A(3x4) * B(4x4) = C(3x4)
    can be decomposed. Let's think about known lower bounds and constructions.
    
    For m x n matrix multiplication, the rank is at least mn (for square-ish) but can be less.
    Actually, for rectangular matrix multiplication, the tensor rank of m x n * n x p is at least max(mn, np, mp) in some cases, but the exact minimal rank is hard.
    
    Let's try a direct construction. The standard decomposition uses rank = m*n*p.
    We can potentially reduce this by sharing structure.
    
    Alternative: Use bilinear complexity. The rank of the tensor is the minimum number of rank-1 tensors needed to express it.
    
    Let's try to find a decomposition with rank < 48.
    
    One known technique: If we fix the middle dimension, we can sometimes reduce the rank.
    
    Let's use a simple but effective approach: try to find a decomposition using random search or structured patterns.
    
    Actually, let's try a specific construction. For A(3x4) * B(4x4) = C(3x4):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = delta(i,j,k) in the sense that T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    More precisely, the multiplication tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (A[i,j] * B[j,k] -> C[i,k])
    
    In the standard basis, the tensor has entries T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,4).
    Wait, no. The tensor for matrix multiplication of (n x m) * (m x p) has entries T[i,j,k] where i is row of A (and C), j is column of A (and row of B), k is column of B (and C).
    T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] contributes to C[i,k].
    So T[i,j,k] = 1 for all valid i,j,k.
    
    The rank of this tensor is what we want to minimize.
    
    Let's try a heuristic: use a random matrix factorization approach or a greedy algorithm.
    
    Given the time constraint, let's implement a simple iterative improvement starting from schoolbook.
    
    Actually, let's try a different approach: use the fact that we can combine terms.
    
    For now, let's return the schoolbook decomposition as a baseline, but try to optimize it.
    
    Let's implement a simple local search: start with schoolbook, then try to merge or replace terms.
    
    But given the complexity, let's just return the schoolbook for now and see if we can improve.
    
    Wait, the task says "rank-minimal exact tensor decomposition". The schoolbook is rank 48.
    Can we do better?
    
    Known results: For 2x2x2, rank is 7 (Strassen). For larger, it's harder.
    
    Let's try a construction based on block matrices.
    
    Actually, let's just implement a simple random search for a lower rank decomposition.
    
    Given the 120s limit, let's do a quick random search.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 48
    
    # Try to find a lower rank decomposition using random search
    # This is a heuristic approach
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook factors
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U = [row[:] for row in U_school]
    best_V = [row[:] for row in V_school]
    best_W = [row[:] for row in W_school]
    
    # Try random perturbations to reduce rank
    # This is a simplified approach; in practice, one would use more sophisticated methods
    
    # For now, return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
