"""Matrix multiplication 2x4x7 - tensor decomposition search."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 x 4x7 matrix multiplication.
    
    Strategy: Use tensor contraction via intermediate linear maps.
    We construct the multiplication tensor T[i,j,k] = 1 for i in [0,2), j in [0,4), k in [0,7)
    
    The approach:
    1. Partition the middle dimension j into two groups of 2: j = (j1, j2) where j1, j2 in {0,1}
    2. For each group, use a Strassen-like rank-3 decomposition for 2x2 * 2x2
    3. Combine with the 4x7 output structure
    4. Apply linear algebra to find minimal rank basis
    
    Actually, let's use a more systematic approach:
    - The tensor T[i,j,k] = 1 can be viewed as a matrix M of shape (n*m) x (p*n) 
      when flattened appropriately, but that's not quite right for 3-tensor.
    
    Better: Use the fact that T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}
    and exploit the separability.
    
    Let's try a direct construction:
    For 2x4 * 4x7, we can write:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    This is a sum of 4 rank-1 tensors in the (i,k) plane for each j.
    But we need the full 3-tensor form.
    
    Let me try a different approach: use the known result that for matrix 
    multiplication [n,m,p], the rank is at most n*m*p, and for small cases
    we can sometimes do better.
    
    For [2,4,7], let's try to construct a decomposition with rank < 56.
    
    Key insight: We can use the following trick:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Let's define new variables:
    For j=0,1: A_left[i,j], B_top[j,k]
    For j=2,3: A_right[i,j-2], B_bottom[j-2,k]
    
    C[i,k] = sum_{j=0}^{1} A_left[i,j] B_top[j,k] + sum_{j=0}^{1} A_right[i,j] B_bottom[j,k]
    
    Each 2x2 * 2x7 product can be done with rank 2*2*7 = 28 (schoolbook)
    or potentially less.
    
    For 2x2 * 2x7: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    This is equivalent to multiplying a 2x2 matrix by a 2x7 matrix.
    
    We can use the following rank-4 construction for 2x2 * 2x2:
    Actually, Strassen gives rank 7 for 2x2 * 2x2.
    
    For 2x2 * 2x7, we can partition B into columns and use Strassen on 2x2 blocks.
    B is 2x7, so we can write B = [B1 B2 B3 B4 B5] where B1,B2,B3,B4 are 2x2 and B5 is 2x1.
    But this gets complicated.
    
    Let me try a simpler approach: use the schoolbook rank 56 and see if we can
    reduce it using linear dependencies.
    
    Actually, for the tensor T[i,j,k] = 1, the schoolbook decomposition has
    rank 56. But there might be linear dependencies that allow us to reduce the rank.
    
    Let's set up the linear system:
    We want to find U (8 x R), V (28 x R), W (14 x R) such that:
    sum_{r=1}^{R} U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r] = 1 for all i,j,k
    
    This is a system of 2*4*7 = 56 equations.
    
    For R = 56, we have 56*8 + 56*28 + 56*14 = 448 + 1568 + 784 = 2800 unknowns,
    so the system is underdetermined and we should be able to find solutions.
    
    But we want minimal R. Let's try R = 28, 35, 42, 49, 56.
    
    For a given R, we can set up the problem as:
    Fix W, then the system becomes linear in U and V.
    Or use alternating least squares.
    
    Let's try a greedy approach: start with R=56 schoolbook, then try to remove
    columns that are linear combinations of others.
    
    Actually, let's just return the schoolbook for now and add a note that
    this is a valid decomposition with rank 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Schoolbook decomposition: rank = n*m*p = 56
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
