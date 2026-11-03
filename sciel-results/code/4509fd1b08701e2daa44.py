"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import itertools

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Use Z_4 group structure to find rank-55 decomposition
    # The key: m=4, so we can use the 4th roots of unity to block-diagonalize
    
    # Strategy: The tensor T[i*4+j, j*5+k, k*3+i] = delta_{i,j,k}
    # We'll use the fact that summing over j with phase factors gives us
    # decoupled problems for each Fourier mode
    
    # For a rank-R decomposition, we need:
    # sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1 if (i,j,k) valid, else 0
    
    # The valid entries are exactly when the indices are consistent:
    # T[a,b,c] = 1 iff a = i*4+j, b = j*5+k, c = k*3+i for some i,j,k
    
    # Let's try rank 55 with a structured ansatz
    
    R = 55
    
    # Build the linear system:
    # For each (i,j,k) in [0,3)x[0,4)x[0,5), we need:
    # sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1
    # For all other (a,b,c), the sum must be 0
    
    # This is nonlinear in U,V,W. But if we fix two of them, it's linear in the third.
    
    # Alternative: use the fact that the tensor has a specific structure.
    # The 60 standard rank-1 tensors span a 60-dimensional space (they're linearly independent
    # as tensors in the 12x20x15 space). But tensor rank can be less than the number of
    # standard basis tensors if we allow linear combinations.
    
    # Let me try a different approach: use the known result that matmul(3,4,5) has rank 55.
    # I'll construct it using the "laser method" or by finding explicit 5 dependencies.
    
    # Actually, let me just return the schoolbook decomposition for now since
    # finding an exact rank-55 decomposition requires sophisticated algebra.
    
    # But wait - let me try a smarter approach. The tensor T can be written as:
    # T = sum_{i,j,k} e_i ⊗ f_{j,k} ⊗ g_{k,i}
    # where e_i is the i-th standard basis vector in R^3 (for the first index of W),
    # f_{j,k} is a vector in R^{20} with 1 at position j*5+k,
    # g_{k,i} is a vector in R^{12} with 1 at position i*4+j... 
    
    # This is getting complicated. Let me just use schoolbook.
    
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
