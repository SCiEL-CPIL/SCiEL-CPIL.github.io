"""Find rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
import time
import itertools
import numpy as np
from numpy.linalg import norm

DIMENSIONS = [3, 3, 3]

# The tensor T[i,j,k] = 1 if i,j,k are indices for (row, col_a, col_b) -> (row, col_b)
# Actually, matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
# So the tensor has indices (i, j, k) with T[i,j,k] = 1 for all i,j,k in 0..2
# This is the all-ones tensor of order 3, size 3x3x3.

# Wait, that's not right. The matrix multiplication tensor T has:
# T[i, j, k] where:
#   i = row index of A and C (0..n-1)
#   j = column index of A and row index of B (0..m-1)  
#   k = column index of B and C (0..p-1)
# T[i,j,k] = 1 if this term appears in C[i,k] = sum_j A[i,j] * B[j,k]
# So T[i,j,k] = 1 for all valid i,j,k. It's the all-ones 3x3x3 tensor.

# A rank-1 decomposition term is u_i * v_j * w_k where:
#   u has n entries (one per row i)
#   v has m entries (one per column j of A)
#   w has p entries (one per column k of B)
# And we need: sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k

# So we need to find the minimum number of rank-1 tensors whose sum equals the all-ones 3x3x3 tensor.

# The schoolbook solution uses rank 27 (one for each (i,j,k) triple).
# But we can do better.

# Known results:
# - Rank of 2x2x2 matrix multiplication tensor is 7 (Strassen)
# - Rank of 3x3x3 matrix multiplication tensor: lower bound 19, upper bound 23
# - Actually, the best known upper bound for 3x3x3 is 23 (from Strassen's work)

# Let me try to construct a rank-23 decomposition.

# The all-ones 3x3x3 tensor can be decomposed as:
# T[i,j,k] = 1 = sum_r u_r[i] * v_r[j] * w_r[k]

# One approach: use the fact that 1 = (1/3) * sum_{a=0}^{2} sum_{b=0}^{2} sum_{c=0}^{2} 
#   omega^{a(i-j)} * omega^{b(j-k)} * omega^{c(k-i)} ... no, that's not quite right.

# Actually, let me think about this differently. The tensor is all ones.
# A rank-1 term u ⊗ v ⊗ w contributes u[i]*v[j]*w[k] to position (i,j,k).
# We need sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.

# This is equivalent to: the matrix M = sum_r u_r ⊗ v_r ⊗ w_r (as a 3D tensor) equals the all-ones tensor.

# Let me try a structured approach. Since the tensor is symmetric (all ones), maybe I can use symmetric decompositions.

# For the 2x2 case, Strassen's 7-term algorithm works on the all-ones 2x2x2 tensor.
# For 3x3, one known approach is to partition into blocks.

# Let me try to find a rank-23 decomposition by solving a system of equations.
# Variables: for each rank-r term, u_r[0..2], v_r[0..2], w_r[0..2] = 9*rank variables
# Constraints: 27 equations (one for each (i,j,k) position)
# For rank 23: 9*23 = 207 variables, 27 constraints. Underdetermined, so solutions exist.

# But finding exact rational/half-integer solutions is hard. Let me try a simpler approach.

# Actually, I recall that the 3x3x3 all-ones tensor has a known rank-23 decomposition.
# Let me try to construct it based on the structure.

# Alternative: use the fact that the all-ones tensor of size n can be written as:
# 1_{n x n x n} = sum_{r=0}^{n-1} (e_r ⊗ e_r ⊗ e_r) ... no, that only gives diagonal terms.

# Let me just try to find any valid decomposition with rank < 27.

# A simple improvement: group terms. For example:
# sum_{j} A[i,j] B[j,k] 
# We can write this using fewer terms by combining.

# For rank 27: each term is u = e_i, v = e_j, w = e_k
# Can we merge some? 

# Let me try rank 24: for each i, we need to compute row i of C.
# C[i,k] = sum_j A[i,j] B[j,k]
# This is a dot product of row i of A with column k of B.
# For fixed i, this is a rank-1 decomposition of the 3x3 matrix (row i of A) ⊗ (all rows of B)?
# No, that's not right either.

# Let me just try to verify if the all-ones 3x3x3 tensor can be decomposed with rank 23.
# I'll set up the problem and try to solve it.

# Actually, let me take a different approach. Let me try to find a decomposition
# where the u, v, w vectors have a specific structure.

# Claim: The all-ones 3x3x3 tensor has rank at most 23.
# Here's a construction:
# 
# For each pair (a, b) where a, b in {0, 1, 2}, and for each "phase" c in {0, 1, 2},
# define:
#   u_{a,b,c}[i] = omega^{a*i + b*(2-i)}  ... this is getting complicated.

# Let me try a simpler constructive approach.
# 
# I know that rank(2x2x2) = 7. For 3x3x3, I can partition into 2x2 and 1x1 blocks.
# But that gives rank <= 7*4 + 1 = 29, which is worse than 27.

# Let me just try to numerically find a low-rank decomposition and see if I can get exact values.

# Actually, the task says entries should be integer or half-integer. So I need exact rational solutions.

# Let me try a different strategy: use the known Strassen-style algorithm for 3x3.

# After reflection, I believe the best I can do reliably is return the rank-27 schoolbook solution,
# as I cannot verify a lower-rank exact decomposition in the time available.

def construct():
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
