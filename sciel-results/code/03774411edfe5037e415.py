"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]
# n=3 (rows of A), m=5 (cols of A / rows of B), p=7 (cols of B)
# U: (n*m, rank) = (15, rank)
# V: (m*p, rank) = (35, rank)
# W: (p*n, rank) = (21, rank)
# Indexing:
# U[i*m + j, r]
# V[j*p + k, r]
# W[k*n + i, r]
# Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# We want T[i,j,k] = 1 for all i,j,k (matrix mult tensor for 1s? No.)
# Wait, the tensor for matrix multiplication C = A * B where C[i,k] = sum_j A[i,j] B[j,k]
# The decomposition is: C[i,k] = sum_r U[i,j,r] V[j,k,r] ... wait.
# Standard tensor decomposition for matmul (n x m) * (m x p) -> (n x p)
# T[i,j,k] is the coefficient of x_i y_j z_k in the product?
# Actually, the tensor is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]?
# No, C[i,k] = sum_j A[i,j] B[j,k].
# The tensor T has indices i (0..n-1), j (0..m-1), k (0..p-1).
# T[i,j,k] = 1 if it corresponds to the term A[i,j]B[j,k] in C[i,k].
# So T[i,j,k] = 1 for all i,j,k?
# Let's check the schoolbook decomposition in the prompt.
# U[i*m+j][column] = 1
# V[j*p+k][column] = 1
# W[k*n+i][column] = 1
# Product: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
# In schoolbook, each column r corresponds to a unique (i,j,k) triple.
# U[i*m+j, r] is 1 only if the row index of U matches i*m+j.
# V[j*p+k, r] is 1 only if the row index of V matches j*p+k.
# W[k*n+i, r] is 1 only if the row index of W matches k*n+i.
# So for a specific column r corresponding to (i0, j0, k0):
# U[i*m+j, r] = 1 iff i=i0 and j=j0.
# V[j*p+k, r] = 1 iff j=j0 and k=k0.
# W[k*n+i, r] = 1 iff k=k0 and i=i0.
# The product is 1 if i=i0, j=j0, k=k0. Otherwise 0.
# So the tensor T[i,j,k] reconstructed is 1 if (i,j,k) is a valid triple?
# Wait, the sum is over r.
# T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
# If we use the schoolbook, T[i,j,k] = 1 for every (i,j,k) in the range.
# This means the tensor is the all-ones tensor of size n x m x p.
# Is this the matrix multiplication tensor?
# The matrix multiplication tensor T_{n,m,p} is defined by T[i,j,k] = 1 if the indices match the multiplication rule?
# Actually, the standard tensor for matmul is T[i,j,k] = 1 if i,j,k are such that C[i,k] += A[i,j]B[j,k].
# Yes, for every i in [0,n), j in [0,m), k in [0,p), the term A[i,j]B[j,k] contributes to C[i,k].
# So the tensor is indeed the all-ones tensor of size n x m x p.
# The goal is to find a rank-minimal decomposition of the all-ones tensor of size 3x5x7.
# Rank of the all-ones tensor of size n x m x p is known to be related to matmul rank?
# No, the rank of the all-ones tensor is 1 if we allow real entries?
# T[i,j,k] = 1.
# Can we do rank 1?
# U[i*m+j, 0] = u_i, V[j*p+k, 0] = v_j, W[k*n+i, 0] = w_k?
# Product: u_i * v_j * w_k.
# We need u_i * v_j * w_k = 1 for all i,j,k.
# This implies u_i = 1, v_j = 1, w_k = 1.
# So rank 1 works for the all-ones tensor!
# Why did the seed program use rank 105?
# Let's re-read the task carefully.
# "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 7 matrix."
# "The output factor follows the official trace convention W[k*n+i,r]."
# Maybe the tensor is NOT the all-ones tensor?
# In matrix multiplication, C = A B.
# C[i,k] = sum_j A[i,j] B[j,k].
# The tensor T is such that C[i,k] = sum_{j,r} A[i,j] B[j,k] ... no.
# The tensor decomposition of the matmul operation is:
# C[i,k] = sum_j A[i,j] B[j,k].
# This can be viewed as a bilinear form on A and B.
# The tensor T has indices i, j, k.
# T[i,j,k] = 1 if the term A[i,j]B[j,k] is part of C[i,k].
# Yes, T[i,j,k] = 1 for all i,j,k.
# So why is the rank not 1?
# Ah, the indices of U, V, W are coupled.
# U depends on i, j.
# V depends on j, k.
# W depends on k, i.
# In the rank 1 case:
# U[i*m+j, 0] = u_{i,j}
# V[j*p+k, 0] = v_{j,k}
# W[k*n+i, 0] = w_{k,i}
# We need sum_r u_{i,j} v_{j,k} w_{k,i} = 1 for all i,j,k.
# With rank 1, u_{i,j} v_{j,k} w_{k,i} = 1.
# This requires u_{i,j} = a_i b_j, v_{j,k} = c_j d_k, w_{k,i} = e_k f_i?
# Let's check: (a_i b_j) (c_j d_k) (e_k f_i) = (a_i f_i) (b_j c_j) (d_k e_k).
# Let A_i = a_i f_i, B_j = b_j c_j, D_k = d_k e_k.
# Product = A_i B_j D_k.
# We need A_i B_j D_k = 1 for all i,j,k.
# This is possible if A_i=1, B_j=1, D_k=1.
# So rank 1 should work?
# Let's test with the seed program's logic.
# Seed program:
# U[i*m+j][column] = 1
# V[j*p+k][column] = 1
# W[k*n+i][column] = 1
# This sets specific entries to 1, others to 0.
# This is the "standard" decomposition where each rank-1 component picks out one specific (i,j,k).
# This gives rank n*m*p.
# But we can do better.
# If we set U[i*m+j, 0] = 1 for all i,j.
# V[j*p+k, 0] = 1 for all j,k.
# W[k*n+i, 0] = 1 for all k,i.
# Then for any (i,j,k), the term is U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1.
# Sum over r (only r=0) is 1.
# So rank 1 is a valid decomposition!
# Why is the seed program rank 105?
# Maybe I am misunderstanding the tensor.
# "multiplying a 3 by 5 matrix by a 5 by 7 matrix"
# A is 3x5, B is 5x7. C is 3x7.
# C[i,k] = sum_j A[i,j] B[j,k].
# The tensor T[i,j,k] represents the coefficient of A[i,j]B[j,k] in C[i,k].
# This coefficient is 1 for all i,j,k.
# So the tensor is indeed all ones.
# And rank 1 works.
# Is there a constraint I'm missing?
# "integer/half-integer entries"
# 1 is an integer.
# So the answer should be rank 1.
# Let's double check if "rank-minimal" implies something else.
# In tensor decomposition, the rank of the all-ones tensor of size n x m x p is 1.
# Is it possible the task implies the standard basis tensors?
# No, "exact tensor decomposition for multiplying...".
# If the seed program is score 1.0, and score is rank, then rank 105 gives score 1.0.
# If I return rank 1, the score would be 1/105?
# "score is the rank"
# "The evaluator measures the raw value (lower is better)."
# "search score is seed/value" -> wait.
# "1.0 equals the seed program"
# Seed program rank = 105.
# If score is rank, then seed score is 105?
# But it says "score 1.0000".
# "search score is seed/value"
# If seed value is 105, and my value is 1, then score = 105/1 = 105.
# "above 1.0 is an improvement"
# So lower rank is better.
# If I find rank 1, I get a huge improvement.
# Let's verify if rank 1 is valid.
# U shape (15, 1). All entries 1.
# V shape (35, 1). All entries 1.
# W shape (21, 1). All entries 1.
# T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
# T[i,j,k] = 1 * 1 * 1 = 1.
# This matches the matmul tensor.
# Why would this be hard?
# Maybe the tensor is NOT all ones?
# Let's look at the indices again.
# U[i*m+j, r]
# V[j*p+k, r]
# W[k*n+i, r]
# The tensor T is defined by T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
# For matrix multiplication, we need T[i,j,k] = 1 for all i,j,k.
# Yes.
# So rank 1 is the answer.
# Why is the seed program so bad?
# Maybe it's a trick question or I'm missing a subtle constraint.
# "integer/half-integer entries"
# 1 is fine.
# Let's try to implement rank 1.

def construct():
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}

# Let's double check if there's any reason rank 1 would fail.
# If the tensor was different, e.g., if it depended on i,j,k in a more complex way.
# But for standard matrix multiplication, it's just 1s.
# Is it possible the task is about the *structure* of the decomposition?
# No, "score is the rank".
# I will bet on rank 1.

# Wait, let me check the "trace convention" mention.
# "The output factor follows the official trace convention W[k*n+i,r]."
# This just defines the index mapping for W.
# W[k*n+i, r] corresponds to the term in C[i,k] coming from A[i,j]B[j,k]?
# Yes, C[i,k] = sum_j A[i,j]B[j,k].
# The "trace" might refer to the cyclic nature of indices i->j->k->i.
# But it doesn't change the values of the tensor.

# Let's consider if the rank can be 0? No, tensor is non-zero.
# Can it be fractional? No, rank is integer.

# I will output rank 1.

# Just in case, I'll add a small check in the code to ensure it doesn't crash,
# although it's trivial.

# One more check: "integer/half-integer entries".
# If the solution required half-integers, 1 is still valid.

# What if the tensor is not all ones?
# Example: if C[i,k] = sum_j A[i,j] B[j,k] * f(i,j,k)?
# No, standard matmul.

# I'm confident rank 1 is correct.

# Let's write the code.

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: Rank = n * m * p
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each rank-1 component corresponds to a specific (i, j, k)
                # U[i*m + j, r] = 1
                U[i * m + j][r] = 1
                # V[j*p + k, r] = 1
                V[j * p + k][r] = 1
                # W[k*n + i, r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
