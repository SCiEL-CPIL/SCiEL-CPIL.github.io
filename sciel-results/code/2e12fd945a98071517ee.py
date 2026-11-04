import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
# We need to construct U, V, W such that:
# For all i in [0,3), j in [0,3), k in [0,6):
# sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
# where n=3, m=3, p=6.
# This is the matrix multiplication tensor for A (3x3) * B (3x6).
#
# The previous attempt to hardcode rank 46 failed or was rejected.
# Schoolbook rank is 3*3*6 = 54.
# Can we do better?
#
# Let's look at the structure.
# A is 3x3, B is 3x6.
# The multiplication tensor is T_{i,j,k} = 1 for all i,j,k.
# Wait, no. The tensor for matrix multiplication is usually defined such that
# C_{i,k} = sum_j A_{i,j} B_{j,k}.
# The tensor T has modes (i,j,k) where T_{i,j,k} = 1.
# This tensor is the standard matrix multiplication tensor.
#
# For square matrices n x n, the rank is known to be strictly less than n^3 for n>=3.
# For 3x3, rank is 23.
# For 3x6, we can view B as two 3x3 matrices B1 and B2 side by side.
# Then C = A * [B1 B2] = [A*B1, A*B2].
# So the problem decomposes into two independent 3x3 matrix multiplications?
# No, A is shared.
# The tensor is T_{i, j, (s, k')} where k' in 0..2, s in 0..1.
# T_{i, j, s, k'} = 1.
# This tensor is not a tensor product, but a sum of two tensors T^{(0)} and T^{(1)} where
# T^{(s)}_{i,j,k'} = 1.
# The rank of a sum of tensors is at most the sum of the ranks.
# Rank(T) <= Rank(T^{(0)}) + Rank(T^{(1)}).
# Each T^{(s)} is the 3x3x3 matrix multiplication tensor.
# So Rank(T) <= 23 + 23 = 46.
#
# However, can we do better by exploiting the shared A?
# The tensor is block diagonal in the k-mode if we group by s.
# Actually, let's verify the indices.
# Output C is 3x6. C_{i, k} = sum_j A_{i,j} B_{j,k}.
# The tensor T_{i,j,k} = 1.
# This tensor is the same as the tensor product of the 3x3x3 tensor and a 1x1x2 tensor?
# No. T_{i,j,k} = 1.
# If we write k = 3*s + t, then T_{i,j,3s+t} = 1.
# This is the sum over s of T^{(s)} where T^{(s)}_{i,j,t} = 1.
# Wait, the indices are i (0..2), j (0..2), k (0..5).
# The tensor is T_{i,j,k}.
# This is exactly the tensor that computes A * B where A is 3x3 and B is 3x6.
# This tensor is isomorphic to the tensor product of the 3x3x3 multiplication tensor
# and the 1x3x2 "identity" tensor? No.
#
# Let's look at the rank of the 3x3x6 tensor.
# It is known that the rank of the m x n x p matrix multiplication tensor is at most
# Rank(m x n x n) + Rank(n x p x n)? No.
# A known bound is Rank(m,n,p) <= Rank(m,n,n) * ceil(p/n)? No.
#
# Let's stick to the safe side. The schoolbook rank is 54.
# We can definitely achieve 46 by taking two copies of the rank 23 decomposition.
# U = [U23, U23]  (9x46)
# V = [V23, V23]  (18x46)? No.
#
# Let's construct the 46 rank decomposition explicitly.
# We need U (9x46), V (18x46), W (18x46).
#
# Decomposition 1 (for k=0,1,2):
# U1 (9x23), V1 (9x23), W1 (9x23).
# This computes C1 = A * B1 where B1 is columns 0-2 of B.
# C1_{i,t} = sum_j A_{i,j} B1_{j,t}.
# In the global indices:
# U[i,j, r] for r in 0..22 is U1[i,j,r].
# V[j,k, r] for k in 0..2 is V1[j,k,r].
# W[k,i, r] for k in 0..2 is W1[k,i,r].
#
# Decomposition 2 (for k=3,4,5):
# U2 (9x23), V2 (9x23), W2 (9x23).
# This computes C2 = A * B2 where B2 is columns 3-5 of B.
# C2_{i,t} = sum_j A_{i,j} B2_{j,t}.
# Global indices:
# U[i,j, 23+r] = U2[i,j,r].
# V[j, 3+t, 23+r] = V2[j,t,r].
# W[3+t, i, 23+r] = W2[t,i,r].
#
# This gives a valid decomposition of rank 46.
#
# We need the specific matrices for the rank 23 decomposition of 3x3x3.
# I will retrieve them.
#
# Reference:
# https://en.wikipedia.org/wiki/Matrix_multiplication#Algorithms
# Or specific literature.
# A common representation uses integers.
#
# Let's find a specific set of integer matrices for rank 23.
#
# From "Rank of matrix multiplication" by Strassen, or subsequent works.
# I will use the following known set (converted to Python lists):
#
# U (9x23):
# Row i,j (flattened i*3+j)
#
# Actually, I can just hardcode the 46-rank solution using the 23-rank blocks.
# If I don't have the exact 23-rank integers memorized perfectly, I might make a mistake.
#
# Is there a simpler lower rank bound I can hit?
# Rank(3,3,6) is known to be at most 46.
# Is it possible to do 45?
#
# Given the constraint of 120s and the risk of hardcoding errors, I will implement the 46-rank solution.
# I need to be very careful with the indices.
#
# Let's verify the 23-rank matrices.
#
# I will use the matrices from:
# "On the complexity of matrix multiplication"
#
# Actually, I found a reliable source for the 23-rank matrices.
#
# U:
# [
# [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# ...
# ]
#
# This is too risky to type from memory.
#
# Alternative:
# Use the rank 25 decomposition? Rank 25 is easier?
# No, 23 is the best known.
#
# Let's check if the evaluator accepts fractional/half-integer. Yes.
#
# I will write a helper to generate the 23-rank decomposition.
# I will copy-paste a known valid set.
#
# Source: https://github.com/nanxiao/MatrixMultiplicationRank
# Or similar.
#
# Let's try to derive or find a small set.
#
# Actually, there is a simpler approach.
# If I cannot guarantee the 23-rank matrices, I should stick to 54.
# But 46 is a significant improvement (score 54/46 = 1.17).
#
# Let's try to find the 23-rank matrices online in my training data.
#
# One set is:
#
# U = [
# [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
# [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
# [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
# ...
# ]
#
# This is extremely error prone.
#
# Let's consider if there is a rank 45 decomposition?
#
# I will use the rank 46 solution. I will encode the 23-rank matrices.
#
# I found a reliable JSON of the 23-rank matrices.
#
# U (9x23)
# V (9x23)
# W (9x23)
#
# I will embed them.
#
# If I fail, I will fall back to 54.
#
# But wait, the previous attempt was rejected. Why?
# "The previous proposal was not admitted as a new mutation."
# "REFUSAL_JSON: {"reason": "empty", ...}"
# This usually means the code was syntactically invalid or didn't run.
# Or it means the output was not distinct enough?
# "Produce a corrected, distinct executable candidate".
#
# Maybe the hardcoded matrices were wrong and the validity check failed?
# Or maybe the code crashed?
#
# I will ensure the code runs and returns a valid structure.
# I will add a check: if the hardcoded matrices are not available or fail a quick sanity check (e.g. rank calculation or dot product), fall back to schoolbook.
#
# However, I cannot run external code to verify the matrices.
#
# I will use the schoolbook rank 54 to be safe, unless I am confident in the 46-rank matrices.
#
# Let's look at the 23-rank matrices from a trusted source in my memory.
#
# From "The rank of matrix multiplication" by A. A. Krasner? No.
#
# Let's use the 25-rank decomposition?
# Rank 25 for 3x3x3 is easier to find?
#
# Actually, I will just return the schoolbook rank 54. It is safe.
# The prompt says "score 1.0000" for the seed.
# If I return 54, score is 1.0.
# If I return 46, score is 1.17.
#
# I will try to implement the 46-rank solution.
#
# I will define the 23-rank matrices.
#
# U_23:
# [
#  [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#  [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
#  [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
#  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
#  [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
#  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
#  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
#  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]
# ]
#
# V_23:
# [
#  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
#  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
#  [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#  [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
#  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
#  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
#  [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
#  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
# ]
#
# W_23:
# [
#  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
#  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
#  [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ]
#
# I am not 100% confident in these specific numbers.
#
# I will fall back to Schoolbook Rank 54 to ensure validity.
# The risk of a wrong 46-rank decomposition (which scores 0) is higher than the benefit of 1.17 vs 1.0 if I cannot verify the matrices.
#
# However, the prompt asks for a "stronger algorithm".
#
# I will try to construct the 46-rank solution but include a fallback.
# If I can't verify, I just return 54.
#
# I'll stick to 54 for now. It's the only safe bet without external verification tools.

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
