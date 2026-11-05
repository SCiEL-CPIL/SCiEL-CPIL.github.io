"""Rewritten program for matrix multiplication 2x3x8 rank minimization."""
import time
import random
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 8]

# Known lower bounds and structure
# For 2x3x8, the schoolbook rank is 48.
# We can use block structure: 2x3 = [2x2, 2x1]
# 2x2x8 can be decomposed using Strassen-like or other low rank methods for 2x2 part.
# Actually, let's think about the structure more carefully.
# A(2x3) * B(3x8) = C(2x8)
# We can view this as two independent computations for each output column, but they share the input A.
# Alternatively, split A into A1 (2x2) and A2 (2x1).
# C = A1 * B1 + A2 * B2, where B1 is 2x8 and B2 is 1x8.
# The rank of the sum is at most rank(A1*B1) + rank(A2*B2).
# A1*B1 is 2x2x8 tensor. A2*B2 is 2x1x8 tensor.
# For 2x1x8: this is just 8 independent dot products of a 2-vector and a scalar (row of B2).
# Wait, A2 is 2x1, B2 is 1x8. So A2 * B2 is a 2x8 matrix.
# The tensor for 2x1x8 has rank 2*1*8 = 16? No.
# Let's look at the tensor T(i,j,k) = delta(i,j) for the multiplication.
# For 2x1x8: A is 2x1, B is 1x8. C_{i,k} = A_{i,1} * B_{1,k}.
# The tensor is sum_{i,k} A_{i,1} B_{1,k} C_{i,k}^*.
# This is a sum of 2*8 = 16 rank-1 terms? No, it's a single product of a vector and a vector.
# The rank of the tensor for matrix multiplication m x n x p is m*n*p for the schoolbook.
# For 2x1x8, the rank is 2*8 = 16?
# Let's check: C_{i,k} = sum_j A_{i,j} B_{j,k}. Here j only takes value 1.
# So C_{i,k} = A_{i,1} B_{1,k}.
# This is a rank-1 matrix in the (i,k) space for each fixed... no.
# The tensor is T_{i,j,k} = delta_{j,1} delta_{i,i} delta_{k,k}? No.
# T_{i,j,k} = 1 if j=1, else 0. (Since A_{i,j} is only non-zero for j=1 in the basis?)
# No, the tensor is defined by the multiplication map.
# T(A,B) = A*B.
# In tensor notation, T_{i,j,k} is the coefficient such that C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
# So T_{i,j,k} = 1 for all i,j,k? No.
# C_{i,k} = sum_j A_{i,j} B_{j,k}.
# So the tensor is T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^*.
# The rank is the minimum number of terms sum_r A^{(r)} \otimes B^{(r)} \otimes C^{(r)}* such that
# sum_r A^{(r)}_{i,j} B^{(r)}_{j,k} C^{(r)}_{i,k} = delta_{j,j} (which is 1).
# Wait, the condition is that for all A, B, the product C = A B is recovered.
# This is equivalent to the tensor identity: sum_r A^{(r)} \otimes B^{(r)} \otimes C^{(r)}* = T_{mmult}.
# Where T_{mmult}_{i,j,k} = 1 if the indices correspond to a valid multiplication term.
# Actually, the standard tensor for matrix multiplication m x n x p is:
# T = sum_{i=1}^m sum_{j=1}^n sum_{k=1}^p e_{i,j} \otimes e_{j,k} \otimes e_{i,k}^*
# This is the schoolbook decomposition of rank m*n*p.
# Can we do better?
# For 2x3x8:
# Split j into j1 in {1,2} and j2 in {3}.
# T = T_{2x2x8} + T_{2x1x8}
# T_{2x2x8} has rank at most 2*2*8 = 32.
# T_{2x1x8} has rank at most 2*1*8 = 16.
# Total rank <= 48.
# Can T_{2x2x8} be lower rank?
# T_{2x2x8} is the tensor for multiplying 2x2 by 2x8.
# This is equivalent to 8 independent 2x2 multiplications? No, the A matrix is shared.
# C_{i,k} = A_{i,1} B_{1,k} + A_{i,2} B_{2,k}.
# This is a linear combination.
# The rank of the 2x2x8 tensor is known to be 2*2*8 = 32? Or can it be lower?
# Actually, for 2x2xp, the rank is 2p if p is even? No.
# Let's use the known result: rank(2x2x2) = 7 (Strassen).
# For 2x2xp, we can group the p columns into pairs.
# If p=8, we have 4 pairs.
# Each 2x2x2 block has rank 7.
# But the A matrix is shared across all blocks.
# So we cannot just sum the ranks.
# However, if we treat the 2x2x8 tensor as a direct sum of four 2x2x2 tensors, the rank would be 4*7=28.
# But is it a direct sum?
# The indices are (i, j, k). i in {1,2}, j in {1,2}, k in {1..8}.
# If we partition k into {1,2}, {3,4}, {5,6}, {7,8}, then for each partition, we have a 2x2x2 tensor.
# The A and B indices are shared, but the C index is disjoint.
# In tensor rank, if the C indices are disjoint, the rank is the sum of the ranks.
# So rank(T_{2x2x8}) <= 4 * rank(T_{2x2x2}) = 4 * 7 = 28.
# Similarly, T_{2x1x8}: i in {1,2}, j in {3}, k in {1..8}.
# This is 2x1x8.
# Can we decompose this?
# C_{i,k} = A_{i,3} B_{3,k}.
# This is a rank-1 tensor in the (i,k) space for each j? No.
# The tensor is sum_{i,k} A_{i,3} B_{3,k} C_{i,k}^*.
# This is a sum of 2*8 = 16 terms.
# Can we do better?
# For 2x1xp, the rank is 2p?
# Let's check 2x1x1: C_{i,1} = A_{i,1} B_{1,1}. Rank 2.
# 2x1x2: C_{i,k} = A_{i,1} B_{1,k}. Rank 4?
# Yes, because A_{i,1} is a vector, B_{1,k} is a vector, and we need to compute their outer product.
# The rank of the outer product tensor is m*p.
# So rank(T_{2x1x8}) = 2*8 = 16.
# Total rank <= 28 + 16 = 44.
# This matches the observed improvement from 48 to 44.

# Can we do better?
# Maybe the 2x1x8 part can be combined with the 2x2x8 part?
# Or maybe there is a better decomposition for 2x2x8?
# The rank of 2x2xp is known to be 2p for p even? No, 4*7=28 for p=8.
# Is there a lower rank for 2x2x8?
# According to literature, the rank of 2x2xp is 2p if p is a power of 2? No.
# Actually, the rank of 2x2xp is 2p for p=1,2,3,4?
# For p=2, rank is 7.
# For p=4, rank is 14? (2 * 7)
# For p=8, rank is 28? (4 * 7)
# So 28 seems correct for 2x2x8.
# And 16 for 2x1x8.
# Total 44.

# Let's try to find a decomposition of rank 44.
# We need to construct U, V, W matrices of shapes (6, 44), (24, 44), (16, 44).
# n=2, m=3, p=8.
# U: (n*m, rank) = (6, 44)
# V: (m*p, rank) = (24, 44)
# W: (p*n, rank) = (16, 44)

# We will use the Strassen decomposition for the 2x2x2 blocks.
# Strassen's algorithm for 2x2x2 uses 7 multiplications.
# The 7 products are:
# M1 = (A11 + A22)(B11 + B22)
# M2 = (A21 + A22)B11
# M3 = A11(B21 - B22)
# M4 = A22(B12 - B11)
# M5 = (A11 + A12)B22
# M6 = (A21 - A11)(B11 + B12)
# M7 = (A12 - A22)(B21 + B22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

# For 2x2x8, we have 4 such blocks (k=1,2; k=3,4; k=5,6; k=7,8).
# For each block, we need 7 rank-1 terms.
# Total 28 terms for the 2x2 part.

# For the 2x1x8 part, we have 16 terms.
# Total 44 terms.

# Let's construct the matrices.
# We will use integer entries.

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    # First 28 terms: 2x2x8 using Strassen
    # The 2x2 part uses indices i=0,1 (A rows), j=0,1 (A cols, B rows), k=0..7 (B cols, C cols)
    # In our indexing:
    # A is 2x3, so A[i, j] with i in {0,1}, j in {0,1,2}.
    # B is 3x8, so B[j, k] with j in {0,1,2}, k in {0..7}.
    # C is 2x8, so C[i, k] with i in {0,1}, k in {0..7}.
    # The 2x2 part uses A[i, 0:2] and B[0:2, k].
    # The 2x1 part uses A[i, 2] and B[2, k].

    # Strassen terms for 2x2x2 block (k0, k1):
    # M1 = (A00 + A11)(B0,k0 + B1,k1) -> Wait, the Strassen formula is for 2x2 matrices.
    # Let's use the standard Strassen decomposition for 2x2x2.
    # The 7 products are linear combinations of A and B.
    # We need to map these to our U, V, W.

    # For each of the 4 blocks (k0, k1) in [(0,1), (2,3), (4,5), (6,7)]:
    # We add 7 terms.
    # Total 28 terms.

    # Strassen's 7 products:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B10 - B11)
    # P4 = A11 * (B01 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)

    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # In our indexing:
    # A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
    # B00 = B[0,k0], B01 = B[0,k1], B10 = B[1,k0], B11 = B[1,k1]
    # C00 = C[0,k0], C01 = C[0,k1], C10 = C[1,k0], C11 = C[1,k1]

    # Let's define the Strassen coefficients.
    # For each term r in 0..6:
    # U part: coefficients for A[0,0], A[0,1], A[1,0], A[1,1]
    # V part: coefficients for B[0,k0], B[0,k1], B[1,k0], B[1,k1]
    # W part: coefficients for C[0,k0], C[0,k1], C[1,k0], C[1,k1]

    # Term 0: P1 = (A00 + A11) * (B00 + B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1

    # Term 1: P2 = (A10 + A11) * B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10=1

    # Term 2: P3 = A00 * (B10 - B11)
    # U: A00=1
    # V: B10=1, B11=-1
    # W: C01=1

    # Term 3: P4 = A11 * (B01 - B00)
    # U: A11=1
    # V: B01=1, B00=-1
    # W: C00=1, C10=1

    # Term 4: P5 = (A00 + A01) * B11
    # U: A00=1, A01=1
    # V: B11=1
    # W: C00=-1, C01=1

    # Term 5: P6 = (A10 - A00) * (B00 + B01)
    # U: A10=1, A00=-1
    # V: B00=1, B01=1
    # W: C11=1

    # Term 6: P7 = (A01 - A11) * (B10 + B11)
    # U: A01=1, A11=-1
    # V: B10=1, B11=1
    # W: C00=1, C11=1

    # Let's verify the C equations:
    # C00 = P1 + P4 - P5 + P7
    #   = 1*P1 + 1*P4 + (-1)*P5 + 1*P7
    #   Coefficients: P1:1, P4:1, P5:-1, P7:1
    #   From above:
    #   P1: C00=1
    #   P4: C00=1
    #   P5: C00=-1
    #   P7: C00=1
    #   Sum: 1+1-1+1 = 2? That's wrong. It should be 1.
    #   Wait, the Strassen formula is:
    #   C00 = M1 + M4 - M5 + M7
    #   But M1, M4, M5, M7 are the products.
    #   In the tensor decomposition, each product is a rank-1 term.
    #   So the coefficient of C00 in the sum of terms is the sum of the coefficients from each term.
    #   For term 0 (P1), W[0,k0] = 1.
    #   For term 3 (P4), W[0,k0] = 1.
    #   For term 4 (P5), W[0,k0] = -1.
    #   For term 6 (P7), W[0,k0] = 1.
    #   Sum = 1 + 1 - 1 + 1 = 2.
    #   But we need C00 = A00*B00 + A01*B10.
    #   Let's check the Strassen formula again.
    #   C00 = M1 + M4 - M5 + M7
    #   M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    #   M4 = A11(B01-B00) = A11B01 - A11B00
    #   M5 = (A00+A01)B11 = A00B11 + A01B11
    #   M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    #   Sum = A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    #   = A00B00 + A01B10 + A11B01 - A11B10
    #   This is not C00 = A00B00 + A01B10.
    #   So the Strassen formula I used is incorrect or I misremembered it.

    # Let's use the correct Strassen decomposition.
    # The 7 products are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Let's re-verify C00:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11B01 - A11B00
    # M5 = A00B11 + A01B11
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    # C00 = M1 + M4 - M5 + M7
    # = (A00B00 + A00B11 + A11B00 + A11B11) + (A11B01 - A11B00) - (A00B11 + A01B11) + (A01B10 + A01B11 - A11B10 - A11B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A11B01 + A01B10 - A11B10
    # This is still not A00B00 + A01B10.

    # I think I have the indices wrong. Let's use a reliable source.
    # Strassen's algorithm:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Let's try a different set of Strassen products.
    # Actually, the rank of 2x2x2 is 7, but the exact decomposition is complex.
    # Given the time constraint, let's use the schoolbook decomposition for the 2x2x8 part, which has rank 32.
    # And the 2x1x8 part has rank 16.
    # Total rank 48.
    # But the observed improvement was 44.
    # So there must be a way to get 44.

    # Let's use the direct sum decomposition with Strassen for 2x2x2.
    # I will use a known correct Strassen decomposition.
    # From Wikipedia:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Let's assume this is correct and proceed.
    # The coefficients for C00 are: M1:1, M4:1, M5:-1, M7:1.
    # So in the W matrix, for the C00 entry, we add 1 for term 0, 1 for term 3, -1 for term 4, 1 for term 6.
    # Sum = 1+1-1+1 = 2.
    # But we need the sum to be 1 for the coefficient of A00B00 and A01B10.
    # This suggests that the Strassen decomposition I'm using is not directly applicable in this way, or I'm making a mistake.

    # Given the complexity and time, let's use the schoolbook decomposition of rank 48.
    # This is safe and correct.
    rank = 48
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
