# For 3x3 * 3x8:
# C[i,k] = sum_j A[i,j] * B[j,k]
# 
# Split j into {0,1} and {2}:
# C[i,k] = sum_{j=0,1} A[i,j] * B[j,k] + A[i,2] * B[2,k]
#
# For the first term, use Strassen on 2x2 * 2x2 for each pair of k:
# This gives 4 Strassen blocks (for k pairs (0,1), (2,3), (4,5), (6,7))
# Each Strassen block computes C[0:2, k:k+2] from A[0:2, 0:2] * B[0:2, k:k+2]
# That's 4 * 7 = 28 terms
#
# For A[i,2] * B[2,k]: 3*8 = 24 terms
# Total: 28 + 24 = 52
