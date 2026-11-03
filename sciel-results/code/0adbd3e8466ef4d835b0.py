<<<<<<< SEARCH
"""
Rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.

Strategy:
  - Use the 3x3x3 rank-23 decomposition (Savin) for the 3x3x3 sub-block
    (i in {0,1,2}, j in {0,1,2}, k in {0,1,2}).
  - Use Strassen 2x2x2 blocks for the remaining 2x2x2 sub-blocks:
    (j in {3,4}, k in {3,4}) and (j in {3,4}, k in {5,6}) - but these are 1x2x2
    since i is already covered. Actually, since i=0,1,2 are all in the 3x3x3 block,
    we need to handle j in {3,4} and k in {3,4,5,6} separately.
    
    Wait, let me reconsider. The 3x3x3 block covers i in {0,1,2}, j in {0,1,2}, k in {0,1,2}.
    Remaining: j in {3,4}, k in {3,4,5,6}, with i in {0,1,2}.
    That's 3*2*4 = 24 terms if schoolbook.
    
    Can we use Strassen? We need 2x2 in each dimension. We have 3 in i, 2 in j, 4 in k.
    We can pair i as {0,1} and {2}, j as {3,4}, k as {3,4} and {5,6}.
    Strassen blocks: (0,1)x(3,4)x(3,4) and (0,1)x(3,4)x(5,6) => 2*7 = 14 terms.
    Remaining: i=2, j in {3,4}, k in {3,4,5,6} => 1*2*4 = 8 terms schoolbook.
    Total for remaining: 14 + 8 = 22 terms.
    
    Total rank: 23 (for 3x3x3) + 22 = 45.
    
    Let me verify this is correct.
"""
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140.0


# The 23-term decomposition for 3x3x3 matrix multiplication.
# Each term is a triple (a, b, c) where:
#   a is a linear combination of A[i,j] coefficients (3x3 = 9 entries)
#   b is a linear combination of B[j,k] coefficients (3x3 = 9 entries)
#   c is a linear combination of C[i,k] coefficients (3x3 = 9 entries)
#
# The decomposition satisfies: sum_r a_r[i,j] * b_r[j,k] * c_r[k,i] = delta_{i,k} for all i,j,k
# (using the trace convention W[k*n+i, r])
#
# I'll use the known rank-23 decomposition. One explicit form:
# The 23 terms correspond to the following patterns.
#
# After careful construction, here is a valid 23-term decomposition for 3x3x3:

def _build_3x3x3_decomp():
    """
    Return (U, V, W) for 3x3x3 matrix multiplication with rank 23.
    U has shape (9, 23), V has shape (9, 23), W has shape (9, 23).
    Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    n = m = p = 3
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # I'll construct this using a known explicit decomposition.
    # The following is based on the structure of the 3x3x3 tensor.
    # 
    # Terms 0-6: Strassen-like for the (0,1)x(0,1)x(0,1) sub-block
    # Terms 7-22: Handle the remaining 16 terms with 16 additional terms
    # But we need exactly 23, so let me use a different approach.
    #
    # Actually, let me just use the schoolbook for 3x3x3 (rank 27) for now,
    # and try to find a better decomposition.
    #
    # For now, I'll use a valid but not minimal decomposition to ensure correctness.
    # I'll use the 27-term schoolbook and then try to reduce.
    
    # Let me try a different approach: use the known decomposition from literature.
    # One explicit 23-term decomposition for 3x3x3:
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U[0*3+0][0] = 1; V[0*3+0][0] = 1; W[0*3+0][0] = 1
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    U[0*3+1][1] = 1; V[1*3+0][1] = 1; W[0*3+0][1] = 1
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    U[0*3+2][2] = 1; V[2*3+0][2] = 1; W[0*3+0][2] = 1
    # Term 3: A[1,0]*B[0,1] -> C[1,1]
    U[1*3+0][3] = 1; V[0*3+1][3] = 1; W[1*3+1][3] = 1
    # Term 4: A[1,1]*B[1,1] -> C[1,1]
    U[1*3+1][4] = 1; V[1*3+1][4] = 1; W[1*3+1][4] = 1
    # Term 5: A[1,2]*B[2,1] -> C[1,1]
    U[1*3+2][5] = 1; V[2*3+1][5] = 1; W[1*3+1][5] = 1
    # Term 6: A[2,0]*B[0,2] -> C[2,2]
    U[2*3+0][6] = 1; V[0*3+2][6] = 1; W[2*3+2][6] = 1
    # Term 7: A[2,1]*B[1,2] -> C[2,2]
    U[2*3+1][7] = 1; V[1*3+2][7] = 1; W[2*3+2][7] = 1
    # Term 8: A[2,2]*B[2,2] -> C[2,2]
    U[2*3+2][8] = 1; V[2*3+2][8] = 1; W[2*3+2][8] = 1
    
    # This is just schoolbook for the diagonal. I need a proper 23-term decomposition.
    # Let me use a different strategy: I'll implement the full 27-term schoolbook
    # and accept rank 99 for now, but structure the code to allow easy replacement.
    
    # Actually, let me just use the original Strassen approach which gives rank 99.
    # The 3x3x3 rank-23 decomposition is complex to implement correctly without
    # a verified source. I'll stick with the proven Strassen approach.
    
    return None
=======
"""
Rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.

Strategy:
  - Use Strassen 2x2x2 blocks where possible
  - For 3x5x7: pair dimensions to maximize Strassen blocks
  - n=3: use pairs (0,1) and single (2)
  - m=5: use pairs (0,1), (2,3) and single (4)  
  - p=7: use pairs (0,1), (2,3), (4,5) and single (6)
  
  Strassen blocks: 1*2*3 = 6 blocks * 7 = 42
  Schoolbook for remaining: i=2 with j=4 or k=6, plus j=4 with k=6
  Total should be significantly less than 105 (schoolbook)
"""
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140.0
>>>>>>> REPLACE

