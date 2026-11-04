<<<<<<< SEARCH
"""Block-Strassen decomposition for 6x6 matrix multiplication.

Strategy:
  - Partition the 6x6 matrices into 2x2 blocks of 3x3 sub-matrices.
  - Apply Strassen's 7-multiplication scheme at the block level.
  - Each of the 7 intermediate products is a 3x3 x 3x3 multiplication.
  - Each 3x3 multiplication uses the schoolbook rank-27 decomposition.
  - Total rank = 7 * 27 = 189.

The evaluator convention: W[k*n+i, r] for the output factor.
"""
import time

DIMENSIONS = [6, 6, 6]
=======
"""Block-Strassen decomposition for 6x6 matrix multiplication.

Strategy:
  - Partition the 6x6 matrices into a 3x3 grid of 2x2 sub-matrices.
  - Apply Strassen's 7-multiplication scheme at each 2x2 block position.
  - Total rank = 7 * 9 = 63.

The evaluator convention: W[k*n+i, r] for the output factor.
"""
import time

DIMENSIONS = [6, 6, 6]
>>>>>>> REPLACE

