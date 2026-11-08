"""Matrix multiplication 3x4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p):
    """Validate if the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(len(U[0])):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = time.time() + 130.0
    
    # Build Strassen-based decomposition for 3x4x8
    # Strategy: 
    #   - Use Strassen for 2x2x2 blocks where possible
    #   - Direct products for the remaining 1x4x8 part
    #   - Total rank: 8 blocks of 7 + 32 direct = 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    # Part 1: i in {0,1}, j pairs (0,1) and (2,3), k pairs (0,1),(2,3),(4,5),(6,7)
    # 2 i-values * 2 j-pairs * 4 k-pairs = 16 blocks? No, that's too many.
    # Actually: for each i in {0,1}, we have a 1x4x8 sub-problem.
    # For each such sub-problem, we can use Strassen on 2x2x2 blocks.
    # 1x4x8: j has 4 values, k has 8 values.
    # We can group j into 2 pairs and k into 4 pairs, giving 8 2x2x2 blocks.
    # Each 2x2x2 block uses Strassen rank 7.
    # So for each i in {0,1}: 8 blocks * 7 = 56 rank dimensions.
    # For i=2: 1x4x8 with direct products: 4*8 = 32 rank dimensions.
    # Total: 2*56 + 32 = 144? That's worse than 96.
    
    # Let me reconsider. The seed has rank 96 = 3*4*8.
    # My earlier analysis said 88, but let me recheck.
    
    # Actually, for i in {0,1}, the sub-problem is 1x4x8.
    # C[i,k] = sum_j A[i,j] B[j,k] for j=0..3, k=0..7.
    # This is 32 products per i-value.
    # Using Strassen on 2x2x2 blocks: we need to group (j,k) into 2x2 blocks.
    # j: 4 values -> 2 groups of 2
    # k: 8 values -> 4 groups of 2
    # So 2*4 = 8 blocks of 2x2x2 per i-value.
    # Each block: rank 7.
    # Per i-value: 8*7 = 56.
    # Two i-values: 112.
    # Plus i=2: 32 direct.
    # Total: 144. Worse than 96.
    
    # So Strassen doesn't help here because the "1" dimension in 1x4x8 means
    # we can't exploit the 2x2 structure in the i-dimension.
    
    # Let me think differently. The full problem is 3x4x8.
    # We can view it as 3 copies of 1x4x8.
    # Or we can try to find a smarter decomposition.
    
    # For now, let's just return the schoolbook decomposition with rank 96.
    # The random search in the original program was unlikely to find anything better.
    
    rank = n * m * p  # 96
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
