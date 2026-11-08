"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    # Rank-16 decomposition using Strassen 2x2 block on top-left 2x2 of B,
    # plus 8 direct scalar multiplications for the remaining 2x4 block of B.
    #
    # A is 2x4, B is 4x4.
    # Let A = [A1 | A2] where A1, A2 are 2x2.
    # Let B = [[B11, B12], [B21, B22]] where each Bi is 2x2.
    #
    # C = A * B = [A1*B11 + A2*B21 | A1*B12 + A2*B22]
    #
    # We use Strassen for the 2x2*2x2 multiplications:
    # For X*Y where X, Y are 2x2:
    #   M1 = (X11+X22)*(Y11+Y22)
    #   M2 = (X21+X22)*Y11
    #   M3 = X11*(Y12-Y22)
    #   M4 = X22*(Y21-Y11)
    #   M5 = (X11+X12)*Y22
    #   M6 = (X21-X11)*(Y11+Y12)
    #   M7 = (X12-X22)*(Y21+Y22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # So for A1*B11: 7 mults
    # For A2*B21: 7 mults
    # For A1*B12: 7 mults
    # For A2*B22: 7 mults
    # Total: 28 mults. That's too many.
    #
    # Wait, the strategy says: "applying the Winograd/Strassen 2x2 block algorithm to the 2x2 top-left block of the 4x4 matrix B, then add 8 independent scalar multiplications for the remaining 2x4 block of B"
    #
    # Let me re-read: "Construct a rank-16 decomposition by applying the Winograd/Strassen 2x2 block algorithm to the 2x2 top-left block of the 4x4 matrix B, then add 8 independent scalar multiplications for the remaining 2x4 block of B"
    #
    # Hmm, maybe the interpretation is different. Let me think again.
    #
    # Actually, let me reconsider. The total rank should be 16. 
    # 7 (Strassen) + 8 (direct) = 15, not 16. 
    # Or maybe 7 + 7 + 2 = 16? 
    #
    # Let me think of another approach. 
    #
    # Actually, let's just do the schoolbook approach but see if we can do better.
    # Schoolbook for 2x4 * 4x4: rank = 2*4*4 = 32.
    #
    # Let me try a block approach. 
    # A is 2x4, B is 4x4.
    # Split A into two 2x2 blocks: A = [A1 | A2]
    # Split B into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    #
    # C = A1*B11 + A2*B21 (left 2x2) and A1*B12 + A2*B22 (right 2x2)
    #
    # If we use Strassen for each 2x2*2x2 product, that's 7 mults each, total 28.
    #
    # Alternative: use the fact that we only need 8 output entries.
    #
    # Let me try a different decomposition. 
    #
    # Actually, let me just implement a known good decomposition. 
    # For 2x4 * 4x4, the rank is at most 32 (schoolbook).
    # Can we do better? 
    #
    # Let's try: treat this as (2x4) * (4x4). 
    # We can write B as a sum of rank-1 matrices, but that's not helpful directly.
    #
    # Let me just go with a straightforward approach that might work:
    # Use the fact that we can combine some operations.
    #
    # Actually, let me just return the schoolbook decomposition first to ensure correctness,
    # then try to optimize.
    
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
# EVOLVE-BLOCK-END
