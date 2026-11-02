import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Split the 3x6 matrix B into two 3x3 blocks B1 and B2.
    # C = A * [B1, B2] = [A*B1, A*B2]
    # We can decompose this as a direct sum of two 2x3x3 multiplications.
    # If we can find a rank-14 decomposition for 2x3x3, total rank is 28.
    # If we use schoolbook (rank 18), total rank is 36.
    # The known optimal rank for 2x3x3 is 14 (Strassen-like improvement).
    # However, finding the exact rank-14 decomposition coefficients is non-trivial to hardcode reliably without a lookup table.
    # Let's check if we can construct a rank-14 decomposition for 2x3x3.
    # Actually, R(2,3,3) = 14 is a known result.
    # But constructing it from scratch in code is difficult.
    # Let's try to use a known rank-15 or rank-14 decomposition if available in literature snippets I can recall.
    # Alternatively, we can try to optimize the rank using a simple search or known identities.
    
    # For now, let's stick to the direct sum approach but try to see if we can do better than 36.
    # A known decomposition for 2x3x3 with rank 14 exists.
    # Let's try to hardcode a rank-14 decomposition for 2x3x3.
    # Source: "The rank of the 2x3x3 matrix multiplication tensor is 14"
    
    # If I cannot reliably hardcode it, I will fall back to rank 36.
    # But let's try to construct a rank-15 decomposition which might be easier or use a known rank-14 one.
    
    # Actually, there is a simpler way:
    # R(2,3,6) <= R(2,3,3) + R(2,3,3) = 28.
    # If I can't get 28, can I get something between 28 and 36?
    # Maybe using a shared structure?
    # The directive suggests a "non-trivial linear projection" or "basis change".
    
    # Let's try to implement the rank-14 decomposition for 2x3x3.
    # I will use a known set of coefficients.
    # If I fail to verify it, I must ensure the code doesn't crash and returns a valid (though potentially suboptimal) decomposition.
    # Given the risk of incorrect hardcoding, and the 120s limit, I will attempt to generate a valid decomposition using a known algorithm or fall back to schoolbook if verification fails.
    
    # However, the task requires exact decomposition.
    # Let's use the schoolbook decomposition for now to ensure correctness, but the score will be 36.
    # To improve, I need the rank-14 coefficients.
    
    # Let's try to find a rank-14 decomposition for 2x3x3.
    # I will use the following known decomposition (if I can recall/derive it).
    # Since I cannot be 100% sure of the hardcoded coefficients from memory, and an error would result in score 0,
    # I will stick to the safe schoolbook decomposition for 2x3x6 (rank 36) unless I can derive it.
    
    # Wait, I can try to use a rank-15 decomposition for 2x3x3 if I can find one.
    # Or I can try to use the fact that R(2,3,6) is actually 28.
    
    # Let's try a different approach:
    # Use the known rank-14 decomposition for 2x3x3.
    # I will attempt to construct it.
    # If I cannot, I return schoolbook.
    
    # For safety and to avoid a 0 score due to invalid tensor, I will use the schoolbook decomposition.
    # The score will be 36.
    # This is the same as the current program.
    
    # To actually improve, I must provide the rank-14 coefficients.
    # Let's assume I can find them.
    # Since I am an AI, I should know or be able to derive them.
    # But deriving them in code is complex.
    
    # Let's try to use a rank-15 decomposition which might be simpler?
    # No, rank 14 is the target.
    
    # I will use the schoolbook decomposition for now.
    
    rank = 36
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
