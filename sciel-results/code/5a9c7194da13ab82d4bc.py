"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose 2x3 * 3x9 into three parallel 2x3 * 3x3 multiplications.
    # The 3x9 matrix B is viewed as three blocks B_1, B_2, B_3 each of size 3x3.
    # A = [a_ij] (2x3). C = A*B.
    # C[i, k] = sum_j A[i,j] B[j, k].
    # Since B is 3x9, let's index B[j, col] where col in 0..8.
    # Let col = 3*b + k where b in 0..2, k in 0..2.
    # Then C[i, 3*b+k] = sum_j A[i,j] B[j, 3*b+k].
    # This is equivalent to computing 3 separate matrix products:
    # C_b = A * B_b for b=0,1,2, where B_b is 3x3.
    # So the problem reduces to 3 instances of 2x3 * 3x3 multiplication.
    
    # Tensor rank of 2x3 * 3x3 multiplication:
    # Standard schoolbook rank is 2*3*3 = 18.
    # Is there a lower rank?
    # For 2x2 * 2x2, rank is 7.
    # For 2x3 * 3x3, the rank is known to be 18? Or less?
    # Let's check known results.
    # R(2,3,3) = 18 is the trivial bound.
    # Strassen's algorithm for 2x2 gives 7.
    # For 2x3 * 3x3, we can use a block approach or specific decompositions.
    # Actually, 2x3 * 3x3 can be decomposed into 18 rank-1 tensors?
    # Let's try to construct a rank-18 decomposition for 2x3 * 3x3 first.
    # If R(2,3,3) = 18, then total rank = 3 * 18 = 54.
    # Current rank = 2*3*9 = 54.
    # So this strategy yields the same rank.
    
    # Wait, the directive says: "leveraging the fact that the tensor rank of 2 x 3 x 3 is 6".
    # Is R(2,3,3) = 6?
    # Let's verify.
    # Matrix multiplication tensor for m x n * n x p has rank at least max(mn, np, pm).
    # For 2x3 * 3x3: max(6, 9, 6) = 9. So rank >= 9.
    # The directive claims rank 6. This is likely incorrect for exact tensor decomposition.
    # However, I must follow the directive's strategy if it implies a specific construction.
    # But if the premise is wrong, the construction might fail or be invalid.
    # Let's re-read: "leveraging the fact that the tensor rank of 2 x 3 x 3 is 6".
    # This is mathematically false for standard matrix multiplication over reals/integers.
    # R(2,3,3) is at least 9.
    # Perhaps the directive refers to a specific *approximate* rank or a different context?
    # Or maybe it's a trick?
    # If I construct a rank-6 decomposition for 2x3*3x3, it won't be exact.
    # The task requires "exact tensor decomposition".
    
    # Let's look at the "Kronecker product structure" part.
    # If I use the standard schoolbook for 2x3*3x3, rank is 18.
    # Total rank 54.
    
    # Is there a better decomposition for 2x3 * 3x9?
    # Maybe using the structure of 9 = 3*3 differently?
    # Or maybe R(2,3,9) is less than 54?
    # Lower bounds: max(2*3, 3*9, 9*2) = max(6, 27, 18) = 27.
    # So rank >= 27.
    # Current rank 54.
    
    # Can we do better than 54?
    # Let's try to find a decomposition with rank < 54.
    # If R(2,3,3) was 9 (lower bound), then 3*9 = 27.
    # Is R(2,3,3) = 9?
    # If R(2,3,3) = 9, then we can achieve rank 27.
    # How to decompose 2x3 * 3x3 with rank 9?
    # This would require a very efficient algorithm.
    # Standard algorithms don't give 9.
    # However, if the directive insists on "rank 6", it's likely a hallucination or error in the prompt's inspiration.
    # I should not blindly follow a mathematically impossible premise.
    # Instead, I should look for the *best* valid decomposition within the "Kronecker/block" family.
    
    # Let's assume the directive meant to suggest looking for a better block decomposition.
    # If I can't find a better one, I stick with 54.
    # But maybe there is a decomposition of 2x3 * 3x9 with rank < 54?
    
    # Let's try to construct a rank-27 decomposition if possible?
    # If R(2,3,3) = 9, then 3 blocks * 9 = 27.
    # Is R(2,3,3) = 9?
    # A 2x3 matrix A and 3x3 matrix B.
    # C = AB.
    # C is 2x3.
    # There are 6 outputs.
    # Rank 9 means 9 rank-1 terms.
    # This is a known open problem or known result?
    # For 2x2 * 2x2, R=7.
    # For 2x3 * 3x2, R= ?
    # For 2x3 * 3x3, it is known that R >= 9.
    # Whether R=9 is achievable is not trivial.
    # Most sources cite R(2,3,3) >= 9 and <= 18.
    # If R(2,3,3) > 9, then 3*R(2,3,3) > 27.
    
    # Given the uncertainty and the explicit (likely incorrect) claim of rank 6 in the directive,
    # and the fact that the current score is 1.0 (rank 54),
    # I will attempt to implement a decomposition that *might* be better if such exists,
    # or stick to 54 if not.
    
    # However, the directive says "constructing a valid rank-18 decomposition".
    # Rank 18 for the whole 2x3 * 3x9?
    # That would be huge improvement (54 -> 18).
    # But lower bound is 27. So rank 18 is impossible.
    # The directive is internally inconsistent (claims R(2,3,3)=6, leading to R(2,3,9)=18, but LB is 27).
    
    # Therefore, the "inspiration" is flawed.
    # I will revert to the standard schoolbook decomposition (rank 54) as it is known to be valid.
    # Any attempt to force a lower rank based on false premises will likely result in an invalid decomposition (score 0).
    
    # To be safe and adhere to "keep this design", I will output the standard decomposition.
    # I will clean up the code slightly.
    
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
