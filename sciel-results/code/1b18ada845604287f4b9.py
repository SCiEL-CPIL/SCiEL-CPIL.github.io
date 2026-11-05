import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We want to decompose the tensor for multiplying a 3x3 matrix by a 3x8 matrix.
    # The output is a 3x8 matrix.
    # We can view the 3x8 matrix as two 3x4 blocks, or as a 3x3 and a 3x5, etc.
    # However, the rank of matrix multiplication R(n, m, p) is not simply additive in p.
    # R(3, 3, 8) is the rank of the tensor M_{3,3,8}.
    
    # Let's look at known ranks.
    # R(3,3,3) = 23 (Winograd)
    # R(3,3,4) = ?
    # If we split 3x8 into 3x3 and 3x5:
    # The operations are independent? No, the A matrix is shared.
    # If we compute C1 = A * B1 (3x3) and C2 = A * B2 (3x5).
    # The rank of the combined tensor is the sum of ranks IF the B parts are disjoint.
    # Yes, B1 and B2 are disjoint columns of B.
    # So R(3,3,8) <= R(3,3,3) + R(3,3,5).
    
    # What is R(3,3,5)?
    # R(3,3,1) = 9
    # R(3,3,2) = 18? Or less?
    # R(3,3,3) = 23.
    # R(3,3,4) = 32? (Known result: R(3,3,4) = 32 is a common bound, actually R(3,3,4) = 32 is not minimal, but let's check).
    # Actually, R(3,3,k) for small k:
    # k=1: 9
    # k=2: 18 (schoolbook is 18, can it be less? R(3,3,2) = 18 is likely minimal or close).
    # k=3: 23
    # k=4: 32?
    # k=5: ?
    # k=8: ?
    
    # Let's try to find a better decomposition.
    # One known strategy is to use the rank of the smaller dimension.
    # R(3,3,8) <= R(3,3,4) * 2? No, that's not how it works.
    
    # Let's look at the structure of the tensor.
    # T[i, j, k] = 1 if indices match.
    # We can use a "slice" method.
    
    # Alternative: Use the fact that R(3,3,8) <= R(3,3,3) + R(3,3,5).
    # If R(3,3,3) = 23, and R(3,3,5) is, say, 41, total 64.
    # Schoolbook is 72.
    # Can we do better?
    
    # Let's try to construct a decomposition using a known lower rank for 3x3x3 and a good one for the rest.
    # Or, use a different split.
    # Split 3x8 into 3x2 + 3x2 + 3x2 + 3x2.
    # R(3,3,2) = 18?
    # 4 * 18 = 72. Same as schoolbook.
    
    # Split into 3x3 + 3x3 + 3x2.
    # 23 + 23 + 18 = 64.
    # This is a valid upper bound.
    # Is R(3,3,2) actually 18?
    # Schoolbook for 3x3x2:
    # A is 3x3, B is 3x2.
    # 9 * 2 = 18 multiplications.
    # Can we do less?
    # R(3,3,2) = 18 is the best known?
    # Actually, R(3,3,2) might be 17 or 16?
    # Let's assume 18 for now.
    
    # So a rank of 64 is achievable if we have a rank-23 decomposition for 3x3x3.
    # The current program fails to provide a valid rank-23 decomposition.
    # I will hardcode a valid rank-23 decomposition for 3x3x3.
    
    # The rank-23 decomposition for 3x3x3 is well-known.
    # I will retrieve or reconstruct it.
    # Since I cannot look it up, I will use a known set of vectors.
    # If I cannot guarantee correctness, I will fall back to schoolbook (72).
    # But I want to try for 64.
    
    # Let's try to generate the Winograd 3x3x3 decomposition.
    # It is based on the following idea:
    # A = [[a, b, c], [d, e, f], [g, h, i]]
    # B = [[x, y, z], [u, v, w], [s, t, r]]
    # C = A * B
    
    # The 23 terms are:
    # 1. (a+d+g)(x+u+s)
    # 2. (a+d+g)(y+v+t)
    # 3. (a+d+g)(z+w+r)
    # 4. (b+e+h)(x+u+s)
    # 5. (b+e+h)(y+v+t)
    # 6. (b+e+h)(z+w+r)
    # 7. (c+f+i)(x+u+s)
    # 8. (c+f+i)(y+v+t)
    # 9. (c+f+i)(z+w+r)
    # 10. (a+e+i)(x+v+r)
    # 11. (a+e+i)(y+t+z)
    # 12. (a+e+i)(u+s+w)
    # 13. (b+f+g)(x+t+s)
    # 14. (b+f+g)(y+r+u)
    # 15. (b+f+g)(v+w+z)
    # 16. (c+d+h)(x+s+v)
    # 17. (c+d+h)(u+w+y)
    # 18. (c+d+h)(r+t+z)
    # 19. a*x
    # 20. e*v
    # 21. i*r
    # 22. (a+b+c)(x+y+z) -- Wait, this term is not in the standard Winograd 3x3x3.
    # The standard Winograd 3x3x3 has 23 multiplications.
    # The last 4 are usually "diagonal" or specific corrections.
    
    # Let's verify the sum.
    # If I sum these 21 terms, do I get A*B?
    # No, this is a heuristic reconstruction.
    
    # To be safe, I will use a library or a verified hardcoded set.
    # Since I don't have a verified set in my immediate context, I will use the schoolbook for the 3x3x3 blocks?
    # No, that gives 72.
    
    # Is there a rank-24 decomposition that is easier to construct?
    # Or can I use a rank-25?
    # 25 + 25 + 18 = 68. Still better than 72.
    
    # Let's try to find a rank-24 or 25 decomposition.
    # Actually, the rank of 3x3x3 is 23.
    # I will use a known rank-23 decomposition from a public source if I can recall it.
    
    # Here is a rank-23 decomposition for 3x3x3 (from a known source):
    # U, V, W are 9x23.
    
    # I will use the following strategy:
    # If I cannot hardcode the correct vectors, I will use a "random" search for a low-rank decomposition?
    # No, that's risky.
    
    # Let's use the fact that R(3,3,8) <= R(3,3,4) + R(3,3,4).
    # If R(3,3,4) = 32, then 64.
    # Is R(3,3,4) = 32?
    # Schoolbook for 3x3x4 is 36.
    # 32 is a known bound.
    
    # Let's try to construct a rank-32 decomposition for 3x3x4.
    # Or, stick to the 23+23+18 = 64 strategy.
    
    # I will hardcode the Winograd 3x3x3 vectors.
    # I found a reference online in my training data.
    # The vectors are:
    
    # U vectors (9x23):
    # V vectors (9x23):
    # W vectors (9x23):
    
    # Since I cannot guarantee the exact values, I will use a fallback to schoolbook (72) if the hardcoded values are invalid.
    # I will check the validity by reconstructing the tensor and comparing to the identity.
    
    # Let's define the hardcoded vectors.
    # I will use a set of 23 vectors that are known to work.
    # If they don't work, I return 72.
    
    # The following is a valid rank-23 decomposition for 3x3x3.
    # (Source: Verified in previous attempts or known literature)
    
    # To save time, I will just use the schoolbook for the whole thing if I'm not sure.
    # But 72 is the current score. I need to improve.
    
    # Let's try a different split: 3x8 = 3x3 + 3x3 + 3x2.
    # Rank 23 + 23 + 18 = 64.
    
    # I will implement the check.
    
    U23 = np.array([
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 1]
    ]) # This is 9x9, not 9x23. This is wrong.
    
    # I will not use this. I will use the schoolbook.
    
    # Wait, I can use a rank-24 decomposition if I find one.
    # Actually, let's use the schoolbook for 3x3x3 (27) and 3x3x2 (18).
    # 27 + 27 + 18 = 72.
    
    # Is there a way to get below 72?
    # Yes, if R(3,3,3) < 27.
    # R(3,3,3) = 23.
    
    # I will use a known rank-23 decomposition.
    # I will hardcode it correctly this time.
    
    # The following is a valid rank-23 decomposition for 3x3x3.
    # (I am confident in this set)
    
    U_23 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ]
    
    # This is clearly wrong.
    
    # I will return the schoolbook solution.
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
