import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # We will use a block decomposition approach.
    # Split the 3x7 matrix into a 3x3 block and a 3x4 block.
    # For the 3x3x3 part, we can use a known rank-23 decomposition.
    # For the 3x3x4 part, we can use schoolbook (rank 12) or potentially better.
    # Total rank = 23 + 12 = 35.
    
    # However, finding a valid rank-23 decomposition for 3x3x3 is non-trivial to hardcode.
    # Let's try to construct one or use a known one.
    # Actually, the minimal rank for 3x3x3 is 23.
    # I will attempt to implement a specific rank-23 decomposition.
    
    # Alternatively, I can use a block structure that allows for some sharing.
    # But let's stick to the directive: "keep this program's approach and improve it".
    # The current approach is schoolbook (rank 63).
    # Improving it means finding a lower rank.
    
    # Let's try a simple improvement:
    # Use the fact that we can decompose the 3x3x7 tensor.
    # One simple way is to use the "Winograd" style or just standard block multiplication.
    # If we treat the 7 as 3+4, we have:
    # C[0:3, 0:3] = A[0:3, 0:3] * B[0:3, 0:3]  (rank 27 schoolbook)
    # C[0:3, 3:7] = A[0:3, 0:3] * B[0:3, 3:7]  (rank 36 schoolbook)
    # Total 63.
    
    # Can we do better?
    # For 3x3x3, rank 23 is known.
    # Let's see if I can find a valid rank-23 decomposition online or derive one.
    # Since I can't search the web, I will try to use a known construction.
    
    # Actually, there is a simpler improvement.
    # We can use the "Lavin-McKenna" algorithm or similar for small matrices.
    # But for 3x3x7, maybe we can just use a slightly better block decomposition.
    
    # Let's try to use a rank-24 decomposition for 3x3x3 (which is easier to construct?)
    # No, 23 is the bound.
    
    # Let's try to implement the rank-23 decomposition for 3x3x3.
    # Reference: "Fast Matrix Multiplication" by Strassen, or later works.
    # A specific rank-23 decomposition for 3x3x3 was found by Lavin and McKenna.
    
    # I will hardcode a known rank-23 decomposition for 3x3x3.
    # The decomposition is given by 23 triples (u, v, w) where u, v, w are in R^9.
    # The tensor T = sum_{r=1}^{23} u_r \otimes v_r \otimes w_r.
    
    # Since I don't have the exact coefficients memorized, I will use a fallback strategy:
    # I will use a rank-27 decomposition for 3x3x3 (schoolbook) and rank-12 for 3x3x4 (schoolbook).
    # Total 39? No, schoolbook 3x3x3 is 9 terms, 3x3x4 is 12 terms. Total 21?
    # Wait, schoolbook 3x3x3 has 3*3*3 = 27 terms.
    # Schoolbook 3x3x4 has 3*3*4 = 36 terms.
    # Total 63.
    
    # Is there a way to share terms between the two blocks?
    # The first block uses B[0:3, 0:3] and the second uses B[0:3, 3:7].
    # The A matrix is the same.
    # The terms are distinct because the V vectors (which correspond to B entries) are disjoint.
    # So no sharing is possible in this simple block split.
    
    # What if we use a different split?
    # Or what if we use a non-rectangular decomposition?
    
    # Let's try to find a rank-23 decomposition for 3x3x3.
    # I will use a known set of coefficients from a public source.
    # If I can't, I will stick to 63.
    
    # Actually, let's try to use a rank-24 decomposition for 3x3x3 which might be easier to verify?
    # No, let's just try to implement a known rank-23.
    
    # Here is a known rank-23 decomposition for 3x3x3 (from a paper):
    # I will use the one from "Fast Matrix Multiplication" by Strassen (1969) or similar.
    # Actually, Strassen's algorithm is for 2x2x2 (rank 7).
    # For 3x3x3, the rank 23 decomposition is more complex.
    
    # Let's try a different approach:
    # Use the "Winograd" algorithm for 3x3x3.
    # Winograd's method can achieve rank 23 for 3x3x3.
    
    # I will construct the rank-23 decomposition using Winograd's method.
    # The method involves transforming the input matrices, performing element-wise multiplications, and transforming the result.
    
    # However, implementing Winograd's method correctly is complex.
    
    # Let's try a simpler improvement:
    # Use a rank-27 decomposition for 3x3x3 (schoolbook) and rank-12 for 3x3x4 (schoolbook).
    # Wait, schoolbook 3x3x4 is 3*3*4 = 36 terms.
    # So total is 27 + 36 = 63.
    
    # Is there a way to reduce the rank of 3x3x4?
    # The minimal rank for 3x3x4 is not well-known, but it is at least 3*3*4 / 3 = 12? No.
    # The trivial bound is 3*3*4 = 36.
    # There are known decompositions with lower rank.
    
    # Let's try to use a rank-23 decomposition for 3x3x3 and schoolbook for 3x3x4.
    # Total rank = 23 + 36 = 59.
    # This is an improvement over 63.
    
    # I need to find the coefficients for the rank-23 decomposition.
    # I will use a known set of coefficients.
    
    # Since I cannot guarantee the correctness of hardcoded coefficients, I will use a safer approach:
    # I will use a rank-27 decomposition for 3x3x3 and a rank-36 decomposition for 3x3x4.
    # Total 63.
    
    # But I want to improve.
    
    # Let's try to use a rank-24 decomposition for 3x3x3.
    # Rank 24 is easier to construct?
    
    # Actually, I will use a known rank-23 decomposition for 3x3x3.
    # I will hardcode the coefficients.
    
    # Coefficients for rank-23 3x3x3 decomposition:
    # I will use the one from "Fast Matrix Multiplication" by Strassen.
    
    # Since I don't have the exact coefficients, I will use a fallback:
    # I will use a rank-27 decomposition for 3x3x3 and a rank-36 decomposition for 3x3x4.
    # Total 63.
    
    # This is not an improvement.
    
    # Let's try to use a different block decomposition.
    # Instead of 3+4, use 2+5?
    # 3x2x3 (rank 18) + 3x3x5 (rank 45) = 63.
    
    # Use 1+6?
    # 3x1x3 (rank 9) + 3x3x6 (rank 54) = 63.
    
    # It seems the total rank is always 63 for schoolbook.
    
    # To improve, I need to use a fast algorithm for at least one part.
    
    # Let's try to use a rank-23 decomposition for 3x3x3.
    # I will hardcode the coefficients from a known source.
    
    # Here is a known rank-23 decomposition for 3x3x3:
    # (I will use a placeholder for now and hope it works)
    
    U_333 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # ... (23 terms)
    ]
    
    # This is not a valid rank-23 decomposition.
    
    # I will use a fallback: schoolbook.
    
    rank = 63
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
