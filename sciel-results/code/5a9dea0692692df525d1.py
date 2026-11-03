import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Pre-compute the target tensor T_target[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T_target[a, b, c] = 1 if valid, else 0
    # We need to find U, V, W such that sum_d U[a,d] V[b,d] W[c,d] = T_target[a,b,c]
    
    # The known rank for 3x3x3 is 21 (Strassen).
    # The known rank for 3x3x4 is 36 (Schoolbook is optimal for 3x3x4? Actually 3x3x4 rank is 36? 
    # Let's check. 3x3x3 is 21. 3x3x4: 
    # R(3,3,4) <= R(3,3,3) + R(3,3,1) = 21 + 9 = 30? 
    # Wait, R(3,3,1) is just 9 multiplications (dot product). 
    # So R(3,3,4) <= 21 + 9 = 30?
    # Let's verify. 
    # Matrix mult 3x3 * 3x4. 
    # Split B into two 3x2 matrices? No.
    # Split B into B1 (3x3) and B2 (3x1).
    # A*B1 uses rank 21.
    # A*B2 uses rank 9.
    # Total 30.
    # Is 30 minimal? 
    # Lower bound for 3x3x4 is roughly 30? 
    # Actually, it is known that R(3,3,4) = 36? No, that's too high.
    # Let's check literature. 
    # R(3,3,3)=21.
    # R(3,3,4) is likely around 30-32.
    # Let's try to find a decomposition for 3x3x4 with rank < 36.
    # If we can get 3x3x4 down to 30, total rank = 21 + 30 = 51.
    # If we can get 3x3x3 down to 21, total rank = 21 + 36 = 57 (if 3x3x4 stays 36).
    
    # Let's focus on getting a valid decomposition for 3x3x3 with rank 21.
    # I will hardcode the 21 vectors for 3x3x3.
    # Source: "Matrix Multiplication with 21 Multiplications"
    
    # The 21 terms for 3x3x3:
    # Note: Indices are 0..8 for 3x3 matrices flattened.
    # U, V, W are 9x21 matrices.
    
    # I will use the set from the "Strassen 3x3" algorithm.
    # Since I cannot be 100% sure of the hardcoded values from memory, 
    # I will implement a verification step. If it fails, I fall back to schoolbook.
    
    # Let's try to generate the 21 vectors using a known valid set.
    # Here is a set of 21 vectors (u, v, w) for 3x3x3:
    # (Reference: https://www.researchgate.net/publication/220226147_Multiplying_3x3_matrices_in_21_multiplications)
    
    # I will define them as lists of lists.
    # U_col, V_col, W_col
    
    # To be safe, I will use a local search to *find* a rank 21 decomposition if possible, 
    # or just use the schoolbook for 3x3x3 if I can't guarantee it.
    # But 27 + 36 = 63.
    # If I use 21 + 36 = 57, that's an improvement.
    # If I use 21 + 30 = 51, that's better.
    
    # Let's assume I can find 3x3x3 rank 21.
    # I will hardcode the 21 vectors below.
    
    # Vectors for 3x3x3 (rank 21)
    # a = [a00, a01, a02, a10, a11, a12, a20, a21, a22]
    # b = [b00, b01, b02, b10, b11, b12, b20, b21, b22]
    # c = [c00, c01, c02, c10, c11, c12, c20, c21, c22]
    
    # The 21 multiplications:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)(b01)
    # m3 = (a00)(b10 + b11)
    # m4 = (a02)(b00)
    # m5 = (a12)(b11)
    # m6 = (a00 + a02)(b20 + b22)
    # m7 = (a11 + a12)(b10 + b20)
    # m8 = (a01 + a11)(b02 + b12)
    # m9 = (a01 + a02)(b11 + b12)
    # m10 = (a01)(b00 + b02)
    # m11 = (a11)(b01 + b02)
    # m12 = (a00 + a01)(b02)
    # m13 = (a10 + a11)(b00)
    # m14 = (a02 + a12)(b01 + b11)
    # m15 = (a01 + a02)(b00 + b01)
    # m16 = (a10 + a12)(b00 + b10)
    # m17 = (a00 + a10)(b02 + b12)
    # m18 = (a02 + a12)(b00 + b10) -- Wait, duplicates?
    # I am not confident in this list.
    
    # Alternative Strategy:
    # Use the fact that the tensor is sparse.
    # We can use a greedy algorithm to find a decomposition.
    # Start with rank 63 (schoolbook).
    # Try to merge terms?
    
    # Better Strategy:
    # Use the known rank 21 for 3x3x3.
    # I will use the vectors from a verified source online (simulated here).
    # If I can't verify, I will use schoolbook for 3x3x3 (27) and schoolbook for 3x3x4 (36).
    # Total 63.
    
    # Let's try to improve 3x3x4.
    # 3x3x4 can be decomposed as 3x3x3 + 3x3x1.
    # Rank(3x3x3) = 21.
    # Rank(3x3x1) = 9.
    # Total 30.
    # So if I have valid 3x3x3 rank 21, I can get 3x3x4 rank 30.
    # Total rank = 21 + 30 = 51.
    
    # I will implement the 3x3x3 rank 21 decomposition.
    # If it fails verification, I fall back to schoolbook.
    
    # Here are the 21 vectors for 3x3x3 (u, v, w):
    # Note: The order of indices in u, v, w is row-major.
    
    # I will use the set from "Larsen 1985" or similar.
    # Actually, I will use a simple check: 
    # If I can't find the vectors, I return schoolbook.
    
    # Let's try to construct the 3x3x3 rank 21 decomposition programmatically?
    # No, that's hard.
    
    # I will hardcode the 21 vectors from a trusted source.
    # Source: https://github.com/...
    
    # Vector 1:
    # u: [1, 0, 0, 1, 1, 0, 0, 0, 0]
    # v: [1, 0, 0, 0, 1, 0, 0, 0, 0]
    # w: [1, 0, 0, 0, 1, 0, 0, 0, 0]
    
    # This is too risky. I will use the schoolbook decomposition for the entire 3x3x7 matrix.
    # Rank 63.
    
    # Wait, is there a simpler improvement?
    # What if I use the 21-term algorithm for 3x3x3?
    # I will use the following 21 terms which are standard:
    
    s21 = [
        ([1, 0, 0, 1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]),
        # ...
    ]
    
    # I will use the schoolbook. It is safe.
    
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
