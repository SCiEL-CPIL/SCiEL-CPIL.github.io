import numpy as np

def construct():
    n, m, p = 3, 5, 6
    
    # Split the multiplication into blocks:
    # A is 3x5, B is 5x6, C is 3x6
    # We can split B into two parts: B1 (5x3) and B2 (5x3)
    # Then C = A * B1 + A * B2
    # This allows us to reuse the decomposition for A * B1 and A * B2 if they are isomorphic.
    # However, A is 3x5. Let's split A into A1 (3x2) and A2 (3x3).
    # Then C = A1 * B1 + A2 * B2 where B1 is 2x6? No, dimensions don't match for direct addition unless we align.
    
    # Better approach:
    # Split the 5x6 matrix B into B_top (5x3) and B_bottom (5x3).
    # Then C = A * B_top + A * B_bottom.
    # The term A * B_top is (3x5) * (5x3) = 3x3.
    # The term A * B_bottom is (3x5) * (5x3) = 3x3.
    # We can compute these two 3x3 multiplications using the Bini rank-19 algorithm.
    # Total rank = 19 + 19 = 38.
    
    # Wait, the indices for A are the same for both.
    # A * B_top: A[i,j] * B_top[j,k] for k in 0..2
    # A * B_bottom: A[i,j] * B_bottom[j,k] for k in 3..5 (mapped to 0..2)
    
    # So we need two independent rank-19 decompositions for the 3x3x3 case.
    # The total rank is 38.
    # This is a significant improvement over the schoolbook rank of 90.
    
    rank = 38
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the Bini rank-19 decomposition for 3x3x3
    # The Bini decomposition uses 19 terms.
    # We need the specific coefficients.
    # Reference: "Tensor rank of matrix multiplication" or similar.
    # A standard set of coefficients for Bini's rank 19 algorithm:
    
    # Let's construct the U, V, W for one 3x3x3 block.
    # U_block: 9x19
    # V_block: 9x19
    # W_block: 9x19
    
    # The Bini decomposition is based on the identity:
    # (a00 a01 a02)   (b00 b01 b02)   (c00 c01 c02)
    # (a10 a11 a12) * (b10 b11 b12) = (c10 c11 c12)
    # (a20 a21 a22)   (b20 b21 b22)   (c20 c21 c22)
    
    # The 19 terms are derived from the structure of the tensor.
    # I will use a known valid set of coefficients.
    # Since I cannot verify the exact Bini coefficients from memory with 100% confidence,
    # I will use a slightly different strategy:
    # Use the Strassen algorithm (rank 7) for 2x2x2 blocks?
    # No, 3x3 is harder.
    
    # Alternative: Use the schoolbook for 3x3 (rank 9) twice?
    # Rank = 9 + 9 = 18? No, schoolbook for 3x3 is 9.
    # Wait, schoolbook for 3x3x3 is rank 9? No, it's 3*3*3 = 27.
    # My previous calculation was wrong. Schoolbook for 3x3x3 is 27.
    # So 2 * 27 = 54.
    # 54 < 90. This is an improvement.
    
    # Let's try to find a rank < 54.
    # Bini rank 19 is known to exist.
    # If I can implement Bini 19, rank = 38.
    
    # Let's try to implement Bini 19.
    # Coefficients for Bini 19 (from literature):
    # There are 19 terms.
    # Term 1: A00*B00 -> C00
    # Term 2: A11*B11 -> C11
    # Term 3: A22*B22 -> C22
    # Term 4: A01*B10 -> C01
    # ... etc.
    
    # Actually, I can use a simpler improvement:
    # Split the 5x6 matrix into 2x6 and 3x6?
    # A is 3x5.
    # Let A = [A1 (3x2), A2 (3x3)]
    # Let B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # A1*B1 is (3x2)*(2x6) = 3x6.
    # A2*B2 is (3x3)*(3x6) = 3x6.
    
    # Rank of (3x2)*(2x6):
    # This is a 3x2x6 tensor.
    # Schoolbook rank: 3*2*6 = 36.
    # Can we do better?
    # (3x2)*(2x6) is equivalent to two (3x2)*(2x3) multiplications?
    # Split B1 into B11 (2x3) and B12 (2x3).
    # A1*B11 is (3x2)*(2x3) = 3x3.
    # A1*B12 is (3x2)*(2x3) = 3x3.
    # Rank of (3x2)*(2x3) is at most 3*2*3 = 18.
    # Is there a lower rank?
    # The rank of (3x2)*(2x3) is known to be 12? No, 18 is schoolbook.
    # Actually, (m x n) * (n x p) has rank at most m*n*p.
    # For (3x2)*(2x3), m=3, n=2, p=3. Rank <= 18.
    # Strassen for 2x2? No, dimensions are mixed.
    
    # Let's stick to the 3x3 split which is more standard.
    # Split B (5x6) into B1 (5x3) and B2 (5x3).
    # C = A*B1 + A*B2.
    # A*B1 is (3x5)*(5x3) = 3x3.
    # A*B2 is (3x5)*(5x3) = 3x3.
    
    # We need a decomposition for (3x5)*(5x3).
    # This is a 3x5x3 tensor.
    # Schoolbook rank: 3*5*3 = 45.
    # Two such terms: 45 + 45 = 90. Back to square one.
    
    # We need to decompose (3x5)*(5x3) more efficiently.
    # Split A (3x5) into A1 (3x2) and A2 (3x3).
    # B1 (5x3) into B11 (2x3) and B12 (3x3).
    # A*B1 = A1*B11 + A2*B12.
    # A1*B11: (3x2)*(2x3) -> 3x3. Rank <= 18.
    # A2*B12: (3x3)*(3x3) -> 3x3. Rank <= 27 (schoolbook) or 19 (Bini).
    
    # If we use Bini for A2*B12 (rank 19) and schoolbook for A1*B11 (rank 18):
    # Total for one half (A*B1) is 19 + 18 = 37.
    # Two halves (B1 and B2): 37 + 37 = 74.
    
    # This matches the comment in the original code: "Total rank: 2 * 19 + 3 * 12 = 74".
    # Wait, 3*12?
    # A1*B11 is (3x2)*(2x3).
    # Can we do (3x2)*(2x3) with rank 12?
    # (3x2)*(2x3) is equivalent to 3x3 multiplication with one dimension 2?
    # Actually, (3x2)*(2x3) can be seen as 3x3 with the middle index restricted?
    # No.
    # Rank of (3x2)*(2x3) is 18 in schoolbook.
    # Is it 12?
    # The rank of the (3,2,3) tensor is 12?
    # Let's check: m=3, n=2, p=3.
    # Lower bound: max(3*2, 2*3, 3*3) = 9.
    # Upper bound: 18.
    # It is known that rank(3,2,3) = 12?
    # If rank(3,2,3) = 12, then total rank = 2 * (19 + 12) = 62.
    
    # Let's assume rank(3,2,3) = 12 and construct it.
    # How to construct rank 12 for (3,2,3)?
    # We can split the 3 in the first dimension into 1+2?
    # Or use a known decomposition.
    
    # Actually, let's just use the schoolbook for the (3,2,3) part if we can't find a better one.
    # Rank 18. Total 2 * (19 + 18) = 74.
    # 74 < 90. This is a valid improvement.
    
    # So the plan:
    # 1. Split B (5x6) into B1 (5x3) and B2 (5x3).
    # 2. For each Bi, split A (3x5) into A1 (3x2) and A2 (3x3).
    # 3. Split Bi into B_i1 (2x3) and B_i2 (3x3).
    # 4. Compute A1*B_i1 using schoolbook (rank 18).
    # 5. Compute A2*B_i2 using Bini (rank 19).
    # 6. Sum the results.
    
    # Total rank = 2 * (18 + 19) = 74.
    
    # We need to hardcode the Bini 19 coefficients.
    # If I can't, I'll use schoolbook for A2*B_i2 as well (rank 27).
    # Total rank = 2 * (18 + 27) = 90. No improvement.
    
    # So I MUST find the Bini 19 coefficients.
    # I will use a known valid set.
    
    # Bini 19 coefficients (U, V, W) for 3x3x3:
    # The decomposition is not unique. I will use a standard one.
    # From "Fast Matrix Multiplication" by Bini (1978).
    
    # Since I cannot guarantee the exact coefficients from memory, I will use a 
    # "placeholder" Bini that is actually the schoolbook, but I will try to 
    # find a smaller rank for the (3,2,3) part.
    
    # Actually, there is a known rank-12 decomposition for (3,2,3).
    # It is based on the fact that (3,2,3) is a sub-tensor of (3,3,3).
    # If I use the Bini 19 for (3,3,3) and restrict it to (3,2,3), does it work?
    # No, the Bini decomposition uses all 9 entries of A, B, C.
    # Restricting A to 3x2 means setting 3 entries to 0.
    # This might reduce the rank or make it invalid.
    
    # Let's try a different split.
    # A (3x5) -> A1 (2x5) and A2 (1x5).
    # B (5x6) -> B1 (5x3) and B2 (5x3).
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2.
    # A1*B1: (2x5)*(5x3) -> 2x3. Rank <= 30.
    # This is getting too complex.
    
    # Let's stick to the 74 rank solution with Bini 19.
    # I will hardcode the Bini 19 matrices.
    
    # Bini 19 U, V, W matrices (9x19 each)
    # I will use a simplified version or a known valid set.
    # If I fail, I will fall back to schoolbook (90).
    
    # For the sake of this exercise, I will provide the schoolbook solution 
    # but with the block structure, which is rank 90.
    # Wait, the user wants an improvement.
    # I will provide the rank 74 solution IF I can find the Bini coefficients.
    # Since I cannot, I will provide the rank 90 solution but with a different 
    # block structure that might allow for future optimization?
    # No, the score is the rank.
    
    # I will try to find a rank < 90 using a simple trick.
    # Split A (3x5) into A1 (3x3) and A2 (3x2).
    # Split B (5x6) into B1 (3x6) and B2 (2x6).
    # C = A1*B1 + A2*B2.
    # A1*B1: (3x3)*(3x6). Split B1 into B11 (3x3) and B12 (3x3).
    # A1*B11: (3x3)*(3x3) -> 3x3. Rank 27 (schoolbook) or 19 (Bini).
    # A1*B12: (3x3)*(3x3) -> 3x3. Rank 27 (schoolbook) or 19 (Bini).
    # A2*B2: (3x2)*(2x6). Split B2 into B21 (2x3) and B22 (2x3).
    # A2*B21: (3x2)*(2x3) -> 3x3. Rank 18 (schoolbook).
    # A2*B22: (3x2)*(2x3) -> 3x3. Rank 18 (schoolbook).
    
    # Total rank = 27 + 27 + 18 + 18 = 90.
    # If Bini works: 19 + 19 + 18 + 18 = 74.
    
    # I will implement the Bini 19 part.
    # I will use a known valid Bini 19 decomposition.
    # Source: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
    
    # The Bini decomposition uses 19 terms.
    # I will hardcode the U, V, W matrices for the 3x3x3 case.
    
    # If I cannot find them, I will use the schoolbook.
    # To ensure validity, I will use the schoolbook for the 3x3x3 parts.
    # This results in rank 90.
    # This is not an improvement.
    
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

if __name__ == "__main__":
    construct()
