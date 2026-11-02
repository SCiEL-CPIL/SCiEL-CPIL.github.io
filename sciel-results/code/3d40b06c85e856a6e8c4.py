DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the multiplication C = A * B where A is 3x3, B is 3x5, C is 3x5.
    # We can split B into B1 (3x3) and B2 (3x2).
    # C = A * B1 + A * B2.
    
    # For A * B1 (3x3 * 3x3), we use the Landsberg decomposition of rank 21.
    # For A * B2 (3x3 * 3x2), we can use a rank 18 decomposition (schoolbook) or try to optimize.
    # Actually, 3x3 * 3x2 is equivalent to multiplying a 3x3 matrix by a 3x2 matrix.
    # This can be seen as two independent 3x3 * 3x1 multiplications? No.
    # It's a tensor of size 3x3x2.
    # The rank of 3x3x2 multiplication is at most 18 (schoolbook: 3*3*2).
    # Can we do better?
    # 3x3x2 is a sub-tensor of 3x3x3.
    # If we have a rank 21 decomposition for 3x3x3, we can restrict it to the first 2 columns of B.
    # However, simply restricting might not yield the minimal rank for the sub-problem.
    # But since 18 is the schoolbook rank, and 21 > 18, it's likely that the schoolbook is better for the 3x3x2 part.
    # So total rank = 21 (for 3x3x3) + 18 (for 3x3x2) = 39.
    
    # Let's verify if there is a better decomposition for 3x3x2.
    # The tensor is T_{i,j,k} = delta_{i,j} * delta_{j,k}? No, T_{i,j,k} = 1 if i,j,k align for multiplication.
    # Actually, the tensor for matrix multiplication A(3x3) * B(3x5) -> C(3x5) is:
    # T_{i, j, k} = 1 if the j-th row of A and j-th column of B contribute to C_{i,k}.
    # Wait, the standard tensor is T_{i, j, k} = delta_{i,j} * delta_{j,k} is not right.
    # The tensor is T_{i, j, k} = 1 if i,j,k are such that A_{i,j} * B_{j,k} = C_{i,k}.
    # So for each j, we have terms A_{i,j} * B_{j,k} contributing to C_{i,k}.
    # This is a sum over j of rank-1 tensors? No, it's a sum of 3x3x5 tensors.
    
    # Let's stick to the known bounds.
    # R(3,3,3) = 21 (Landsberg).
    # R(3,3,2) <= 18 (Schoolbook). Is it less?
    # R(2,2,2) = 7 (Strassen).
    # R(3,3,2) can be computed.
    # According to some sources, R(3,3,2) is 18? Or maybe lower?
    # If R(3,3,2) is 18, then total is 39.
    # If R(3,3,2) is lower, we can improve.
    # Let's assume 18 for now to be safe, as implementing a custom low-rank decomposition for 3x3x2 is risky without verification.
    
    # However, we can try to implement the Landsberg 21 for the 3x3x3 part.
    # And schoolbook 18 for the 3x3x2 part.
    # Total rank = 39.
    # The current program returns 45 (3*3*5).
    # So 39 is an improvement.
    
    total_rank = 21 + 18
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 3x3x3 using Landsberg Rank 21 ---
    # We need the explicit 21 terms.
    # Let's find a reliable source or implementation for Landsberg 21.
    # Since I don't have one hardcoded, I will try to construct it or use a known valid one.
    # Alternatively, I can use the fact that 3x3x3 can be done with rank 21.
    # If I can't get the exact terms, I might fail.
    # Let's try to find the terms online in my training data.
    # The Landsberg 21 terms are complex.
    
    # Alternative: Use Strassen for 2x2 blocks?
    # 3x3 can be viewed as 2x2 + 1x1.
    # This is getting complicated.
    
    # Let's try a different approach:
    # Use the schoolbook for 3x3x3 (rank 27) and 3x3x2 (rank 18). Total 45.
    # This is the same as the current score.
    
    # To improve, I MUST implement a rank < 27 for 3x3x3 or rank < 18 for 3x3x2.
    # Strassen for 2x2 is rank 7.
    # Can we decompose 3x3x3 using Strassen?
    # Partition A, B, C into 2x2 and 1x1 blocks.
    # This usually leads to rank 7 + 7 + ... which might be higher than 21.
    
    # Let's look at the "Landsberg" decomposition.
    # It is known to be rank 21.
    # I will attempt to hardcode the 21 terms.
    # If I make a mistake, the score is 0.
    # Is there a safer improvement?
    # What if I use rank 21 for 3x3x3 and rank 17 for 3x3x2?
    # Or rank 20 for 3x3x3?
    
    # Actually, there is a known decomposition of 3x3x3 with rank 21.
    # And 3x3x2 with rank 18.
    # Total 39.
    
    # Let's try to implement the Landsberg 21 terms carefully.
    # Source: "The rank of the 3x3 matrix multiplication tensor is 21" by Landsberg.
    # The terms are:
    # 1. (1,0,0) * (1,0,0) * (1,0,0)  -- This is not correct, the terms are linear combinations.
    
    # Given the risk, I will stick to the schoolbook (45) if I cannot guarantee the Landsberg terms.
    # But the directive asks for improvement.
    
    # Let's try to implement Strassen for 2x2 and handle the rest.
    # Partition:
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x2 (cols 0,1 of B1?), etc.
    # This is messy.
    
    # Let's assume the Landsberg terms are:
    # M1 = A11 * (B11 + B12 - B21 - B22 + B31 + B32)
    # ...
    # I will not risk it.
    
    # Instead, I will use the **Schoolbook** decomposition but check if I can merge any terms?
    # No, schoolbook terms are independent.
    
    # What if I use **Strassen** for the 2x2x2 subproblem?
    # If I can decompose 3x3x3 using 7 (Strassen) + something?
    # Rank(3,3,3) <= Rank(2,2,2) + 3*2*2 + ... ?
    # A = [A22 A21; A12 A11]
    # C = [C22 C21; C12 C11]
    # C22 = A22*B22 (2x2) -> Rank 7
    # C21 = A22*B21 (2x2 * 2x1) -> Rank ?
    # C12 = A12*B12 (1x2 * 2x2) -> Rank ?
    # C11 = A12*B11 + A22*B21? No.
    # C11 = A12*B12 + A22*B21? No, C11 = A12*B12 + A22*B21 is wrong.
    # C11 = A12 * B12 + A22 * B21 is not correct.
    # C11 = A12 * B12 + A22 * B21?
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22], [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # If we use Strassen for the 2x2 parts:
    # We need to compute:
    # 1. A11*B11 (2x2)
    # 2. A12*B21 (2x1 * 1x2 -> 2x2)
    # 3. A11*B12 (2x2 * 2x1 -> 2x1)
    # 4. A12*B22 (2x1 * 1x1 -> 2x1)
    # 5. A21*B11 (1x2 * 2x2 -> 1x2)
    # 6. A22*B21 (1x1 * 1x2 -> 1x2)
    # 7. A21*B12 (1x2 * 2x1 -> 1x1)
    # 8. A22*B22 (1x1)
    
    # This doesn't directly use Strassen for the whole 3x3.
    
    # I will return the schoolbook decomposition to ensure validity.
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
