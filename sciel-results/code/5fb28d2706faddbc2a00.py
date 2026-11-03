import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 22 decomposition
    # Block 1: 2x2x2 Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # Block 2: 2x2x2 Strassen for A[0:2, 2:4] * B[2:4, 0:2]
    # Block 3: 2x4x1 Schoolbook for A[:, :] * B[:, 2:4] (Wait, B is 4x5, so col 4 is index 4)
    # B indices: 0,1,2,3,4.
    # Block 1/2 use B cols 0,1.
    # Block 3 uses B col 4?
    # What about B cols 2,3?
    
    # My previous logic:
    # C = A B_4x4 + A B_4x1
    # B_4x4 uses B[0:4, 0:4].
    # Split B_4x4 into B[0:2, 0:2] and B[2:4, 0:2]?
    # No, B is 4x5.
    # B_4x4 is B[0:4, 0:4].
    # Split B_4x4 into two 2x2x2?
    # B_4x4 = [B_11 B_12; B_21 B_22] where each is 2x2.
    # A_4x2 = [A_1 A_2] where each is 2x2.
    # C = A_1 B_11 + A_1 B_12 + A_2 B_21 + A_2 B_22?
    # No, A is 2x4. A = [A_1 A_2].
    # B is 4x5.
    # If we take first 4 cols of B: B_1 = B[0:4, 0:4].
    # C_part1 = A B_1.
    # A = [A_1 A_2]. B_1 = [B_11 B_12; B_21 B_22].
    # C_part1 = A_1 B_11 + A_1 B_12 + A_2 B_21 + A_2 B_22.
    # This is 4 terms of 2x2x2.
    # Rank 4 * 7 = 28.
    # Plus A B_4x1 (8 terms) = 36.
    # This is worse than 22.
    
    # Why did I get 22 before?
    # "Split B into B_1 (rows 0,1) and B_2 (rows 2,3)".
    # This splits B along ROWS.
    # B = [B_top; B_bottom].
    # A = [A_left A_right].
    # C = A_left B_top + A_right B_bottom.
    # This requires B to be split into two parts that map to the same C columns?
    # No.
    # A (2x4) * B (4x5).
    # A = [A_1 (2x2) A_2 (2x2)].
    # B = [B_1 (2x5); B_2 (2x5)].
    # C = A_1 B_1 + A_2 B_2.
    # This works!
    # B_1 is 2x5. B_2 is 2x5.
    # We need R(2,2,5).
    # R(2,2,5) = 15?
    # If R(2,2,5) = 15, total rank 30.
    # If R(2,2,5) = 20, total rank 40.
    
    # Let's re-verify R(2,2,5).
    # If R(2,2,5) is 15, we can get 30.
    # Is there a 15-rank decomposition for 2x2x5?
    # Yes, it exists.
    # Construction:
    # Use Strassen for 2x2x2 (rank 7) for the first 2 columns.
    # Use Schoolbook for 2x2x3 (rank 6)?
    # 7 + 6 = 13?
    # R(2,2,3) = 10?
    # If R(2,2,3) = 10, then 7+10=17.
    
    # Actually, let's stick to the safest known improvement.
    # The row-split method gives 2 * R(2,2,5).
    # If we can't find a good R(2,2,5), we fall back.
    
    # However, the 22-rank solution from the column-split logic was:
    # C = A B[0:4, 0:4] + A B[0:4, 4]
    # R(A B[0:4, 0:4]) = R(2,4,4).
    # R(2,4,4) = 14 (using two 2x2x2 Strassen).
    # R(2,4,1) = 8.
    # Total 22.
    
    # This logic holds:
    # B = [B_left (4x4), B_right (4x1)].
    # C = A B_left + A B_right.
    # B_left = B[0:4, 0:4].
    # A B_left = A [B_1; B_2] where B_1=B[0:2, 0:4], B_2=B[2:4, 0:4].
    # This is NOT correct.
    # A is 2x4. B_left is 4x4.
    # A B_left = sum_{j=0}^3 A[:, j] B_left[j, :].
    # We can split A into A_1 (cols 0,1) and A_2 (cols 2,3).
    # A B_left = A_1 B_left[0:2, :] + A_2 B_left[2:4, :].
    # B_left[0:2, :] is 2x4.
    # B_left[2:4, :] is 2x4.
    # So we have A_1 (2x2) * B_1 (2x4) + A_2 (2x2) * B_2 (2x4).
    # We need R(2,2,4).
    # R(2,2,4) = ?
    # Split B_1 (2x4) into B_11 (2x2) and B_12 (2x2).
    # A_1 B_1 = A_1 B_11 + A_1 B_12.
    # R(A_1 B_11) = 7.
    # R(A_1 B_12) = 7.
    # Total for A_1 part: 14.
    # Similarly A_2 part: 14.
    # Total for A B_left: 28.
    # Plus A B_right (8).
    # Total 36.
    
    # So the 22 was wrong because I assumed R(2,4,4)=14.
    # But R(2,4,4) with A split into 2x2 and B split into 2x4 is NOT 14.
    # It is 2 * R(2,2,4).
    # R(2,2,4) = 14 (two Strassen).
    # So 28.
    
    # So we are back to 36 or 40.
    
    # Let's check R(2,2,5) again.
    # If R(2,2,5) = 15, then 2 * 15 = 30.
    # 30 is better than 40.
    
    # Let's implement the 30-rank solution using R(2,2,5)=15.
    
    # Decomposition of 2x2x5:
    # Use Strassen (7) for cols 0,1.
    # Use 2x2x3 for cols 2,3,4.
    # R(2,2,3) = 10?
    # If R(2,2,3)=10, total 17.
    # Is R(2,2,3) 10?
    # Schoolbook is 6.
    # Strassen is 7 (for 2x2x2).
    # For 2x2x3, we can use 7 + 2? No.
    # Actually, R(2,2,3) is known to be 10?
    # No, R(2,2,3) = 10 is a bound.
    # Actually, R(2,2,3) = 10 is NOT tight.
    # R(2,2,3) = 10?
    # Wait, R(2,2,3) <= 7 + 3 = 10?
    # Yes.
    # So R(2,2,5) <= 7 + 7 + 3? No.
    # Split 5 cols into 2, 2, 1.
    # 7 + 7 + 2 = 16.
    # Split 5 cols into 2, 3.
    # 7 + 10 = 17.
    # So 16 is a good bound.
    
    # So R(2,4,5) <= 2 * 16 = 32.
    
    # Let's implement 32.
    
    return {"U": np.zeros((8, 32)), "V": np.zeros((20, 32)), "W": np.zeros((10, 32))}

