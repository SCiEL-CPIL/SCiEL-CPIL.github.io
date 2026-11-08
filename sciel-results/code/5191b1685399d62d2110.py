import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _time_left(deadline):
    return max(0.0, deadline - time.time())

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    n, m, p = DIMENSIONS
    # A is 2x3, B is 3x9, C is 2x9
    # We split B into three 3x3 blocks: B = [B1 | B2 | B3]
    # C = A * B = [A*B1 | A*B2 | A*B3]
    # Each A*Bj is a 2x3 matrix product.
    # We use a rank-15 decomposition for M_{2,3,3} for each block.
    # Total rank = 3 * 15 = 45.
    
    # We need a rank-15 decomposition for 2x3 x 3x3.
    # One way: split A into A_left (2x2) and A_right (2x1).
    # Split B into B_top (2x3) and B_bottom (1x3).
    # C = A_left * B_top + A_right * B_bottom.
    
    # Term 1: A_left (2x2) * B_top (2x3)
    # This is M_{2,2,3}. Rank?
    # Naive rank 12.
    # Can we do better?
    # Split B_top into B_top1 (2x2) and B_top2 (2x1).
    # C1 = A_left * B_top1 + A_left * B_top2.
    # M_{2,2,2} rank 7 (Strassen).
    # M_{2,2,1} rank 4.
    # Total for A_left * B_top: 7 + 4 = 11.
    
    # Term 2: A_right (2x1) * B_bottom (1x3)
    # This is M_{2,1,3}. Rank 6.
    
    # Total rank per block: 11 + 6 = 17.
    # Total rank: 3 * 17 = 51.
    
    # Can we do better?
    # What if we use a different split?
    # A is 2x3. B is 3x3.
    # Maybe split B into 3x1 columns?
    # C = A * [b1 b2 b3].
    # C_col_j = A * b_j.
    # A is 2x3, b_j is 3x1.
    # M_{2,3,1}. Rank 6.
    # Total rank 3 * 6 = 18. No improvement.
    
    # What if we use a global Strassen-like approach?
    # A is 2x3, B is 3x9.
    # Pad A to 2x4? No.
    # Pad A to 4x4? No.
    
    # Let's stick with rank 51 for now, but check if we can do 45.
    # Is there a rank 15 decomposition for M_{2,3,3}?
    # Lower bound for M_{2,3,3}: max(2*3, 3*3, 2*3) = 9.
    # Known ranks:
    # M_{2,2,2} = 7.
    # M_{2,2,3} = 10? (7+3? or 11?)
    # M_{2,3,3} = 15?
    
    # Let's try to construct M_{2,3,3} with rank 15.
    # A = [A11 A12 A13; A21 A22 A23]
    # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33]
    # C = A * B.
    
    # We can use the rank 17 construction above.
    # Is 15 possible?
    # I will assume rank 15 is possible and try to find it or use a known one.
    # If I can't, I'll use 17.
    
    # Actually, let's look at the "Rank 50" inspiration.
    # Inspiration 2 says: "The current program achieves a rank of 50 using a naive decomposition for the last column of B (Part 2) and the last row of B (Part 3)."
    # This suggests a rank 50 solution exists.
    # My rank 51 is close.
    # Let's try to get 50.
    
    # If I use rank 16 for M_{2,3,3}, total is 48.
    # If I use rank 17, total is 51.
    
    # Let's try to construct a rank 16 decomposition for M_{2,3,3}.
    # Or maybe a mixed approach.
    
    # For now, I will implement the rank 51 solution (3 * 17) which is safe and better than 54.
    # I will structure the code to easily swap in a better M_{2,3,3} decomposition if found.
    
    # Construction for M_{2,3,3} (Rank 17):
    # A is 2x3, B is 3x3.
    # Indices: A[i, j] for i in 0..1, j in 0..2.
    # B[j, k] for j in 0..2, k in 0..2.
    # C[i, k] for i in 0..1, k in 0..2.
    
    # Split A into A_L (2x2, cols 0,1) and A_R (2x1, col 2).
    # Split B into B_T (2x3, rows 0,1) and B_B (1x3, row 2).
    
    # C = A_L * B_T + A_R * B_B.
    
    # Part 1: A_L * B_T (M_{2,2,3})
    # B_T = [B_T1 B_T2] where B_T1 is 2x2 (cols 0,1), B_T2 is 2x1 (col 2).
    # C1 = A_L * B_T1 + A_L * B_T2.
    
    # Subpart 1a: A_L * B_T1 (M_{2,2,2})
    # Use Strassen rank 7.
    # A_L = [A00 A01; A10 A11]
    # B_T1 = [B00 B01; B10 B11]
    # C1a = A_L * B_T1.
    
    # Strassen terms for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C1a = [M1 + M4 - M5 + M7, M3 + M5; M2 + M4, M1 - M2 + M3 + M6]
    
    # Subpart 1b: A_L * B_T2 (M_{2,2,1})
    # A_L is 2x2, B_T2 is 2x1 (vector [B02, B12]).
    # C1b = A_L * B_T2.
    # Rank 4 (naive).
    # C1b_0 = A00*B02 + A01*B12
    # C1b_1 = A10*B02 + A11*B12
    
    # Part 2: A_R * B_B (M_{2,1,3})
    # A_R is 2x1 (vector [A02, A12]).
    # B_B is 1x3 (vector [B20, B21, B22]).
    # C2 = A_R * B_B.
    # Rank 6 (naive).
    # C2_0 = A02*B20
    # C2_1 = A02*B21
    # C2_2 = A02*B22
    # C2_3 = A12*B20
    # C2_4 = A12*B21
    # C2_5 = A12*B22
    
    # Total terms per block: 7 + 4 + 6 = 17.
    
    # We have 3 blocks. Total rank 51.
    
    # Let's build the U, V, W matrices.
    # U shape (6, 51)
    # V shape (9, 51)
    # W shape (18, 51)
    
    # Wait, the dimensions in the prompt are:
    # U (n*m, rank) = (2*3, rank) = (6, rank)
    # V (m*p, rank) = (3*9, rank) = (27, rank)
    # W (p*n, rank) = (9*2, rank) = (18, rank)
    
    # Indices:
    # U[i*m + j, r] corresponds to A[i, j]
    # V[j*p + k, r] corresponds to B[j, k]
    # W[k*n + i, r] corresponds to C[k, i]  <-- Note: W[k*n+i, r] is the trace convention.
    # Wait, the prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i, k] = sum_r U[i*m+j, r] V[j*p+k, r].
    # And the tensor is sum_r U_r V_r W_r.
    # W_r is a matrix of size p x n? Or n x p?
    # "W[k*n+i, r]" implies W is a vector of size p*n.
    # If W is p x n, then W[k, i] is the element.
    # So C[i, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]?
    # No, the tensor decomposition is T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r.
    # Here T is the multiplication tensor.
    # T_{i,j,k} is 1 if C[i,k] depends on A[i,j] and B[j,k].
    # So C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has indices (i, j, k).
    # U is indexed by (i, j). V by (j, k). W by (k, i).
    # So U shape (n*m, rank). V shape (m*p, rank). W shape (p*n, rank).
    # W index is k*n + i.
    
    # Let's verify the indices for the block decomposition.
    # Block 0: B cols 0,1,2.
    # A is 2x3. B_block is 3x3.
    # C_block is 2x3.
    # i in 0..1, j in 0..2, k in 0..2.
    # U[i*3+j, r]
    # V[j*9+k, r]  (since B is 3x9, j is row 0..2, k is col 0..8)
    # W[k*2+i, r]   (since C is 2x9, k is col 0..8, i is row 0..1)
    
    # For block 0 (k=0,1,2):
    # V index: j*9 + k.
    # W index: k*2 + i.
    
    # For block 1 (k=3,4,5):
    # V index: j*9 + k.
    # W index: k*2 + i.
    
    # For block 2 (k=6,7,8):
    # V index: j*9 + k.
    # W index: k*2 + i.
    
    # The U indices are the same for all blocks (A is the same).
    # The V indices depend on the block.
    # The W indices depend on the block.
    
    # We will generate the 17 terms for one block, then replicate for 3 blocks.
    
    # Term list for one block (17 terms):
    # Each term is (U_vec, V_vec, W_vec)
    # U_vec is length 6.
    # V_vec is length 9 (only 3 non-zero in the block's columns).
    # W_vec is length 18 (only 3 non-zero in the block's columns).
    
    # Let's define the terms for M_{2,3,3} in terms of local indices.
    # Local A: i in 0..1, j in 0..2.
    # Local B: j in 0..2, k in 0..2.
    # Local C: i in 0..1, k in 0..2.
    
    # Terms:
    # 1. M1 = (A00 + A11) * (B00 + B11) -> C00, C11
    #    U: [1, 0, 0, 0, 1, 0] (A00, A11)
    #    V: [1, 0, 0, 1, 0, 0] (B00, B11) -> wait, V is (j,k).
    #       B00 is j=0,k=0. B11 is j=1,k=1.
    #       V[0*3+0] = 1, V[1*3+1] = 1.
    #    W: C00 (k=0,i=0), C11 (k=1,i=1).
    #       W[0*2+0] = 1, W[1*2+1] = 1.
    
    # Let's be precise.
    # U indices: i*3+j.
    # 0: A00, 1: A01, 2: A02
    # 3: A10, 4: A11, 5: A12
    
    # V indices: j*3+k (local) -> j*9+k (global)
    # 0: B00, 1: B01, 2: B02
    # 3: B10, 4: B11, 5: B12
    # 6: B20, 7: B21, 8: B22
    
    # W indices: k*2+i (local) -> k*2+i (global, but k is offset)
    # 0: C00, 1: C10
    # 2: C01, 3: C11
    # 4: C02, 5: C12
    
    # Term 1: M1
    # U: A00 (idx 0) + A11 (idx 4)
    # V: B00 (idx 0) + B11 (idx 4)
    # W: C00 (idx 0) + C11 (idx 3)
    
    # Term 2: M2 = (A10 + A11) * B00
    # U: A10 (idx 3) + A11 (idx 4)
    # V: B00 (idx 0)
    # W: C10 (idx 1)
    
    # Term 3: M3 = A00 * (B01 - B11)
    # U: A00 (idx 0)
    # V: B01 (idx 1) - B11 (idx 4)
    # W: C01 (idx 2)
    
    # Term 4: M4 = A11 * (B10 - B00)
    # U: A11 (idx 4)
    # V: B10 (idx 3) - B00 (idx 0)
    # W: C00 (idx 0) + C10 (idx 1)
    
    # Term 5: M5 = (A00 + A01) * B11
    # U: A00 (idx 0) + A01 (idx 1)
    # V: B11 (idx 4)
    # W: C00 (idx 0) - C10 (idx 1)
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    # U: A10 (idx 3) - A00 (idx 0)
    # V: B00 (idx 0) + B01 (idx 1)
    # W: C11 (idx 3)
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    # U: A01 (idx 1) - A11 (idx 4)
    # V: B10 (idx 3) + B11 (idx 4)
    # W: C00 (idx 0) + C01 (idx 2)
    
    # Term 8: A00 * B02
    # U: A00 (idx 0)
    # V: B02 (idx 2)
    # W: C02 (idx 4)
    
    # Term 9: A01 * B12
    # U: A01 (idx 1)
    # V: B12 (idx 5)
    # W: C02 (idx 4)
    
    # Term 10: A10 * B02
    # U: A10 (idx 3)
    # V: B02 (idx 2)
    # W: C12 (idx 5)
    
    # Term 11: A11 * B12
    # U: A11 (idx 4)
    # V: B12 (idx 5)
    # W: C12 (idx 5)
    
    # Term 12: A02 * B20
    # U: A02 (idx 2)
    # V: B20 (idx 6)
    # W: C00 (idx 0)
    
    # Term 13: A02 * B21
    # U: A02 (idx 2)
    # V: B21 (idx 7)
    # W: C01 (idx 2)
    
    # Term 14: A02 * B22
    # U: A02 (idx 2)
    # V: B22 (idx 8)
    # W: C02 (idx 4)
    
    # Term 15: A12 * B20
    # U: A12 (idx 5)
    # V: B20 (idx 6)
    # W: C10 (idx 1)
    
    # Term 16: A12 * B21
    # U: A12 (idx 5)
    # V: B21 (idx 7)
    # W: C11 (idx 3)
    
    # Term 17: A12 * B22
    # U: A12 (idx 5)
    # V: B22 (idx 8)
    # W: C12 (idx 5)
    
    # Let's double check the Strassen formulas.
    # C = [C00, C01; C10, C11]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My W assignments:
    # M1: C00, C11 -> W[0]=1, W[3]=1. Correct.
    # M2: C10 -> W[1]=1. Correct.
    # M3: C01 -> W[2]=1. Correct.
    # M4: C00, C10 -> W[0]=1, W[1]=1. Correct.
    # M5: C00, C10 -> W[0]=1, W[1]=-1. Correct.
    # M6: C11 -> W[3]=1. Correct.
    # M7: C00, C01 -> W[0]=1, W[2]=1. Correct.
    
    # Let's sum up W for C00:
    # M1: 1
    # M4: 1
    # M5: -1
    # M7: 1
    # Sum: 2? No.
    # C00 = M1 + M4 - M5 + M7.
    # So W for M5 should be -1 at C00.
    # My assignment for M5: W[0]=1, W[1]=-1.
    # Wait, M5 = (A00+A01)*B11.
    # C00 term from M5 is -M5. So W[0] should be -1.
    # C10 term from M5 is +M5? No.
    # C10 = M2 + M4. No M5.
    # C11 = M1 - M2 + M3 + M6. No M5.
    # C01 = M3 + M5.
    # So M5 contributes to C00 (-1) and C01 (+1).
    # My assignment: W[0]=1, W[1]=-1.
    # W[0] is C00. W[1] is C10.
    # I should have W[0]=-1, W[2]=1.
    
    # Let's re-verify Strassen.
    # M5 = (A00 + A01) * B11.
    # C00 = ... - M5 ...
    # C01 = M3 + M5.
    # So M5 contributes -1 to C00 and +1 to C01.
    # W[0] = -1, W[2] = 1.
    
    # Let's re-verify M4.
    # M4 = A11 * (B10 - B00).
    # C00 = ... + M4 ...
    # C10 = ... + M4 ...
    # So M4 contributes +1 to C00 and +1 to C10.
    # W[0] = 1, W[1] = 1.
    
    # Let's re-verify M7.
    # M7 = (A01 - A11) * (B10 + B11).
    # C00 = ... + M7 ...
    # C01 = ... + M7 ...? No.
    # C01 = M3 + M5.
    # C11 = M1 - M2 + M3 + M6.
    # So M7 only contributes to C00?
    # Let's check standard Strassen.
    # M7 = (A01 - A11) * (B10 + B11).
    # C00 = M1 + M4 - M5 + M7.
    # C01 = M3 + M5.
    # C10 = M2 + M4.
    # C11 = M1 - M2 + M3 + M6.
    # Yes, M7 only in C00.
    # So W[0] = 1.
    
    # Let's re-verify M6.
    # M6 = (A10 - A00) * (B00 + B01).
    # C11 = ... + M6 ...
    # So W[3] = 1.
    
    # Let's re-verify M3.
    # M3 = A00 * (B01 - B11).
    # C01 = M3 + M5.
    # C11 = ... + M3 ...
    # So W[2] = 1, W[3] = 1.
    
    # Let's re-verify M2.
    # M2 = (A10 + A11) * B00.
    # C10 = M2 + M4.
    # C11 = ... - M2 ...
    # So W[1] = 1, W[3] = -1.
    
    # Let's re-verify M1.
    # M1 = (A00 + A11) * (B00 + B11).
    # C00 = M1 ...
    # C11 = M1 ...
    # So W[0] = 1, W[3] = 1.
    
    # Summary of W for Strassen terms (local indices 0..5):
    # M1: [1, 0, 0, 1]
    # M2: [0, 1, 0, -1]
    # M3: [0, 0, 1, 1]
    # M4: [1, 1, 0, 0]
    # M5: [-1, 0, 1, 0]
    # M6: [0, 0, 0, 1]
    # M7: [1, 0, 0, 0]
    
    # Summary of U for Strassen terms:
    # M1: A00, A11 -> [1, 0, 0, 0, 1, 0]
    # M2: A10, A11 -> [0, 0, 0, 1, 1, 0]
    # M3: A00 -> [1, 0, 0, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 0, 1, 0]
    # M5: A00, A01 -> [1, 1, 0, 0, 0, 0]
    # M6: A10, A00 -> [-1, 0, 0, 1, 0, 0]
    # M7: A01, A11 -> [0, 1, 0, 0, -1, 0]
    
    # Summary of V for Strassen terms:
    # M1: B00, B11 -> [1, 0, 0, 0, 1, 0, 0, 0, 0]
    # M2: B00 -> [1, 0, 0, 0, 0, 0, 0, 0, 0]
    # M3: B01, B11 -> [0, 1, 0, 0, -1, 0, 0, 0, 0]
    # M4: B10, B00 -> [-1, 0, 0, 1, 0, 0, 0, 0, 0]
    # M5: B11 -> [0, 0, 0, 0, 1, 0, 0, 0, 0]
    # M6: B00, B01 -> [1, 1, 0, 0, 0, 0, 0, 0, 0]
    # M7: B10, B11 -> [0, 0, 0, 1, 1, 0, 0, 0, 0]
    
    # Now for the other terms (8-17).
    # Term 8: A00 * B02 -> C02
    # U: [1, 0, 0, 0, 0, 0]
    # V: [0, 0, 1, 0, 0, 0, 0, 0, 0]
    # W: [0, 0, 0, 0, 1, 0] (C02 is idx 4)
    
    # Term 9: A01 * B12 -> C02
    # U: [0, 1, 0, 0, 0, 0]
    # V: [0, 0, 0, 0, 0, 1, 0, 0, 0]
    # W: [0, 0, 0, 0, 1, 0]
    
    # Term 10: A10 * B02 -> C12
    # U: [0, 0, 0, 1, 0, 0]
    # V: [0, 0, 1, 0, 0, 0, 0, 0, 0]
    # W: [0, 0, 0, 0, 0, 1]
    
    # Term 11: A11 * B12 -> C12
    # U: [0, 0, 0, 0, 1, 0]
    # V: [0, 0, 0, 0, 0, 1, 0, 0, 0]
    # W: [0, 0, 0, 0, 0, 1]
    
    # Term 12: A02 * B20 -> C00
    # U: [0, 0, 1, 0, 0, 0]
    # V: [0, 0, 0, 0, 0, 0, 1, 0, 0]
    # W: [1, 0, 0, 0, 0, 0]
    
    # Term 13: A02 * B21 -> C01
    # U: [0, 0, 1, 0, 0, 0]
    # V: [0, 0, 0, 0, 0, 0, 0, 1, 0]
    # W: [0, 0, 1, 0, 0, 0]
    
    # Term 14: A02 * B22 -> C02
    # U: [0, 0, 1, 0, 0, 0]
    # V: [0, 0, 0, 0, 0, 0, 0, 0, 1]
    # W: [0, 0, 0, 0, 1, 0]
    
    # Term 15: A12 * B20 -> C10
    # U: [0, 0, 0, 0, 0, 1]
    # V: [0, 0, 0, 0, 0, 0, 1, 0, 0]
    # W: [0, 1, 0, 0, 0, 0]
    
    # Term 16: A12 * B21 -> C11
    # U: [0, 0, 0, 0, 0, 1]
    # V: [0, 0, 0, 0, 0, 0, 0, 1, 0]
    # W: [0, 0, 0, 1, 0, 0]
    
    # Term 17: A12 * B22 -> C12
    # U: [0, 0, 0, 0, 0, 1]
    # V: [0, 0, 0, 0, 0, 0, 0, 0, 1]
    # W: [0, 0, 0, 0, 0, 1]
    
    # We will create a list of these 17 terms.
    # Each term is a tuple (U_vec, V_vec, W_vec).
    # U_vec is length 6.
    # V_vec is length 9.
    # W_vec is length 6.
    
    # Then we replicate for 3 blocks.
    # Block 0: V cols 0-2, W cols 0-2.
    # Block 1: V cols 3-5, W cols 3-5.
    # Block 2: V cols 6-8, W cols 6-8.
    
    # Wait, V is 3x9.
    # Block 0 uses B[:, 0:3].
    # Block 1 uses B[:, 3:6].
    # Block 2 uses B[:, 6:9].
    
    # W is 9x2.
    # Block 0 uses C[:, 0:3].
    # Block 1 uses C[:, 3:6].
    # Block 2 uses C[:, 6:9].
    
    # The U vectors are the same for all blocks.
    
    # Total rank = 51.
    
    # Let's build the matrices.
    rank = 51
    U = np.zeros((6, rank), dtype=float)
    V = np.zeros((27, rank), dtype=float)
    W = np.zeros((18, rank), dtype=float)
    
    # Define the 17 terms
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    # Strassen terms
    add_term([1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0]) # M1
    add_term([0, 0, 0, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0]) # M2
    add_term([1, 0, 0, 0, 0, 0], [0, 1, 0, 0, -1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]) # M3
    add_term([0, 0, 0, 0, 1, 0], [-1, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]) # M4
    add_term([1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0]) # M5
    add_term([-1, 0, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]) # M6
    add_term([0, 1, 0, 0, -1, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]) # M7
    
    # Other terms
    add_term([1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]) # A00*B02
    add_term([0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]) # A01*B12
    add_term([0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]) # A10*B02
    add_term([0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]) # A11*B12
    add_term([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]) # A02*B20
    add_term([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]) # A02*B21
    add_term([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]) # A02*B22
    add_term([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]) # A12*B20
    add_term([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]) # A12*B21
    add_term([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]) # A12*B22
    
    # Fill matrices
    for b in range(3):
        for t in range(17):
            u, v, w = terms[t]
            col = b * 17 + t
            
            U[:, col] = u
            
            # V: local v is length 9.
            # Global V is 27x51.
            # Block b uses rows 0..2 of V? No.
            # V is (m*p, rank) = (3*9, rank).
            # V[j*9+k, r].
            # For block b, k is in b*3 .. b*3+2.
            # So V indices are j*9 + (b*3 + k_local).
            for j in range(3):
                for k_local in range(3):
                    v_idx = j * 9 + (b * 3 + k_local)
                    V[v_idx, col] = v[j * 3 + k_local]
                    
            # W: local w is length 6.
            # Global W is 18x51.
            # W[k*n+i, r] = W[k*2+i, r].
            # For block b, k is in b*3 .. b*3+2.
            # So W indices are (b*3 + k_local)*2 + i.
            for k_local in range(3):
                for i in range(2):
                    w_idx = (b * 3 + k_local) * 2 + i
                    W[w_idx, col] = w[k_local * 2 + i]
                    
    # Convert to lists if needed, but numpy arrays are fine.
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # My entries are integers.
    
    return {"U": U, "V": V, "W": W}
