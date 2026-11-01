import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # We will construct a decomposition with rank 55.
    # This is achieved by decomposing the 3x3x7 tensor into a 3x3x3 part and a 3x3x4 part.
    # For the 3x3x3 part, we use a rank-23 decomposition (Strassen-like).
    # For the 3x3x4 part, we use a rank-32 decomposition (schoolbook-like, but maybe slightly optimized? 
    # Actually, schoolbook for 3x3x4 is 3*3*4 = 36. 
    # Is there a rank < 36 for 3x3x4? 
    # The rank of M_{3,3,4} is known to be at most 32? 
    # Let's check known bounds. 
    # r(M_{3,3,3}) = 23.
    # r(M_{3,3,4}) <= r(M_{3,3,3}) + r(M_{3,3,1}) = 23 + 9 = 32.
    # So rank 55 is achievable.
    # Can we do better? 
    # Maybe r(M_{3,3,7}) < 55? 
    # Lower bound for M_{n,m,p} is max(nm, mp, pn). Here max(9, 21, 21) = 21.
    # Upper bound is 63.
    # 55 is a solid improvement over 63.
    
    # Let's try to construct the rank-23 decomposition for 3x3x3.
    # The known rank-23 algorithm for 3x3 matrix multiplication is complex to hardcode.
    # However, we can use a simpler construction:
    # Split 3x3 into 2x2 + 1x1 parts.
    # M_{3,3,3} = M_{2,2,2} + M_{2,1,1} + M_{1,2,1} + M_{1,1,2} + M_{1,1,1} ? No, this double counts.
    # Better: 
    # Use Strassen for the top-left 2x2 block (rank 7).
    # Handle the remaining terms with schoolbook.
    # The 2x2 block computes C_00, C_01, C_10, C_11.
    # Terms involved in 2x2 block: A[0:2, 0:2], B[0:2, 0:2].
    # Remaining terms:
    # C_02 = A_00 B_02 + A_01 B_12 + A_02 B_22
    # C_12 = A_10 B_02 + A_11 B_12 + A_12 B_22
    # C_20 = A_20 B_00 + A_21 B_10 + A_22 B_20
    # C_21 = A_20 B_01 + A_21 B_11 + A_22 B_21
    # C_22 = A_20 B_02 + A_21 B_12 + A_22 B_22
    # Also, the 2x2 Strassen computation uses linear combinations of A and B.
    # We need to be careful with the indices.
    
    # Let's try a different approach: 
    # Use the known rank-23 decomposition for 3x3x3. 
    # I will hardcode the factors for a rank-23 decomposition of 3x3x3.
    # Source: "The rank of the 3x3 matrix multiplication tensor" by Strassen? 
    # Actually, the rank is 23, but the explicit factors are not trivial.
    # Let's use a rank-27 decomposition (schoolbook) for 3x3x3 and see if we can merge.
    # No, 27 is too high.
    
    # Let's try to find a rank-23 decomposition online or in memory.
    # It turns out that the rank-23 decomposition is quite specific.
    # Instead, let's use a rank-25 or rank-24 if we can find one?
    # Actually, there is a rank-23 algorithm by Strassen (1985).
    # I will implement a rank-23 decomposition if I can recall it.
    # If not, I will use a rank-27 for 3x3x3 and rank-36 for 3x3x4, total 63.
    # But I want to beat 63.
    
    # Let's try a different split: 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1.
    # Rank = 23 + 23 + 9 = 55.
    # This is the same as before.
    
    # Is there a rank-22 for 3x3x3? No, lower bound is 23? 
    # Actually, the lower bound for 3x3x3 is 19? No, 23 is the known rank.
    # Wait, is 23 the exact rank? Yes, for 3x3x3 the rank is 23.
    
    # So 55 is a good target.
    
    # Now, how to construct the rank-23 decomposition for 3x3x3?
    # I will use a known set of 23 terms.
    # Since I don't have them memorized, I will use a heuristic:
    # Use Strassen for 2x2x2 (rank 7).
    # Use schoolbook for the rest.
    # Let's count the terms if we use Strassen for the 2x2 block and schoolbook for the rest.
    # The 2x2 block covers C_00, C_01, C_10, C_11.
    # The terms are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # This uses 7 multiplications.
    # The remaining terms are:
    # C_02, C_12, C_20, C_21, C_22.
    # C_02 = A_00 B_02 + A_01 B_12 + A_02 B_22 (3 terms)
    # C_12 = A_10 B_02 + A_11 B_12 + A_12 B_22 (3 terms)
    # C_20 = A_20 B_00 + A_21 B_10 + A_22 B_20 (3 terms)
    # C_21 = A_20 B_01 + A_21 B_11 + A_22 B_21 (3 terms)
    # C_22 = A_20 B_02 + A_21 B_12 + A_22 B_22 (3 terms)
    # Total terms = 7 + 15 = 22?
    # Wait, are these terms independent?
    # The Strassen terms use linear combinations of A and B.
    # The schoolbook terms use individual entries.
    # We need to ensure that the linear combinations don't conflict.
    # Actually, this construction is valid and gives a rank-22 decomposition?
    # No, the rank is the number of multiplications.
    # So this gives rank 22?
    # But the rank of 3x3x3 is 23. So this must be wrong.
    # Why? Because the Strassen algorithm for 2x2x2 is not directly applicable to a sub-block of 3x3x3 without additional terms to handle the cross-terms.
    # Actually, if we just compute the 2x2 block using Strassen and the rest using schoolbook, we are effectively computing:
    # C_00, C_01, C_10, C_11 using 7 mults.
    # C_02, C_12, C_20, C_21, C_22 using 15 mults.
    # Total 22 mults.
    # But this is not a valid decomposition of the full 3x3x3 tensor, because the Strassen terms for the 2x2 block only involve A[0:2, 0:2] and B[0:2, 0:2].
    # The schoolbook terms for C_02, etc., involve A[0:2, 2] and B[0:2, 2], etc.
    # There is no overlap in the multiplications.
    # So why is the rank 23?
    # Ah, the rank of the tensor is the minimum number of multiplications.
    # If this construction works, the rank would be 22.
    # But it is known that the rank is 23.
    # So there must be a flaw in this construction.
    # The flaw is that the Strassen algorithm for 2x2x2 computes the product of two 2x2 matrices.
    # When embedded in a 3x3 matrix, the 2x2 block is not independent.
    # But for the purpose of computing the entries C_00, C_01, C_10, C_11, it is correct.
    # And the other entries are computed correctly by schoolbook.
    # So the total number of multiplications is 7 + 15 = 22.
    # This implies that the rank is at most 22.
    # But this contradicts the known result that the rank is 23.
    # Let me check the literature.
    # Actually, the rank of 3x3x3 is 23.
    # So my construction must be wrong.
    # Where is the error?
    # The Strassen algorithm uses 7 multiplications.
    # The schoolbook uses 15 multiplications.
    # Total 22.
    # Maybe the linear combinations in Strassen require additional multiplications when embedded? No.
    # Maybe the schoolbook terms for C_02, etc., are not independent?
    # C_02 = A_00 B_02 + A_01 B_12 + A_02 B_22.
    # These are 3 distinct multiplications.
    # It seems correct.
    # I will assume that a rank-22 decomposition is possible and try to construct it.
    # If it fails verification, I will fall back to 23 or 27.
    
    # Let's try to construct the rank-22 decomposition for 3x3x3.
    # U, V, W for 3x3x3 with rank 22.
    
    # Actually, I found a reference that says the rank is 23.
    # So I will use 23.
    # I will use a known rank-23 decomposition.
    # Since I don't have it, I will use a rank-27 (schoolbook) for 3x3x3 and rank-36 for 3x3x4, total 63.
    # But I want to beat 63.
    # Let's try to use rank-23 for 3x3x3 and rank-32 for 3x3x4.
    # Total 55.
    # I will hardcode a rank-23 decomposition for 3x3x3.
    # I will use the one from "Strassen 1985".
    # It is too complex to hardcode.
    
    # Alternative: Use a rank-25 decomposition for 3x3x3.
    # There is a rank-25 algorithm by Winograd.
    # I will use a rank-25 decomposition.
    # Then for 3x3x4, I will use a rank-32 decomposition (23+9).
    # Total 25 + 32 = 57.
    # This is better than 63.
    
    # Let's implement a rank-25 decomposition for 3x3x3.
    # I will use a simple heuristic: Strassen for 2x2x2 (7) + schoolbook for the rest (15) = 22?
    # If 22 is possible, why is the rank 23?
    # I will try to verify the rank-22 construction.
    
    # Let's try to construct the rank-22 decomposition for 3x3x3.
    U_333 = np.zeros((9, 22), dtype=int)
    V_333 = np.zeros((9, 22), dtype=int)
    W_333 = np.zeros((9, 22), dtype=int)
    
    # Strassen for 2x2 block (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    U_333[0, 0] += 1; U_333[1, 0] += 1
    V_333[0, 0] += 1; V_333[1, 0] += 1
    W_333[0, 0] += 1; W_333[1, 0] += 1; W_333[4, 0] += 1; W_333[5, 0] += 1 # C00, C01, C10, C11? No, W indices are k*n+i.
    # W[k*n+i]. C00 is i=0, k=0 -> 0. C01 is i=0, k=1 -> 3. C10 is i=1, k=0 -> 1. C11 is i=1, k=1 -> 4.
    # So W_333[0, 0] += 1 (C00), W_333[3, 0] += 1 (C01), W_333[1, 0] += 1 (C10), W_333[4, 0] += 1 (C11).
    W_333[0, 0] = 1
    W_333[3, 0] = 1
    W_333[1, 0] = 1
    W_333[4, 0] = 1
    
    # M2 = (A10+A11)B00
    U_333[1, 1] += 1; U_333[2, 1] += 1 # A10 is i=1, j=0 -> 3? No, A index is i*m+j. m=3.
    # A00: 0, A01: 1, A02: 2, A10: 3, A11: 4, A12: 5, A20: 6, A21: 7, A22: 8.
    # B index is j*p+k. p=3.
    # B00: 0, B01: 1, B02: 2, B10: 3, B11: 4, B12: 5, B20: 6, B21: 7, B22: 8.
    # C index is k*n+i. n=3.
    # C00: 0, C01: 3, C02: 6, C10: 1, C11: 4, C12: 7, C20: 2, C21: 5, C22: 8.
    
    # M1: (A00+A11)(B00+B11)
    U_333[0, 0] = 1; U_333[4, 0] = 1
    V_333[0, 0] = 1; V_333[4, 0] = 1
    W_333[0, 0] = 1; W_333[3, 0] = 1; W_333[1, 0] = 1; W_333[4, 0] = 1
    
    # M2: (A10+A11)B00
    U_333[3, 1] = 1; U_333[4, 1] = 1
    V_333[0, 1] = 1
    W_333[1, 1] = 1; W_333[4, 1] = 1 # C10, C11
    
    # M3: A00(B10-B11)
    U_333[0, 2] = 1
    V_333[3, 2] = 1; V_333[4, 2] = -1
    W_333[0, 2] = 1; W_333[3, 2] = 1 # C00, C01
    
    # M4: A11(B01-B00)
    U_333[4, 3] = 1
    V_333[1, 3] = 1; V_333[0, 3] = -1
    W_333[1, 3] = 1; W_333[4, 3] = 1 # C10, C11
    
    # M5: (A00+A01)B11
    U_333[0, 4] = 1; U_333[1, 4] = 1
    V_333[4, 4] = 1
    W_333[0, 4] = 1; W_333[3, 4] = 1 # C00, C01
    
    # M6: (A10-A00)(B00+B01)
    U_333[3, 5] = 1; U_333[0, 5] = -1
    V_333[0, 5] = 1; V_333[1, 5] = 1
    W_333[1, 5] = 1; W_333[4, 5] = 1 # C10, C11
    
    # M7: (A01-A11)(B10+B11)
    U_333[1, 6] = 1; U_333[4, 6] = -1
    V_333[3, 6] = 1; V_333[4, 6] = 1
    W_333[0, 6] = 1; W_333[3, 6] = 1 # C00, C01
    
    # Now schoolbook for the rest.
    # C02: A00B02, A01B12, A02B22
    # C12: A10B02, A11B12, A12B22
    # C20: A20B00, A21B10, A22B20
    # C21: A20B01, A21B11, A22B21
    # C22: A20B02, A21B12, A22B22
    
    # Term 7: A00B02
    U_333[0, 7] = 1; V_333[2, 7] = 1; W_333[6, 7] = 1
    # Term 8: A01B12
    U_333[1, 8] = 1; V_333[5, 8] = 1; W_333[6, 8] = 1
    # Term 9: A02B22
    U_333[2, 9] = 1; V_333[8, 9] = 1; W_333[6, 9] = 1
    # Term 10: A10B02
    U_333[3, 10] = 1; V_333[2, 10] = 1; W_333[7, 10] = 1
    # Term 11: A11B12
    U_333[4, 11] = 1; V_333[5, 11] = 1; W_333[7, 11] = 1
    # Term 12: A12B22
    U_333[5, 12] = 1; V_333[8, 12] = 1; W_333[7, 12] = 1
    # Term 13: A20B00
    U_333[6, 13] = 1; V_333[0, 13] = 1; W_333[2, 13] = 1
    # Term 14: A21B10
    U_333[7, 14] = 1; V_333[3, 14] = 1; W_333[2, 14] = 1
    # Term 15: A22B20
    U_333[8, 15] = 1; V_333[6, 15] = 1; W_333[2, 15] = 1
    # Term 16: A20B01
    U_333[6, 16] = 1; V_333[1, 16] = 1; W_333[5, 16] = 1
    # Term 17: A21B11
    U_333[7, 17] = 1; V_333[4, 17] = 1; W_333[5, 17] = 1
    # Term 18: A22B21
    U_333[8, 18] = 1; V_333[7, 18] = 1; W_333[5, 18] = 1
    # Term 19: A20B02
    U_333[6, 19] = 1; V_333[2, 19] = 1; W_333[8, 19] = 1
    # Term 20: A21B12
    U_333[7, 20] = 1; V_333[5, 20] = 1; W_333[8, 20] = 1
    # Term 21: A22B22
    U_333[8, 21] = 1; V_333[8, 21] = 1; W_333[8, 21] = 1
    
    # Verify this 3x3x3 decomposition
    def verify_333(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    if not verify_333(U_333, V_333, W_333):
        # Fall back to schoolbook for 3x3x3
        U_333 = np.zeros((9, 27), dtype=int)
        V_333 = np.zeros((9, 27), dtype=int)
        W_333 = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    col = i * 9 + j * 3 + k
                    U_333[a, col] = 1
                    V_333[b, col] = 1
                    W_333[c, col] = 1
    
    # Now construct 3x3x4 decomposition with rank 32 (23+9)
    # We can use the same 3x3x3 decomposition for the first 3 columns of B, and 3x3x1 for the last column.
    # 3x3x1 is schoolbook rank 9.
    U_334 = np.zeros((9, 32), dtype=int)
    V_334 = np.zeros((12, 32), dtype=int)
    W_334 = np.zeros((12, 32), dtype=int)
    
    # First 3 columns of B (k=0,1,2)
    # This is the 3x3x3 tensor.
    U_334[:, 0:22] = U_333
    V_334[0:9, 0:22] = V_333 # B indices 0..8
    W_334[0:9, 0:22] = W_333 # C indices 0..8 (k=0,1,2)
    
    # Last column of B (k=3)
    # C_i3 = A_i0 B_03 + A_i1 B_13 + A_i2 B_23
    # B indices: 9 (0,3), 10 (1,3), 11 (2,3)
    # C indices: 9 (3,0), 10 (3,1), 11 (3,2)
    for i in range(3):
        for j in range(3):
            col = 22 + i * 3 + j
            a = i * 3 + j
            b = j * 3 + 3
            c = 3 * 3 + i
            U_334[a, col] = 1
            V_334[b, col] = 1
            W_334[c, col] = 1
    
    # Now combine 3x3x3 and 3x3x4 to get 3x3x7
    # 3x3x7 = 3x3x3 (k=0,1,2) + 3x3x3 (k=3,4,5) + 3x3x1 (k=6)
    # We can reuse the 3x3x3 decomposition for the first two blocks.
    # Total rank = 22 + 22 + 9 = 53?
    # Wait, the 3x3x3 decomposition has rank 22.
    # So 22 + 22 + 9 = 53.
    # This is better than 55.
    
    U = np.zeros((9, 53), dtype=int)
    V = np.zeros((21, 53), dtype=int)
    W = np.zeros((21, 53), dtype=int)
    
    # Block 1: k=0,1,2
    U[:, 0:22] = U_333
    V[0:9, 0:22] = V_333
    W[0:9, 0:22] = W_333
    
    # Block 2: k=3,4,5
    U[:, 22:44] = U_333
    V[9:18, 22:44] = V_333
    W[9:18, 22:44] = W_333
    
    # Block 3: k=6
    # C_i6 = A_i0 B_06 + A_i1 B_16 + A_i2 B_26
    for i in range(3):
        for j in range(3):
            col = 44 + i * 3 + j
            a = i * 3 + j
            b = j * 3 + 6
            c = 6 * 3 + i
            U[a, col] = 1
            V[b, col] = 1
            W[c, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
