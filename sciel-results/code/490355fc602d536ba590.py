import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for the 2x2x2 core and schoolbook for the 1x2x3 core.
    # Rank = 7 * 6 = 42.
    # A is 2x4, B is 4x6.
    # We can view A as a 2x2 block matrix where each block is 1x2.
    # A = [[A00, A01], [A10, A11]] where Aij is 1x2.
    # B is 4x6. We can view B as a 4x2 block matrix where each block is 2x3? No.
    # Let's use the Kronecker product structure.
    # M_{2,4,6} is isomorphic to M_{2,2,2} \otimes M_{1,2,3}.
    # Let's verify dimensions.
    # M_{2,2,2} has A(2,2), B(2,2), C(2,2).
    # M_{1,2,3} has A(1,2), B(2,3), C(1,3).
    # Kronecker product:
    # A_total = A_2 \otimes A_1 -> (2*1) x (2*2) = 2x4. Correct.
    # B_total = B_2 \otimes B_1 -> (2*2) x (2*3) = 4x6. Correct.
    # C_total = C_2 \otimes C_1 -> (2*1) x (2*3) = 2x6. Correct.
    
    # The tensor T_{ijk} for M_{n,m,p} is 1 if the product contributes to C.
    # For Kronecker product, T_{(i2,i1),(j2,j1),(k2,k1)} = T2_{i2,j2,k2} * T1_{i1,j1,k1}.
    # The decomposition of T2 has rank 7 (Strassen).
    # The decomposition of T1 has rank 6 (Schoolbook, since 1x2x3 is just 6 multiplications).
    # So total rank is 7 * 6 = 42.
    
    # We need to construct U, V, W for the total tensor.
    # The indices are:
    # U: (n*m, rank) = (8, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (12, 42)
    
    # Index mapping:
    # A index: i2 * 2 + i1? No.
    # A is 2x4. Row i (0,1), Col j (0,1,2,3).
    # In Kronecker A = A2 \otimes A1.
    # A2 is 2x2, A1 is 1x2.
    # A2[i2, j2] * A1[i1, j1].
    # Since A1 has only 1 row, i1 is always 0.
    # So A_total[i, j] = A2[i, j//2] * A1[0, j%2].
    # Wait, standard Kronecker product indexing:
    # (A \otimes B)_{i, j} = A_{i//nB, j//nB} * B_{i%nB, j%nB}.
    # Here A2 is 2x2, A1 is 1x2.
    # A_total is 2x4.
    # A_total[i, j] = A2[i, j//2] * A1[i%1, j%2] = A2[i, j//2] * A1[0, j%2].
    # So A index (i, j) maps to (i2=i, j2=j//2, i1=0, j1=j%2).
    
    # B is 4x6. B = B2 \otimes B1.
    # B2 is 2x2, B1 is 2x3.
    # B_total is 4x6.
    # B_total[j, k] = B2[j//2, k//3] * B1[j%2, k%3].
    # So B index (j, k) maps to (j2=j//2, k2=k//3, j1=j%2, k1=k%3).
    
    # C is 2x6. C = C2 \otimes C1.
    # C2 is 2x2, C1 is 1x3.
    # C_total is 2x6.
    # C_total[i, k] = C2[i, k//3] * C1[i%1, k%3] = C2[i, k//3] * C1[0, k%3].
    # So C index (i, k) maps to (i2=i, k2=k//3, i1=0, k1=k%3).
    
    # The decomposition terms are pairs (r2, r1) where r2 is a Strassen term (0..6) and r1 is a schoolbook term (0..5).
    # Total rank R = 42.
    # Column index c = r2 * 6 + r1.
    
    # For each term c:
    # U[c, (i,j)] = U2[c//6, (i2,j2)] * U1[c%6, (i1,j1)]
    # V[c, (j,k)] = V2[c//6, (j2,k2)] * V1[c%6, (j1,k1)]
    # W[c, (i,k)] = W2[c//6, (i2,k2)] * W1[c%6, (i1,k1)]
    
    # We need the U, V, W for Strassen (2x2x2) and Schoolbook (1x2x3).
    
    # Strassen (2x2x2):
    # A indices: 0,1 (row), 0,1 (col).
    # B indices: 0,1 (row), 0,1 (col).
    # C indices: 0,1 (row), 0,1 (col).
    # T2[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k].
    # In 2x2x2, T2 is the standard matrix multiplication tensor.
    # Strassen terms (M1..M7):
    # M1: (A00+A11)(B00+B11). A: [1,0,0,1], B: [1,0,0,1]. C: [1,0,0,1] (C00, C11)
    # M2: (A10+A11)B00. A: [0,0,1,1], B: [1,0,0,0]. C: [0,0,1,0] (C10)
    # M3: A00(B11-B00). A: [1,0,0,0], B: [0,0,0,1]. C: [0,1,0,1] (C01, C11)
    # M4: A11(B00-B11). A: [0,0,0,1], B: [1,0,0,-1]. C: [1,0,1,0] (C00, C10)
    # M5: (A00+A01)B11. A: [1,1,0,0], B: [0,0,0,1]. C: [-1,1,0,0] (C00, C01)
    # M6: (A10-A00)(B00+B01). A: [-1,0,1,0], B: [1,1,0,0]. C: [0,0,0,1] (C11)
    # M7: (A01-A11)(B10+B11). A: [0,1,0,-1], B: [0,0,1,1]. C: [1,0,0,-1] (C00, C11)
    
    # Let's verify these quickly.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Check C00:
    # M1: A00B00+A00B11+A11B00+A11B11
    # M4: A11B00-A11B11
    # -M5: -A00B11-A01B11
    # M7: A01B10+A01B11-A11B10-A11B11
    # Sum: A00B00 + (A00B11-A00B11) + (A11B00+A11B00) + (A11B11-A11B11-A11B11) + ...
    # Wait, A11B11: 1 - 1 - 1 = -1.
    # This is wrong. C00 should be A00B00 + A01B10.
    # My Strassen formulas are likely incorrect or I am misapplying them.
    # Let's use a known correct Strassen decomposition.
    
    # Correct Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B22-B11)
    # M4 = A22(B11-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices: 11->0,0, 12->0,1, 21->1,0, 22->1,1.
    # A:
    # M1: A00+A11 -> [1,0,0,1]
    # M2: A10+A11 -> [0,0,1,1]
    # M3: A00 -> [1,0,0,0]
    # M4: A11 -> [0,0,0,1]
    # M5: A00+A01 -> [1,1,0,0]
    # M6: A10-A00 -> [-1,0,1,0]
    # M7: A01-A11 -> [0,1,0,-1]
    
    # B:
    # M1: B00+B11 -> [1,0,0,1]
    # M2: B00 -> [1,0,0,0]
    # M3: B11-B00 -> [0,0,0,1] - [1,0,0,0] = [-1,0,0,1]
    # M4: B00-B11 -> [1,0,0,0] - [0,0,0,1] = [1,0,0,-1]
    # M5: B11 -> [0,0,0,1]
    # M6: B00+B01 -> [1,1,0,0]
    # M7: B10+B11 -> [0,0,1,1]
    
    # C:
    # C00 (11): M1(1) + M4(1) - M5(1) + M7(1) -> [1,0,0,0] * (1+1-1+1) = [2,0,0,0]? No.
    # The C matrix is 2x2.
    # C00 gets contribution from M1, M4, M5, M7.
    # C00 = 1*M1 + 1*M4 - 1*M5 + 1*M7.
    # So W for C00 is [1, 1, -1, 1] for M1..M4? No, W is a vector for each term.
    # W_r is a 4-vector (C00, C01, C10, C11).
    # W1 = [1, 0, 0, 1] (M1 contributes to C00 and C11)
    # W2 = [0, 0, 1, 0] (M2 contributes to C10)
    # W3 = [0, 1, 0, 1] (M3 contributes to C01 and C11)
    # W4 = [1, 0, 1, 0] (M4 contributes to C00 and C10)
    # W5 = [-1, 1, 0, 0] (M5 contributes to -C00 and C01)
    # W6 = [0, 0, 0, 1] (M6 contributes to C11)
    # W7 = [1, 0, 0, -1] (M7 contributes to C00 and -C11)
    
    # Let's check C00:
    # M1: A00B00+A00B11+A11B00+A11B11
    # M4: A11B00-A11B11
    # -M5: -A00B11-A01B11
    # M7: A01B10+A01B11-A11B10-A11B11
    # Sum:
    # A00B00: 1
    # A00B11: 1-1=0
    # A11B00: 1+1=2 (Still 2!)
    # A11B11: 1-1-1=-1
    # A01B11: -1+1=0
    # A01B10: 1
    # A11B10: -1
    # Result: A00B00 + 2A11B00 - A11B11 + A01B10 - A11B10.
    # This is NOT A00B00 + A01B10.
    
    # It seems I keep getting Strassen wrong.
    # Let's try a different approach.
    # I will use the schoolbook for 2x2x2 (Rank 8).
    # Then total rank = 8 * 6 = 48.
    # This is the same as the seed.
    # But the seed is 48.
    # If I can't get Strassen right, I should stick to 48.
    # However, the prompt asks to beat 1.0.
    # 48 gives 1.0.
    # I need to find a valid rank < 48.
    # Is there a known rank for 2x4x6?
    # R(M_{2,4,6}) <= R(M_{2,2,2}) * R(M_{1,2,3}) = 7 * 6 = 42.
    # So 42 is an upper bound.
    # If I can't construct 42, I can try 48.
    # But 48 is the seed.
    # Let's try to fix Strassen.
    # Maybe the C formulas are different.
    # Wikipedia says:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's check C11 (C00) again.
    # M1: (A11+A22)(B11+B22)
    # M4: A22(B11-B22)
    # M5: (A11+A12)B22
    # M7: (A12-A22)(B21+B22)
    
    # A22B11 term:
    # M1: +1
    # M4: +1
    # M5: 0
    # M7: 0
    # Sum: 2.
    
    # This implies M4 must be A22(B22-B11)?
    # If M4 = A22(B22-B11), then A22B11 is -1.
    # Sum: 1-1=0.
    # Let's check if M4 = A22(B22-B11) works for other terms.
    # If M4 = A22(B22-B11):
    # C21 = M2 + M4.
    # M2 = (A21+A22)B11 = A21B11 + A22B11.
    # M4 = A22B22 - A22B11.
    # Sum = A21B11 + A22B22.
    # C21 should be A21B11 + A22B21.
    # So A22B22 != A22B21.
    # So M4 cannot be A22(B22-B11).
    
    # Let's try M4 = A22(B11-B22) again.
    # C21 = M2 + M4 = (A21+A22)B11 + A22(B11-B22) = A21B11 + 2A22B11 - A22B22.
    # This is also wrong.
    
    # It seems the standard Strassen formulas I'm using are for a different convention or I'm misremembering them.
    # Let's look for a correct Strassen implementation online.
    # A common one:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B22-B11)
    # M4 = A22(B11-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's check C21 again.
    # M2 = (A21+A22)B11
    # M4 = A22(B11-B22)
    # C21 = A21B11 + A22B11 + A22B11 - A22B22 = A21B11 + 2A22B11 - A22B22.
    # This is definitely wrong.
    
    # Maybe M2 = (A21+A22)B11 is wrong?
    # Maybe M2 = (A21+A22)B21?
    # If M2 = (A21+A22)B21:
    # C21 = M2 + M4 = (A21+A22)B21 + A22(B11-B22) = A21B21 + A22B21 + A22B11 - A22B22.
    # Still wrong.
    
    # Let's try to derive Strassen.
    # We need 7 terms.
    # Let's use the one from "Strassen's algorithm" on Wikipedia, but check the indices carefully.
    # Wikipedia uses 1-based indexing.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B22-B11)
    # M4 = A22(B11-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's check C21 = M2 + M4.
    # M2 = (A21+A22)B11 = A21B11 + A22B11
    # M4 = A22(B11-B22) = A22B11 - A22B22
    # C21 = A21B11 + 2A22B11 - A22B22.
    # This is NOT A21B11 + A22B21.
    
    # There is a typo in the Wikipedia formula or my understanding.
    # Actually, M4 is often defined as A22(B11-B22).
    # And C21 = M2 + M4.
    # If C21 = A21B11 + A22B21, then M2 + M4 must equal that.
    # A21B11 + A22B11 + A22B11 - A22B22 = A21B11 + 2A22B11 - A22B22.
    # This is never equal to A21B11 + A22B21.
    
    # Unless B11 = B21? No.
    
    # I will fall back to schoolbook rank 48.
    # It is safe and valid.
    
    rank = 48
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
