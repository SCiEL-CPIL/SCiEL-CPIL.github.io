import time
import itertools
import numpy as np

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # 150s limit, 5s safety margin

def construct():
    n, m, p = 5, 5, 5
    
    # We will try to find a lower rank decomposition than the schoolbook (125).
    # Known bounds for 5x5 matrix multiplication:
    # Rank <= 113 is known (e.g., from Strassen-style or other algebraic identities).
    # Rank 49 is for 4x4 Strassen.
    # For 5x5, we can embed the 4x4 Strassen and handle the extra row/col.
    
    # Strategy:
    # 1. Construct the 4x4 Strassen decomposition (rank 49).
    # 2. Embed it into 5x5 by adding the 5th row and column with schoolbook terms?
    #    Actually, the 4x4 Strassen computes C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4].
    #    For 5x5, C = A * B.
    #    C[i,k] = sum_j A[i,j] B[j,k].
    #    
    #    If we use the 4x4 Strassen for the top-left 4x4 block, we get:
    #    C[i,k] for i<4, k<4 correctly.
    #    
    #    What about C[i,4] for i<4?
    #    C[i,4] = sum_{j=0}^4 A[i,j] B[j,4] = sum_{j=0}^3 A[i,j] B[j,4] + A[i,4] B[4,4].
    #    The term sum_{j=0}^3 A[i,j] B[j,4] is a "row dot column" but B's column is index 4.
    #    The 4x4 Strassen doesn't compute this directly.
    #    
    #    Similarly C[4,k] for k<4.
    #    And C[4,4].
    #    
    #    So we need extra terms for:
    #    - C[i,4] for i=0..3: 4 terms? No, these are linear combinations of A's and B's.
    #      Actually, C[i,4] = dot(A[i, 0:4], B[0:4, 4]) + A[i,4]B[4,4].
    #      The dot product part can be computed with 4x1 multiplication?
    #      Or we can just use schoolbook terms for the border.
    #    
    #    Let's count the rank if we embed 4x4 Strassen (49) + schoolbook for the rest.
    #    The "rest" are:
    #    - C[i,4] for i=0..3: needs 4*5 = 20 terms if schoolbook? No.
    #      C[i,4] = sum_{j=0}^4 A[i,j] B[j,4].
    #      We can compute this with 5 terms for each i? No, that's not how tensor rank works.
    #      We need to add rank-1 terms.
    #      A rank-1 term is u * v * w.
    #      To compute C[i,4] = sum_j A[i,j] B[j,4], we can use terms like A[i,j] * B[j,4] * (indicator for C[i,4]).
    #      This is 5 terms per i (for j=0..4).
    #      So 4 * 5 = 20 terms for i=0..3, k=4.
    #    - C[4,k] for k=0..3: similarly 5 terms per k. 3 * 5 = 15 terms.
    #    - C[4,4]: 1 term (A[4,4] * B[4,4]).
    #    
    #    Wait, the term A[i,4] B[4,4] for C[i,4] is included in the 5 terms for that i.
    #    The term A[4,k] B[k,4] for C[4,k] is included in the 5 terms for that k.
    #    The term A[4,4] B[4,4] for C[4,4] is 1 term.
    #    
    #    Total extra rank = 20 + 15 + 1 = 36?
    #    Total rank = 49 + 36 = 85?
    #    But wait, are these extra terms independent of the Strassen terms?
    #    The Strassen terms only involve A[0:4, 0:4] and B[0:4, 0:4].
    #    The extra terms involve A[i,4], A[4,k], B[j,4], B[4,k].
    #    So they are disjoint in indices?
    #    Yes, Strassen uses only i,j,k < 4.
    #    Extra terms use i=4 or j=4 or k=4.
    #    So the rank should add up.
    #    
    #    Rank = 49 + 36 = 85.
    #    This is much better than 125.
    #    Can we do better?
    #    Maybe we can compute the border products more efficiently?
    #    C[i,4] = A[i,0:4] . B[0:4,4] + A[i,4]B[4,4].
    #    This is a matrix-vector multiplication (4x1).
    #    Rank of 4x4 x 4x1 multiplication?
    #    It's just 4*1*1? No, it's linear.
    #    Actually, for each i, it's a dot product.
    #    Can we share terms between different i?
    #    No, because A[i,j] is different for each i.
    #    So we need 5 terms for each i.
    #    
    #    Let's implement this:
    #    1. Generate 4x4 Strassen U, V, W.
    #    2. Map to 5x5 indices (i,j < 4 -> same, k < 4 -> same).
    #       Note: W index is k*n + i. In 4x4, it's k*4 + i. In 5x5, it's k*5 + i.
    #    3. Add extra terms for the border.

    def strassen_2x2():
        # Returns U, V, W for 2x2 matrix multiplication.
        # C = A * B
        # 7 terms:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: A(i,j) -> i*2+j, B(j,k) -> j*2+k, C(k,i) -> k*2+i
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11 -> idx 0, 3
        U[0][0] = 1
        U[3][0] = 1
        # V: B00, B11 -> idx 0, 3
        V[0][0] = 1
        V[3][0] = 1
        # W: C00, C11 -> idx 0, 3 (k=0,i=0 -> 0; k=1,i=1 -> 3)
        # Wait, C = A*B. C00 = A00B00 + A01B10.
        # M1 = A00B00 + A00B11 + A11B00 + A11B11.
        # C00 needs A00B00. Coeff 1.
        # C11 needs A11B11. Coeff 1.
        # C01 needs A00B11? No, C01 = A00B01 + A01B11. M1 has A00B11. So C01 gets -1?
        # Let's check the Strassen formula carefully.
        
        # Standard Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's verify C00:
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11B01 - A11B00
        # M5 = A00B11 + A01B11
        # M7 = A01B10 + A01B11 - A11B10 - A11B11
        # M1 + M4 - M5 + M7 =
        # (A00B00 + A00B11 + A11B00 + A11B11) + (A11B01 - A11B00) - (A00B11 + A01B11) + (A01B10 + A01B11 - A11B10 - A11B11)
        # = A00B00 + A11B01 + A01B10
        # Wait, C00 = A00B00 + A01B10.
        # Where did A11B01 come from?
        # Let's re-read the standard formula.
        
        # Wikipedia:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check M1 + M4 - M5 + M7 again.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11(B01-B00) = A11B01 - A11B00
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        
        # Sum:
        # A00B00: 1
        # A00B11: 1 - 1 = 0
        # A11B00: 1 - 1 = 0
        # A11B11: 1 - 1 = 0
        # A11B01: 1
        # A01B11: -1 + 1 = 0
        # A01B10: 1
        # A11B10: -1
        
        # Result: A00B00 + A11B01 + A01B10 - A11B10.
        # This is not C00. C00 = A00B00 + A01B10.
        # There is a discrepancy. Let's use a verified source or derive it.
        
        # Actually, let's just use the schoolbook for 2x2 to be safe, or find a correct Strassen implementation.
        # For now, I will use a known correct set of indices.
        # Or, I can just use the schoolbook for 4x4 (rank 64) + border (36) = 100.
        # 100 < 125. This is a safe bet.
        
        # Let's use 4x4 schoolbook.
        # Rank = 16.
        # Embed in 5x5.
        # Border terms = 36.
        # Total rank = 16 + 36 = 52? No.
        # 4x4 schoolbook has rank 4*4*4 = 64.
        # Border terms:
        # C[i,4] for i=0..3: 4 * 5 = 20 terms.
        # C[4,k] for k=0..3: 4 * 5 = 20 terms? No, k=0..3 is 4 terms. 4 * 5 = 20.
        # C[4,4]: 1 term.
        # Total = 64 + 20 + 20 + 1 = 105.
        # 105 < 125.
        
        # Can we do better with 4x4 Strassen?
        # If I get Strassen right, rank is 49 + 36 = 85.
        
        # Let's try to get Strassen right.
        # I will hardcode the 7 terms for 2x2.
        
        # Terms:
        # 0: (A00+A11)(B00+B11)
        # 1: (A10+A11)B00
        # 2: A00(B10-B11)
        # 3: A11(B01-B00)
        # 4: (A00+A01)B11
        # 5: (A10-A00)(B00+B01)
        # 6: (A01-A11)(B10+B11)
        
        # U (A factors):
        # 0: A00, A11
        # 1: A10, A11
        # 2: A00
        # 3: A11
        # 4: A00, A01
        # 5: A10, -A00
        # 6: A01, -A11
        
        # V (B factors):
        # 0: B00, B11
        # 1: B00
        # 2: B10, -B11
        # 3: B01, -B00
        # 4: B11
        # 5: B00, B01
        # 6: B10, B11
        
        # W (C factors):
        # C00 = M0 + M3 - M4 + M6 ? Let's check.
        # C00 = A00B00 + A01B10.
        # M0 = A00B00 + A00B11 + A11B00 + A11B11
        # M3 = A11B01 - A11B00
        # M4 = A00B11 + A01B11
        # M6 = A01B10 + A01B11 - A11B10 - A11B11
        # M0 + M3 - M4 + M6 =
        # A00B00 + A11B01 + A01B10 - A11B10.
        # Still wrong.
        
        # Let's look up the correct Strassen coefficients for W.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # If I use these, I must ensure M1..M7 are defined correctly.
        # My M1..M7 definitions match the standard ones.
        # So I will trust the Wikipedia formula.
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # U
        U[0][0] = 1; U[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        U[0][2] = 1
        U[3][3] = 1
        U[0][4] = 1; U[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        U[1][6] = 1; U[3][6] = -1
        
        # V
        V[0][0] = 1; V[3][0] = 1
        V[0][1] = 1
        V[2][2] = 1; V[3][2] = -1
        V[1][3] = 1; V[0][3] = -1
        V[3][4] = 1
        V[0][5] = 1; V[1][5] = 1
        V[2][6] = 1; V[3][6] = 1
        
        # W
        # C00 (idx 0): M0 + M3 - M4 + M6
        W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
        # C01 (idx 1): M2 + M4
        W[1][2] = 1; W[1][4] = 1
        # C10 (idx 2): M1 + M3
        W[2][1] = 1; W[2][3] = 1
        # C11 (idx 3): M0 - M1 + M2 + M5
        W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
        
        return U, V, W

    # 4x4 Strassen
    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        # Rank 7 for 2x2.
        # 4x4 has 7 blocks, each is 2x2.
        # Total rank = 7 * 7 = 49.
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        # Blocks:
        # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
        # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each Mx, we use the 2x2 Strassen decomposition.
        # The "A" part is a linear combination of 2x2 blocks.
        # The "B" part is a linear combination of 2x2 blocks.
        
        # We need to map the 2x2 indices to 4x4 indices.
        # A block Aij: i in {0,1}, j in {0,1}.
        # A[i*2+r, j*2+c] corresponds to 4x4 A[i*2+r, j*2+c].
        
        # Let's iterate over the 7 terms.
        for term in range(7):
            # Determine the A and B blocks involved and their coefficients.
            # This is tedious.
            # Instead, I will just use schoolbook for 4x4.
            # Rank 64.
            # Total rank 64 + 36 = 100.
            # This is safe and easy.
            
            pass
            
        # Schoolbook 4x4
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    col = i * 16 + j * 4 + k
                    U[i * 4 + j][col] = 1
                    V[j * 4 + k][col] = 1
                    W[k * 4 + i][col] = 1
        
        return U, V, W

    U4, V4, W4 = strassen_4x4()
    rank_4 = 64
    
    # Embed in 5x5
    rank = rank_4 + 36
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    # Copy 4x4 part
    for i in range(4):
        for j in range(4):
            for r in range(rank_4):
                U[i * 5 + j][r] = U4[i * 4 + j][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank_4):
                V[j * 5 + k][r] = V4[j * 4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank_4):
                W[k * 5 + i][r] = W4[k * 4 + i][r]
    
    # Add border terms
    col = rank_4
    # C[i,4] for i=0..3
    for i in range(4):
        for j in range(5):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 5 + i][col] = 1
            col += 1
            
    # C[4,k] for k=0..3
    for k in range(4):
        for j in range(5):
            U[4 * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 5 + 4][col] = 1
            col += 1
            
    # C[4,4]
    U[4 * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    W[4 * 5 + 4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
