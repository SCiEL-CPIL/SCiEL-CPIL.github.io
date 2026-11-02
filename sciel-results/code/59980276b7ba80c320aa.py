import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    n, m, p = 2, 2, 10
    
    # Standard Strassen rank-7 decomposition for 2x2 matrix multiplication
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # Indices for A: [0: A11, 1: A12, 2: A21, 3: A22]
    # Indices for B: [0: B11, 1: B12, 2: B21, 3: B22]
    # Indices for C: [0: C11, 1: C12, 2: C21, 3: C22]
    
    # Standard Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M7 - M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M2
    
    # Each term is (a_vec, b_vec, c_vec)
    # a_vec: coefficients for A11, A12, A21, A22
    # b_vec: coefficients for B11, B12, B21, B22
    # c_vec: coefficients for C11, C12, C21, C22 (signs included)
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [-1, 0, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]), # M7
    ]
    
    # Let's verify this decomposition locally with a simple test
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = AB = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    
    A_test = [1, 2, 3, 4]
    B_test = [5, 6, 7, 8]
    
    C_computed = [0, 0, 0, 0]
    for a_vec, b_vec, c_vec in strassen_terms:
        # Compute scalar product for A part
        a_dot = sum(a_vec[i] * A_test[i] for i in range(4))
        # Compute scalar product for B part
        b_dot = sum(b_vec[i] * B_test[i] for i in range(4))
        # Add to C
        for i in range(4):
            C_computed[i] += a_dot * b_dot * c_vec[i]
            
    # Expected C: [19, 22, 43, 50]
    if C_computed != [19, 22, 43, 50]:
        # If it fails, it means my Strassen formulas or indexing is wrong.
        # Let's fall back to schoolbook rank 8 to ensure valid score.
        # But first, let's try to debug quickly.
        # M1 = (1+4)(5+8) = 65. C11 += 65, C22 += 65
        # M2 = (3+4)5 = 35. C21 += 35
        # M3 = 1(7-8) = -1. C12 += -1
        # M4 = 4(6-5) = 4. C11 -= 4, C21 += 4
        # M5 = (1+2)8 = 24. C12 += 24, C22 -= 24
        # M6 = (3-1)(5+6) = 22. C11 -= 22
        # M7 = (2-4)(7+8) = -30. C11 += -30, C22 -= -30 => C22 += 30
        
        # C11 = 65 - 4 - 22 - 30 = 9. Expected 19. Mismatch.
        # C12 = -1 + 24 = 23. Expected 22. Mismatch.
        # C21 = 35 + 4 = 39. Expected 43. Mismatch.
        # C22 = 65 - (-1) - 24 + 30 = 72. Expected 50. Mismatch.
        
        # The standard Strassen formulas I used above are for C = B*A? Or different indexing?
        # Let's check Wikipedia "Strassen algorithm" again.
        # M1=(A11+A22)(B11+B22)
        # M2=(A21+A22)B11
        # M3=A11(B21−B22)
        # M4=A22(B12−B11)
        # M5=(A11+A12)B22
        # M6=(A21−A11)(B11+B12)
        # M7=(A12−A22)(B21+B22)
        # C11=M1−M4+M7−M6
        # C12=M3+M5
        # C21=M2+M4
        # C22=M1−M3+M5−M2
        
        # My implementation matches this. Why did it fail?
        # Let's re-calculate manually carefully.
        # A11=1, A12=2, A21=3, A22=4
        # B11=5, B12=6, B21=7, B22=8
        
        # M1 = (1+4)(5+8) = 65
        # M2 = (3+4)5 = 35
        # M3 = 1(7-8) = -1
        # M4 = 4(6-5) = 4
        # M5 = (1+2)8 = 24
        # M6 = (3-1)(5+6) = 2*11 = 22
        # M7 = (2-4)(7+8) = -2*15 = -30
        
        # C11 = 65 - 4 + (-30) - 22 = 65 - 56 = 9. (Expected 19)
        # C12 = -1 + 24 = 23. (Expected 22)
        # C21 = 35 + 4 = 39. (Expected 43)
        # C22 = 65 - (-1) + 24 - 35 = 65 + 1 + 24 - 35 = 55. (Expected 50)
        
        # There is a systematic error.
        # C11 = A11B11 + A12B21 = 1*5 + 2*7 = 19.
        # C12 = A11B12 + A12B22 = 1*6 + 2*8 = 22.
        # C21 = A21B11 + A22B21 = 3*5 + 4*7 = 43.
        # C22 = A21B12 + A22B22 = 3*6 + 4*8 = 50.
        
        # Let's check the algebra for C11 from Strassen:
        # M1-M4+M7-M6
        # (A11+A22)(B11+B22) - A22(B12-B11) + (A12-A22)(B21+B22) - (A21-A11)(B11+B12)
        # = A11B11 + A11B22 + A22B11 + A22B22 - A22B12 + A22B11 + A12B21 + A12B22 - A22B21 - A22B22 - A21B11 - A21B12 + A11B11 + A11B12
        # Group by A_i B_j:
        # A11B11: 1 + 1 = 2. (We want 1)
        # A11B12: 1. (We want 0)
        # A11B22: 1. (We want 0)
        # A22B11: 1 + 1 = 2. (We want 0)
        # A22B22: 1 - 1 = 0. (We want 0)
        # A22B12: -1. (We want 0)
        # A12B21: 1. (We want 0)
        # A12B22: 1. (We want 0)
        # A22B21: -1. (We want 0)
        # A21B11: -1. (We want 0)
        # A21B12: -1. (We want 0)
        
        # The sum is NOT A11B11 + A12B21.
        # The standard Strassen formulas are for C = A * B where:
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        
        # Wait, I might have copied the formulas wrong or they are for a different convention.
        # Let's look at a verified source.
        # C11 = M1 + M7 - M5 - M6 ?
        # C12 = M4 + M5 ?
        # C21 = M2 + M3 ?
        # C22 = M1 + M4 - M3 - M7 ?
        
        # Let's try this set:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # This is the same set of Ms. The combinations for C must be wrong.
        
        # Let's derive C11 = A11B11 + A12B21.
        # We have 7 Ms.
        # This is a linear algebra problem.
        
        # Alternatively, just use Schoolbook Rank 8. It's safe.
        # The goal is to get a valid score. Rank 40 (5 blocks of 8) is valid.
        # Rank 35 (5 blocks of 7) is better if correct.
        
        # Let's try to find the correct combination for C11.
        # Target: A11B11 + A12B21
        
        # Let's check the product of the vectors.
        # This is getting complex. I will implement Schoolbook to guarantee a valid submission.
        # Then I can try to optimize if time permits, but safety first.
        
        # Schoolbook terms for 2x2:
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        
        # We can decompose this into 8 rank-1 terms?
        # Actually, schoolbook is 8 multiplications? No, 2x2 is 4 multiplications.
        # Wait, rank of 2x2 matrix multiplication tensor is 7.
        # Schoolbook uses 4 scalar multiplications, but as a bilinear form over the field?
        # No, the tensor T = sum_{i,j,k} T_{ijk} e_i e_j e_k.
        # T_{ijk} = 1 if k = i*m + j (for C = A*B, C_{ij} = sum_k A_{ik} B_{kj}).
        # Here m=2.
        # C_{00} = A_{00}B_{00} + A_{01}B_{10}. Indices: A[0], A[1], B[0], B[1].
        # C_{01} = A_{00}B_{01} + A_{01}B_{11}. Indices: A[0], A[1], B[2], B[3].
        # C_{10} = A_{10}B_{00} + A_{11}B_{10}. Indices: A[2], A[3], B[0], B[1].
        # C_{11} = A_{10}B_{01} + A_{11}B_{11}. Indices: A[2], A[3], B[2], B[3].
        
        # Schoolbook decomposition:
        # Term 1: A11 B11 -> C11. u=[1,0,0,0], v=[1,0,0,0], w=[1,0,0,0]
        # Term 2: A12 B21 -> C11. u=[0,1,0,0], v=[0,0,1,0], w=[1,0,0,0]
        # Term 3: A11 B12 -> C12. u=[1,0,0,0], v=[0,1,0,0], w=[0,1,0,0]
        # Term 4: A12 B22 -> C12. u=[0,1,0,0], v=[0,0,0,1], w=[0,1,0,0]
        # Term 5: A21 B11 -> C21. u=[0,0,1,0], v=[1,0,0,0], w=[0,0,1,0]
        # Term 6: A22 B21 -> C21. u=[0,0,0,1], v=[0,0,1,0], w=[0,0,1,0]
        # Term 7: A21 B12 -> C22. u=[0,0,1,0], v=[0,1,0,0], w=[0,0,0,1]
        # Term 8: A22 B22 -> C22. u=[0,0,0,1], v=[0,0,0,1], w=[0,0,0,1]
        
        # This is rank 8.
        
        # I will use this schoolbook set to ensure validity.
        # If I have time, I will try to fix Strassen.
        
        schoolbook_terms = [
            ([1,0,0,0], [1,0,0,0], [1,0,0,0]),
            ([0,1,0,0], [0,0,1,0], [1,0,0,0]),
            ([1,0,0,0], [0,1,0,0], [0,1,0,0]),
            ([0,1,0,0], [0,0,0,1], [0,1,0,0]),
            ([0,0,1,0], [1,0,0,0], [0,0,1,0]),
            ([0,0,0,1], [0,0,1,0], [0,0,1,0]),
            ([0,0,1,0], [0,1,0,0], [0,0,0,1]),
            ([0,0,0,1], [0,0,0,1], [0,0,0,1]),
        ]
        
        # Use schoolbook
        terms_to_use = schoolbook_terms
    else:
        terms_to_use = strassen_terms

    rank = len(terms_to_use) * (p // 2)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in terms_to_use:
            # U: A is 2x2, indices [i*m+j]
            for idx in range(4):
                U[idx][col] = a_vec[idx]
            
            # V: B is 2x10. Block k covers columns 2k, 2k+1.
            # B[0, 2k] is local B11. Index 0*10 + 2k.
            # B[0, 2k+1] is local B12. Index 0*10 + 2k+1.
            # B[1, 2k] is local B21. Index 1*10 + 2k.
            # B[1, 2k+1] is local B22. Index 1*10 + 2k+1.
            V[0 * p + 2 * k][col] = b_vec[0]
            V[0 * p + 2 * k + 1][col] = b_vec[1]
            V[1 * p + 2 * k][col] = b_vec[2]
            V[1 * p + 2 * k + 1][col] = b_vec[3]
            
            # W: C is 2x10. Convention W[k'*n + i].
            # C[0, 2k] is local C11. Index 2k*2 + 0 = 4k.
            # C[0, 2k+1] is local C12. Index (2k+1)*2 + 0 = 4k+2.
            # C[1, 2k] is local C21. Index 2k*2 + 1 = 4k+1.
            # C[1, 2k+1] is local C22. Index (2k+1)*2 + 1 = 4k+3.
            
            # c_vec is [C11, C12, C21, C22]
            W[4 * k][col] = c_vec[0]       # C11
            W[4 * k + 2][col] = c_vec[1]   # C12
            W[4 * k + 1][col] = c_vec[2]   # C21
            W[4 * k + 3][col] = c_vec[3]   # C22
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
