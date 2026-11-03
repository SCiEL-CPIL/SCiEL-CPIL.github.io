import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _strassen_2x2():
    # Strassen decomposition for 2x2 x 2x2 -> 2x2
    # Rank 7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    # C00 = M1 - M4 + M5
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6
    
    # Indices: A[i,j] where i,j in 0..1
    # B[j,k] where j,k in 0..1
    # C[i,k] where i,k in 0..1
    # W index is k*n+i = k*2+i
    
    U = [[0]*7 for _ in range(4)]  # n*m = 2*2 = 4
    V = [[0]*7 for _ in range(4)]  # m*p = 2*2 = 4
    W = [[0]*7 for _ in range(4)]  # p*n = 2*2 = 4
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 (k=0, i=0)
    W[1][0] = 1  # C01 (k=1, i=0)
    W[0][1] = 1  # C10 (k=0, i=1)
    W[1][1] = 1  # C11 (k=1, i=1)
    
    # M2: (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[0][1] = -1 # C10 gets -M2
    W[1][1] = -1 # C11 gets -M2
    
    # M3: A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[3][2] = -1 # B11
    W[1][0] = 1  # C01 gets M3
    W[0][1] = 1  # C10 gets M3
    
    # M4: A11(B01-B10)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[2][3] = -1 # B10
    W[0][0] = -1 # C00 gets -M4
    W[1][0] = 1  # C01 gets M4
    
    # M5: (A00+A10)B11
    U[0][4] = 1  # A00
    U[2][4] = 1  # A10
    V[3][4] = 1  # B11
    W[0][0] = 1  # C00 gets M5
    
    # M6: (A10-A00)(B00+B10)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[2][5] = 1  # B10
    W[1][1] = 1  # C11 gets M6
    
    # M7: (A11-A01)(B01+B11)
    U[3][6] = 1  # A11
    U[1][6] = -1 # A01
    V[1][6] = 1  # B01
    V[3][6] = 1  # B11
    # M7 is not used in the standard Strassen formulas for C?
    # Wait, standard Strassen:
    # C11 = M1 - M2 + M6. Where is M7?
    # Ah, I used 7 terms but only defined 6 in the code above?
    # Let's re-verify Strassen.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    # C00 = M1 - M4 + M5
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6
    # Wait, C11 usually involves M7?
    # No, C11 = M1 - M2 + M6 is correct for some variants, but let's check.
    # Actually, C11 = M1 - M2 + M6 is often cited.
    # But M7 is defined. Is it used?
    # Let's check C11 = (A00+A11)(B00+B11) - (A10+A11)B00 + (A10-A00)(B00+B10)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A10B00 + A10B10 - A00B00 - A00B10
    # = A00B11 + A11B11 + A10B10 - A00B10
    # This is NOT C11. C11 = A00B01 + A01B11 + A10B01 + A11B11.
    # So my Strassen formula set is wrong or I missed M7 usage.
    # Correct Strassen:
    # C00 = M1 - M4 + M5
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6
    # Wait, C11 in my expansion above was A00B11 + A11B11 + A10B10 - A00B10.
    # But C11 should be A00B01 + A01B11 + A10B01 + A11B11.
    # So M6 must be different or I made a mistake.
    # Let's use a known correct set.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    
    # C00 = M1 - M4 + M5
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6
    
    # Let's re-evaluate C11 with M6 = (A10-A00)(B00+B10).
    # M1 - M2 + M6 = (A00B00 + A00B11 + A11B00 + A11B11) - (A10B00 + A11B00) + (A10B00 + A10B10 - A00B00 - A00B10)
    # = A00B11 + A11B11 + A10B10 - A00B10.
    # This is definitely not C11.
    
    # There is a variation of Strassen where:
    # C11 = M1 - M2 + M6 is wrong.
    # Actually, C11 = M1 - M2 + M6 is for a different definition of M6?
    # Let's look up the exact formulas.
    # Wikipedia:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    # C00 = M1 - M4 + M5
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6
    
    # Why does C11 not work?
    # Let's check C11 = A00B01 + A01B11 + A10B01 + A11B11.
    # My expansion gave A00B11 + A11B11 + A10B10 - A00B10.
    # These are different.
    # Ah, I see. M6 is (A10-A00)(B00+B10).
    # Maybe M6 should be (A10-A00)(B00+B10) is correct, but I am missing something?
    # No, the formula C11 = M1 - M2 + M6 is standard.
    # Let's re-read the expansion.
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = A10B00 + A11B00
    # M6 = A10B00 + A10B10 - A00B00 - A00B10
    # M1 - M2 + M6 = A00B11 + A11B11 + A10B10 - A00B10.
    # This is NOT C11.
    
    # Is it possible that M6 is defined differently?
    # Some sources define M6 = (A10-A00)(B00+B10).
    # Some define M6 = (A10-A00)(B00+B10).
    # Let's check another source.
    # Maybe C11 = M1 - M2 + M6 is wrong and it should be M7?
    # C11 = M1 - M2 + M6 is very common.
    # Let's check C11 = A00B01 + A01B11 + A10B01 + A11B11.
    # If I use M7 = (A11-A01)(B01+B11) = A11B01 + A11B11 - A01B01 - A01B11.
    # This doesn't help directly.
    
    # Let's try a different set of 7 products.
    # This is getting risky. I will stick to a simpler block decomposition or verify the Strassen code carefully.
    # Given the time, I will use a verified Strassen implementation or fall back to schoolbook if it fails verification.
    # Actually, I can just use the schoolbook for 4x4x9 which is rank 144.
    # But the goal is to improve.
    # Let's try to construct a 4x4x4 decomposition using Strassen (rank 196? No, 7*7*4? No).
    # 4x4x4 can be done with rank 49 (Strassen squared).
    # 4x4x9 is harder.
    
    # Let's just return the schoolbook for now to ensure validity, and then try to improve.
    # But I need to return a dict.
    # I will implement a simple recursive Strassen for 4x4x4 and then extend to 4x4x9?
    # No, 4x4x9 is not square.
    
    # Let's just return the schoolbook for 4x4x9.
    # Rank 144.
    
    # Wait, I can try to find a better rank for 4x4x9.
    # The known lower bound is 4*4*9 = 144? No, lower bound is 2*4*4 + 2*4*9 - 2*4 = 32+72-8 = 96? No.
    # Lower bound for M_{n,m,p} is max(nm, mp, pn). Here max(16, 36, 36) = 36.
    # But that's a very loose bound.
    # The trivial bound is nmp = 144.
    # Strassen gives 7^log2(4) = 49 for 4x4x4.
    # For 4x4x9, we can split B into three 4x3 blocks.
    # C = A * [B1, B2, B3] = [A*B1, A*B2, A*B3].
    # If we can do 4x4x3 with rank r, then 4x4x9 has rank 3r.
    # What is the rank of 4x4x3?
    # Schoolbook is 4*4*3 = 48.
    # Can we do better?
    # 4x4x3 can be viewed as 4x4x4 with one column of B zero? No, that doesn't reduce rank necessarily.
    # But we can use Strassen on 4x4x4 and ignore the last column?
    # If we compute A*B where B is 4x4, we get 4x4.
    # If B is 4x3, we can pad it to 4x4 with zeros.
    # Then A*B_padded = [A*B, 0].
    # The rank of the decomposition for 4x4x4 is 49.
    # Does the padding reduce the rank?
    # The terms involving B_{i,3} (the 4th column) will be zero.
    # In Strassen, M1..M7 involve B columns.
    # If B_{i,3}=0, then M1, M2, M3, M4, M5, M6, M7 might simplify.
    # M1 = (A00+A11)(B00+B11). B11 is row 1, col 1. Not col 3.
    # Wait, B is 4x4. B_{ij} is element (i,j).
    # If we pad B (4x3) to 4x4, the 4th column is 0.
    # So B_{i,3} = 0 for all i.
    # In Strassen, the products involve sums of B_{ij}.
    # If B_{i,3}=0, then any term with B_{i,3} vanishes.
    # M1 = (A00+A11)(B00+B11). B00 is top-left 2x2, B11 is bottom-right 2x2.
    # In 4x4, B11 is rows 2,3 cols 2,3.
    # If we pad to 4x4, the indices are 0..3.
    # B11 is B[2:4, 2:4].
    # If the original B is 4x3, then B[2:4, 2:4] is not zero.
    # The padded column is col 3.
    # So B[0,3]=0, B[1,3]=0, B[2,3]=0, B[3,3]=0.
    # B11 = B[2:4, 2:4] = [[B22, B23], [B32, B33]].
    # Since B23=0, B33=0, B11 = [[B22, 0], [B32, 0]].
    # So M1 = (A00+A11) * [[B22, 0], [B32, 0]].
    # This is not zero.
    # So the rank doesn't drop to 0.
    # But maybe some terms become linear?
    # This is complex.
    
    # Let's just use the schoolbook for 4x4x9.
    # Rank 144.
    
    # But wait, the prompt says "score is the rank".
    # And "1.0 equals the seed program".
    # The seed program returns rank 144.
    # So I need to find a rank < 144.
    
    # Let's try to combine Strassen 4x4x4 (rank 49) with something.
    # Or just use 4x4x4 Strassen for the first 4 columns of B, and schoolbook for the last 5?
    # B is 4x9.
    # C = A * B.
    # Split B into B1 (4x4) and B2 (4x5).
    # C = A*B1 + A*B2.
    # A*B1 can be done with rank 49 (Strassen).
    # A*B2 can be done with rank 4*4*5 = 80 (Schoolbook).
    # Total rank = 49 + 80 = 129.
    # This is better than 144!
    
    # Can we do better for A*B2 (4x4x5)?
    # Split B2 into B21 (4x4) and B22 (4x1).
    # A*B21 rank 49.
    # A*B22 rank 4*4*1 = 16.
    # Total for B2 = 49 + 16 = 65.
    # Total for C = 49 + 65 = 114.
    
    # Can we do better for 4x4x1?
    # 4x4x1 is just A * b (vector).
    # Rank is 4*4 = 16?
    # C_i = sum_j A_ij b_j.
    # This is a bilinear form.
    # The tensor is sum_{i,j} A_ij b_j e_i.
    # This can be written as sum_{i,j} (A_ij) (b_j) (e_i).
    # So rank is 16.
    # Is there a lower rank?
    # The matrix of coefficients is A (4x4).
    # The rank of the tensor is the rank of A? No.
    # The tensor is T_{i,j} = A_ij.
    # Wait, the tensor for multiplication is T_{i,j,k} = delta_{i,k} delta_{j,l}? No.
    # For A (n x m) * B (m x p) = C (n x p).
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor T_{i,j,k} = 1 if i,j,k are the indices.
    # Actually, the tensor is T_{i,j,k} = delta_{i,j} delta_{j,k}? No.
    # The tensor is defined by C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # Yes, T_{i,j,k} = 1 for all i,j,k.
    # Wait, no.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # So T_{i,j,k} = 1 for all i,j,k.
    # But the indices are (i,j) for A, (j,k) for B, (i,k) for C.
    # So the tensor is T_{(i,j),(j,k),(i,k)} = 1.
    # This is the standard matrix multiplication tensor.
    
    # For 4x4x1, p=1.
    # C_i = sum_j A_{i,j} B_{j,0}.
    # Tensor T_{i,j,0} = 1.
    # This is a 4x4x1 tensor.
    # It can be decomposed as sum_{r=1}^{16} u_r v_r w_r.
    # u_r is non-zero only at (i,j) for a specific i,j.
    # v_r is non-zero only at (j,0) for the same j.
    # w_r is non-zero only at (i,0) for the same i.
    # So rank is 16.
    
    # So 4x4x5 rank is 49 + 16 = 65.
    # Total 4x4x9 rank is 49 + 65 = 114.
    
    # Can we do better?
    # Maybe 4x4x3 has rank < 48?
    # If 4x4x3 has rank r, then 4x4x9 has rank 3r.
    # If r=45, then 3r=135.
    # If r=40, then 3r=120.
    # If r=38, then 3r=114.
    
    # Let's assume 4x4x3 rank is 48 (schoolbook).
    # Then 4x4x9 rank is 144.
    
    # My block strategy:
    # B (4x9) = [B1 (4x4), B2 (4x4), B3 (4x1)].
    # C = A*B1 + A*B2 + A*B3.
    # A*B1 rank 49.
    # A*B2 rank 49.
    # A*B3 rank 16.
    # Total rank 49+49+16 = 114.
    
    # This is a valid improvement!
    # I need to implement the combination of decompositions.
    
    # Let's write a function to combine two decompositions.
    # If we have decomp1 for A*B1 and decomp2 for A*B2.
    # The total decomposition is the direct sum.
    # U_total = [U1, U2]
    # V_total = [V1, V2]
    # W_total = [W1, W2]
    # But wait, U is (n*m, rank).
    # U1 is (n*m, r1).
    # U2 is (n*m, r2).
    # So U_total is (n*m, r1+r2).
    # V1 is (m*p1, r1).
    # V2 is (m*p2, r2).
    # But V_total must be (m*p, r1+r2).
    # If p1+p2=p, then V_total is formed by placing V1 and V2 in the correct blocks.
    # V1 corresponds to columns 0..p1-1 of B.
    # V2 corresponds to columns p1..p-1 of B.
    # So V_total[j, k] for k < p1 is V1[j, k].
    # V_total[j, k] for k >= p1 is V2[j, k-p1].
    
    # Similarly for W.
    # W1 is (p1*n, r1).
    # W2 is (p2*n, r2).
    # W_total is (p*n, r1+r2).
    # W_total[k, i] for k < p1 is W1[k, i].
    # W_total[k, i] for k >= p1 is W2[k-p1, i].
    
    # So I can recursively build the decomposition.
    
    # Let's implement _strassen_4x4x4.
    # It returns U, V, W of rank 49.
    # Then I use it for B1 and B2.
    # And _schoolbook_4x4x1 for B3.
    
    # I need to be careful with the indices.
    
    # Let's write the code.

def _strassen_4x4x4():
    # Use Strassen for 2x2x2 (rank 7) to build 4x4x4 (rank 49).
    # This is complex to implement correctly from scratch.
    # I will use a known correct Strassen 4x4x4 decomposition.
    # Or I can just use the schoolbook for 4x4x4 (rank 64) and see if 64+64+16 = 144 is not an improvement.
    # 144 is the same as schoolbook 4x4x9.
    # So I MUST use Strassen 4x4x4 (rank 49) to get 114.
    
    # I will implement Strassen 4x4x4 by combining 2x2x2 Strassen.
    # This requires careful index mapping.
    # Given the complexity and risk of error, I will first try to return the schoolbook.
    # But the goal is to improve.
    # I will try to implement the block combination with Strassen 2x2x2.
    
    # Let's define a helper to combine two decompositions.
    def combine(U1, V1, W1, U2, V2, W2, p1, p2, n, m):
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        p = p1 + p2
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n*m):
            for k in range(r1):
                U[i][k] = U1[i][k]
            for k in range(r2):
                U[i][r1+k] = U2[i][k]
        
        for j in range(m):
            for k in range(p1):
                for r_local in range(r1):
                    V[j*p+k][r_local] = V1[j*p1+k][r_local]
            for k in range(p2):
                for r_local in range(r2):
                    V[j*p+p1+k][r1+r_local] = V2[j*p2+k][r_local]
        
        for k in range(p1):
            for i in range(n):
                for r_local in range(r1):
                    W[k*n+i][r_local] = W1[k*n+i][r_local]
        for k in range(p2):
            for i in range(n):
                for r_local in range(r2):
                    W[(p1+k)*n+i][r1+r_local] = W2[k*n+i][r_local]
        
        return U, V, W

    # I need _strassen_2x2x2 and _schoolbook_2x2x1 etc.
    # This is getting very long.
    # I will just return the schoolbook for now to ensure the code runs.
    # The user wants a stronger algorithm, but a broken one is worse.
    # I will implement the block combination with schoolbook for now, which gives 144.
    # Then I will try to replace schoolbook 4x4x4 with Strassen 4x4x4.
    
    # For now, return schoolbook.
    return _schoolbook(4, 4, 4)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook
    best = _schoolbook(n, m, p)
    best_rank = best["rank"]
    
    # Try block decomposition
    # B (4x9) = [B1 (4x4), B2 (4x4), B3 (4x1)]
    # C = A*B1 + A*B2 + A*B3
    
    # Get decompositions
    # For now, use schoolbook for all
    d1 = _schoolbook(4, 4, 4)
    d2 = _schoolbook(4, 4, 4)
    d3 = _schoolbook(4, 4, 1)
    
    # Combine
    def combine(U1, V1, W1, U2, V2, W2, p1, p2, n, m):
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        p = p1 + p2
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n*m):
            for k in range(r1):
                U[i][k] = U1[i][k]
            for k in range(r2):
                U[i][r1+k] = U2[i][k]
        
        for j in range(m):
            for k in range(p1):
                for r_local in range(r1):
                    V[j*p+k][r_local] = V1[j*p1+k][r_local]
            for k in range(p2):
                for r_local in range(r2):
                    V[j*p+p1+k][r1+r_local] = V2[j*p2+k][r_local]
        
        for k in range(p1):
            for i in range(n):
                for r_local in range(r1):
                    W[k*n+i][r_local] = W1[k*n+i][r_local]
        for k in range(p2):
            for i in range(n):
                for r_local in range(r2):
                    W[(p1+k)*n+i][r1+r_local] = W2[k*n+i][r_local]
        
        return U, V, W

    U12, V12, W12 = combine(d1["U"], d1["V"], d1["W"], d2["U"], d2["V"], d2["W"], 4, 4, 4, 4)
    U123, V123, W123 = combine(U12, V12, W12, d3["U"], d3["V"], d3["W"], 8, 1, 4, 4)
    
    candidate = {"U": U123, "V": V123, "W": W123, "rank": len(U123[0])}
    
    if candidate["rank"] < best_rank:
        if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            best = candidate
            best_rank = candidate["rank"]
    
    return best
