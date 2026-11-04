import numpy as np

def _strassen_terms():
    # 7 Strassen terms for 2x2 matrix multiplication.
    # Each term is a triple (a, b, c) where a is a 2x2 matrix for A-block,
    # b is a 2x2 matrix for B-block, c is a 2x2 matrix for C-block, such that
    #   sum_t (a_t ⊗ b_t) ⊗ c_t = M_2 (the 2x2x2 multiplication tensor).
    # We verify: for any 2x2 A, B, sum_t (a_t^T A a_t)(b_t^T B b_t) = C = A @ B.
    # Equivalently: C[i,j] = sum_t sum_{p,q} a_t[p,i] * A[p,q] * a_t[q, ...]
    # Let's define it via the standard Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Each M_t is a product (A_part) @ (B_part), which is itself a rank-4 term
    # (schoolbook for 2x2 @ 2x2). But we want to express each M_t as a single
    # rank-1 term in the U,V,W factorization. That's not possible directly.
    #
    # Instead, we keep the Strassen structure at the "block" level: each M_t
    # is computed via schoolbook (4 scalar multiplications), giving 7*4 = 28
    # rank-1 terms per 2x2 block. That's worse than schoolbook (8)!
    #
    # The key insight: Strassen reduces the NUMBER of block multiplications from
    # 8 to 7, but each block multiplication is itself 2x2x2 (rank 8 schoolbook).
    # So total rank = 7 * 8 = 56 per 2x2 block, vs schoolbook 8 * 8 = 64? No.
    #
    # Wait, I'm confusing levels. Let me reconsider.
    #
    # For 2x2 @ 2x2 -> 2x2: schoolbook uses 8 scalar multiplications (rank 8).
    # Strassen uses 7 "macro" multiplications, but each macro multiplication
    # is itself a 2x2 @ 2x2 = 8 scalar mults. So Strassen gives 7*8 = 56,
    # which is WORSE than schoolbook 8. That's not right either.
    #
    # The correct statement: the rank of the 2x2x2 multiplication tensor is 7
    # (Strassen proved this). This means there exist 7 rank-1 decompositions.
    # Each rank-1 term has U (2x2), V (2x2), W (2x2) matrices such that
    # the tensor equals sum of 7 outer products.
    #
    # Let me just look up / construct the explicit Strassen rank-7 decomposition.
    #
    # The 2x2x2 multiplication tensor T has T[i,j,k] = delta_{i,j} * delta_{j,k}... no.
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i,j,k in {0,1} (it's the all-ones tensor in the
    # appropriate sense). Actually T[i,j,k] = 1 iff the indices are consistent,
    # which for matrix mult means T[i,j,k] = 1 for all i,j,k (the "identity"
    # multiplication tensor).
    #
    # Strassen's rank-7 decomposition:
    # Define 7 triples (u_t, v_t, w_t) where u_t, v_t, w_t are 2x2 matrices.
    # The decomposition is: T = sum_t u_t ⊗ v_t ⊗ w_t
    # meaning T[i,j,k] = sum_t u_t[i,?] ... 
    #
    # Actually, the standard formulation: for the tensor M_2 (2x2 mat mult),
    # rank-7 means: there exist 7 pairs of "linear forms" on A and B, and
    # 7 "linear forms" on C, such that C = sum_t (L_t(A) * M_t(B)) * N_t
    # where L_t(A) is a 2x2 matrix (linear in A), M_t(B) is 2x2 (linear in B),
    # and the product is element-wise, then N_t maps the result to C.
    #
    # More precisely: C[i,k] = sum_t sum_{p,q} U_t[i,p] * A[p,q] * V_t[p,q] * W_t[q,k]
    # Hmm, this is getting complicated. Let me just use the known explicit form.
    #
    # From Strassen (1969), the 7 terms are:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[s,t],[u,v]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+c)h
    # M6 = (d-c)(e+g)
    # M7 = (b-d)(f+h)
    # s = M1 + M4 - M5 + M7
    # t = M3 + M5
    # u = M2 + M4
    # v = M1 - M2 + M3 + M6
    #
    # Each M_t is a product of two linear combinations. To express as rank-1
    # in the U,V,W factorization, each M_t = (linear form in A) * (linear form in B)
    # is a rank-1 term where:
    #   U maps A -> scalar (the linear form), V maps B -> scalar (the linear form),
    #   W maps the product scalar -> C (adds to specific entries).
    #
    # So for each t:
    #   U_t is a 2x2 matrix (coefficient of A in the linear form)
    #   V_t is a 2x2 matrix (coefficient of B in the linear form)
    #   W_t is a 2x2 matrix (how the product contributes to C)
    #
    # M1 = (a+d)(e+h): U1 = [[1,0],[1,0]] (a+d = [1,1,0,0] dot [a,b,c,d])
    #   Wait, a is A[0,0], b is A[0,1], c is A[1,0], d is A[1,1].
    #   a+d = 1*a + 0*b + 0*c + 1*d, so U1 = [[1,0],[0,1]] (diagonal, a+d = trace-like)
    #   Actually U1 should be a 2x2 matrix such that the "linear form" is U1 flattened.
    #   Let me define U_t as a 2x2 matrix where the linear form is sum_{p,q} U_t[p,q] * A[p,q].
    #   a+d: U1 = [[1,0],[0,1]]
    #   e+h: V1 = [[1,0],[0,1]]
    #   M1 contributes to: s (+1), v (+1). So W1 = [[1,0],[0,1]]
    #
    # M2 = (c+d)e: U2 = [[0,0],[1,1]], V2 = [[1,0],[0,0]], W2 = [[0,0],[1,0]]
    # M3 = a(f-h): U3 = [[1,0],[0,0]], V3 = [[0,1],[0,-1]], W3 = [[0,1],[0,1]]
    # M4 = d(g-e): U4 = [[0,0],[0,1]], V4 = [[-1,0],[1,0]], W4 = [[1,0],[1,0]]
    # M5 = (a+c)h: U5 = [[1,0],[1,0]], V5 = [[0,0],[0,1]], W5 = [[-1,1],[0,0]]
    # M6 = (d-c)(e+g): U6 = [[0,0],[-1,1]], V6 = [[1,0],[1,0]], W6 = [[0,0],[0,1]]
    # M7 = (b-d)(f+h): U7 = [[0,1],[0,-1]], V7 = [[0,1],[0,1]], W7 = [[1,1],[0,0]]
    #
    # Let me verify M1: (a+d)(e+h) contributes +1 to s and +1 to v.
    # s = M1 + M4 - M5 + M7: M1 coefficient +1, M4 +1, M5 -1, M7 +1.
    # t = M3 + M5: M3 +1, M5 +1.
    # u = M2 + M4: M2 +1, M4 +1.
    # v = M1 - M2 + M3 + M6: M1 +1, M2 -1, M3 +1, M6 +1.
    #
    # So W1 = [[1,0],[0,1]] (s and v get +1).
    # W2 = [[0,0],[1,0]] (u gets +1).
    # W3 = [[0,1],[0,1]] (t and v get +1).
    # W4 = [[1,0],[1,0]] (s and u get +1).
    # W5 = [[-1,1],[0,0]] (s gets -1, t gets +1).
    # W6 = [[0,0],[0,1]] (v gets +1).
    # W7 = [[1,1],[0,0]] (s and t get +1).
    #
    # Let me verify with a simple example: A = I, B = I.
    # a=1,b=0,c=0,d=1, e=1,f=0,g=0,h=1.
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)*1 = 1
    # M3 = 1*(0-1) = -1
    # M4 = 1*(0-1) = -1
    # M5 = (1+0)*1 = 1
    # M6 = (1-0)(1+0) = 1
    # M7 = (0-1)(0+1) = -1
    # s = 4 + (-1) - 1 + (-1) = 1 ✓
    # t = -1 + 1 = 0 ✓
    # u = 1 + (-1) = 0 ✓
    # v = 4 - 1 + (-1) + 1 = 3 ✗ (should be 1)
    #
    # Something is wrong. Let me recheck the Strassen formulas.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For A=B=I: A00=1,A01=0,A10=0,A11=1; B00=1,B01=0,B10=0,B11=1.
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)*1 = 1
    # M3 = 1*(0-1) = -1
    # M4 = 1*(0-1) = -1
    # M5 = (1+0)*1 = 1
    # M6 = (1-0)(1+0) = 1
    # M7 = (0-1)(0+1) = -1
    # C00 = 4 + (-1) - 1 + (-1) = 1 ✓
    # C01 = -1 + 1 = 0 ✓
    # C10 = 1 + (-1) = 0 ✓
    # C11 = 4 - 1 + (-1) + 1 = 3 ✗
    #
    # C11 should be 1. Let me recheck M6: (A11-A10)(B00+B10) = (1-0)(1+0) = 1.
    # C11 = M1 - M2 + M3 + M6 = 4 - 1 + (-1) + 1 = 3. That's wrong.
    #
    # I think I have the Strassen formulas slightly wrong. Let me look them up more carefully.
    #
    # Correct Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Hmm, let me try A = [[1,2],[3,4]], B = [[5,6],[7,8]].
    # C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(6-8) = -2
    # M4 = 4*(7-5) = 8
    # M5 = (1+3)*8 = 32
    # M6 = (4-3)(5+7) = 12
    # M7 = (2-4)(6+8) = -28
    # C00 = 65 + 8 - 32 + (-28) = 13 ✗ (should be 19)
    #
    # I clearly have the formulas wrong. Let me just use the correct ones from a reference.
    #
    # After more careful thought, the correct Strassen formulas are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let me recheck with A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
    # C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[5+14, 6+16],[15+28, 18+32]] = [[19,22],[43,50]]
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(6-8) = -2
    # M4 = 4*(7-5) = 8
    # M5 = (1+3)*8 = 32
    # M6 = (4-3)(5+7) = 12
    # M7 = (2-4)(6+8) = -2*14 = -28
    # C00 = 65 + 8 - 32 + (-28) = 13. Expected 19. Off by 6.
    #
    # I think the issue is with M5. Let me try M5 = (A00+A10)B11 = (1+3)*8 = 32. That seems right.
    # Maybe C00 = M1 + M4 - M5 + M7 is wrong. Let me try C00 = M1 + M4 - M5 + M7... 
    #
    # Actually, I recall now: the correct formula might be:
    # C00 = M1 + M4 - M5 + M7
    # But let me try a different source's version:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # I'm getting the same formulas. Let me just verify numerically in code and adjust if needed.
    # For now, let me just use schoolbook for the 2x2 blocks to be safe, but use Strassen
    # only if I can verify it works.
    #
    # Actually, you know what, let me just implement the full thing with schoolbook for all
    # sub-blocks but use the block structure to potentially find a lower rank.
    # The rank of 4x5x6 is at most 120 (schoolbook). We can do better.
    #
    # Known results: rank(4x5x6) is between 72 and 120. 
    # A simple improvement: use Strassen for the 2x2x2 sub-tensors.
    # 
    # Let me just implement schoolbook carefully and verify, then try to improve.
    pass

def construct():
    n, m, p = 4, 5, 6
    
    # Schoolbook decomposition: rank = n*m*p = 120
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # for each (i,j,k) term.
    
    # But we can do better using block structure.
    # Split A (4x5) into A1 (2x2), A2 (2x2), A3 (2x1)
    # Split B (5x6) into B1 (2x2), B2 (2x2), B3 (2x2), B4 (1x2), B5 (1x2), B6 (1x2)
    # Wait, B is 5x6, so split into rows: B_top (2x6), B_bottom (1x6)? No, 5 = 2+2+1.
    # B = [B1 (2x6); B2 (2x6); B3 (1x6)]
    # A = [A1 (2x2) | A2 (2x2) | A3 (2x1)]
    # C = A1@B1 + A2@B2 + A3@B3
    # A1@B1: 2x2 @ 2x6 -> split B1 into B1a (2x2), B1b (2x2), B1c (2x2)
    #   A1@B1a: 2x2@2x2, rank 8 (or 7 with Strassen)
    #   A1@B1b: 2x2@2x2, rank 8 (or 7)
    #   A1@B1c: 2x2@2x2, rank 8 (or 7)
    # A2@B2: similar, 3 sub-blocks of 2x2@2x2
    # A3@B3: 2x1 @ 1x6, rank 12
    #
    # Total with schoolbook: 6*8 + 12 = 48 + 12 = 60? No wait.
    # A is 4x5, so A has 4 rows and 5 columns.
    # A1 = A[0:2, 0:2] (2x2), A2 = A[0:2, 2:4] (2x2), A3 = A[0:2, 4:5] (2x1)
    # A4 = A[2:4, 0:2] (2x2), A5 = A[2:4, 2:4] (2x2), A6 = A[2:4, 4:5] (2x1)
    # B1 = B[0:2, 0:2] (2x2), B2 = B[0:2, 2:4] (2x2), B3 = B[0:2, 4:6] (2x2)
    # B4 = B[2:4, 0:2] (2x2), B5 = B[2:4, 2:4] (2x2), B6 = B[2:4, 4:6] (2x2)
    # B7 = B[4:5, 0:2] (1x2), B8 = B[4:5, 2:4] (1x2), B9 = B[4:5, 4:6] (1x2)
    #
    # C = A1@B1 + A2@B4 + A3@B7   (rows 0-1 of C, cols 0-1)
    #   + A1@B2 + A2@B5 + A3@B8   (rows 0-1, cols 2-3)
    #   + A1@B3 + A2@B6 + A3@B9   (rows 0-1, cols 4-5)
    #   + A4@B1 + A5@B4 + A6@B7   (rows 2-3, cols 0-1)
    #   + A4@B2 + A5@B5 + A6@B8   (rows 2-3, cols 2-3)
    #   + A4@B3 + A5@B6 + A6@B9   (rows 2-3, cols 4-5)
    #
    # 2x2@2x2 terms: 6*3 = 18 terms, each rank 8 (schoolbook) or 7 (Strassen)
    # 2x1@1x2 terms: 6*3 = 18 terms, each rank 2 (outer product of 2x1 and 1x2)
    #
    # Total with schoolbook: 18*8 + 18*2 = 144 + 36 = 180. That's worse than 120!
    #
    # I'm making an error. Let me reconsider.
    # A is 4x5, B is 5x6. The schoolbook rank is 4*5*6 = 120.
    # If I split into 2x2 blocks, I get 2x2 blocks of A (there are 2*3=6 such blocks,
    # but the last column is 2x1) and 2x2 blocks of B (3*3=9, but last row is 1x2).
    #
    # Actually, the cleanest split:
    # A = [A11 A12 A13; A21 A22 A23] where A11,A12,A21,A22 are 2x2, A13,A23 are 2x1
    # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33] where top 4 rows are 2x2 blocks,
    #     last row is 1x2 blocks.
    #
    # C = A11@B11 + A12@B21 + A13@B31  (C[0:2, 0:2])
    #   + A11@B12 + A12@B22 + A13@B32  (C[0:2, 2:4])
    #   + A11@B13 + A12@B23 + A13@B33  (C[0:2, 4:6])
    #   + A21@B11 + A22@B21 + A23@B31  (C[2:4, 0:2])
    #   + A21@B12 + A22@B22 + A23@B32  (C[2:4, 2:4])
    #   + A21@B13 + A22@B23 + A23@B33  (C[2:4, 4:6])
    #
    # 2x2@2x2: 12 terms (4 output blocks * 3 each)
    # 2x1@1x2: 12 terms
    #
    # Schoolbook: 12*8 + 12*2 = 96 + 24 = 120. ✓ (matches n*m*p)
    # Strassen: 12*7 + 12*2 = 84 + 24 = 108.
    #
    # So with Strassen for the 2x2x2 sub-tensors, we get rank 108.
    #
    # Now I need the explicit Strassen rank-7 decomposition. Let me just hardcode
    # the 7 terms and verify numerically.

    # Strassen rank-7 decomposition for 2x2x2:
    # Each term t has U_t (2x2), V_t (2x2), W_t (2x2) such that
    # C = sum_t (U_t^T A U_t) * (V_t^T B V_t) ... no, that's not right.
    #
    # The correct formulation: for each term t, we have:
    #   scalar_t = (linear form in A) * (linear form in B)
    #   C += scalar_t * W_t
    # where the linear form in A is sum_{i,j} U_t[i,j] * A[i,j]
    # and the linear form in B is sum_{i,j} V_t[i,j] * B[i,j].
    #
    # From Strassen:
    # M1 = (A00+A11)(B00+B11): U1=[[1,0],[0,1]], V1=[[1,0],[0,1]], W1=[[1,0],[0,1]]
    # M2 = (A10+A11)B00: U2=[[0,0],[1,1]], V2=[[1,0],[0,0]], W2=[[0,0],[1,0]]
    # M3 = A00(B01-B11): U3=[[1,0],[0,0]], V3=[[0,1],[0,-1]], W3=[[0,1],[0,1]]
    # M4 = A11(B10-B00): U4=[[0,0],[0,1]], V4=[[-1,0],[1,0]], W4=[[1,0],[1,0]]
    # M5 = (A00+A10)B11: U5=[[1,0],[1,0]], V5=[[0,0],[0,1]], W5=[[-1,1],[0,0]]
    # M6 = (A11-A10)(B00+B10): U6=[[0,0],[-1,1]], V6=[[1,0],[1,0]], W6=[[0,0],[0,1]]
    # M7 = (A01-A11)(B01+B11): U7=[[0,1],[0,-1]], V7=[[0,1],[0,1]], W7=[[1,1],[0,0]]
    
    # Let me verify this numerically.
    U_terms = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[1,0]],
        [[0,0],[-1,1]],
        [[0,1],[0,-1]],
    ]
    V_terms = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,0],[1,0]],
        [[0,1],[0,1]],
    ]
    W_terms = [
        [[1,0],[0,1]],
        [[0,0],[1,0]],
        [[0,1],[0,1]],
        [[1,0],[1,0]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
    ]
    
    # Verify: for random A, B, check that sum_t (sum U_t*A) * (sum V_t*B) * W_t = A@B
    A_test = np.array([[1.0, 2.0], [3.0, 4.0]])
    B_test = np.array([[5.0, 6.0], [7.0, 8.0]])
    C_expected = A_test @ B_test  # [[19,22],[43,50]]
    C_recon = np.zeros((2,2))
    for t in range(7):
        a_val = sum(U_terms[t][i][j] * A_test[i][j] for i in range(2) for j in range(2))
        b_val = sum(V_terms[t][i][j] * B_test[i][j] for i in range(2) for j in range(2))
        C_recon += a_val * b_val * np.array(W_terms[t])
    if not np.allclose(C_recon, C_expected):
        # Try to fix the Strassen formulas
        # The issue might be in my W matrices or the signs.
        # Let me just fall back to schoolbook for 2x2x2 (rank 8) for safety.
        strassen_ok = False
    else:
        strassen_ok = True
    
    # Build the full decomposition.
    # If Strassen works, use rank 7 per 2x2x2 block. Otherwise use rank 8.
    rank_2x2 = 7 if strassen_ok else 8
    
    # 12 blocks of 2x2@2x2, 12 blocks of 2x1@1x2
    rank = 12 * rank_2x2 + 12 * 2
    
    U = [[0.0] * rank for _ in range(n * m)]  # 20 x rank
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x rank
    
    col = 0
    
    # Define the block structure:
    # A blocks: A[r1:r2, c1:c2] for r1 in [0,2], c1 in [0,2,4]
    # B blocks: B[r1:r2, c1:c2] for r1 in [0,2,4], c1 in [0,2,4]
    # C blocks: C[r1:r2, c1:c2] for r1 in [0,2], c1 in [0,2,4]
    
    # For each output block C[ri:ri+2, cj:cj+2] (ri in [0,2], cj in [0,2,4]):
    #   C_block = sum over k of A[ri:ri+2, k] @ B[k, cj:cj+2]
    #   where k ranges over the 5 columns/rows, grouped as:
    #     k=0,1: A is 2x2, B is 2x2 -> 2x2@2x2
    #     k=2,3: A is 2x2, B is 2x2 -> 2x2@2x2
    #     k=4:   A is 2x1, B is 1x2 -> 2x1@1x2
    
    for ri in [0, 2]:
        for cj in [0, 2, 4]:
            # k=0,1: 2x2 @ 2x2
            # A block: A[ri:ri+2, 0:2], B block: B[0:2, cj:cj+2]
            if strassen_ok:
                for t in range(7):
                    for ai in range(2):
                        for aj in range(2):
                            U[(ri+ai)*m + aj][col] = U_terms[t][ai][aj]
                    for bi in range(2):
                        for bj in range(2):
                            V[(bi)*p + (cj+bj)][col] = V_terms[t][bi][bj]
                    for ci in range(2):
                        for cj2 in range(2):
                            W[(cj+cj2)*n + (ri+ci)][col] = W_terms[t][ci][cj2]
                    col += 1
            else:
                for ai in range(2):
                    for aj in range(2):
                        for bj in range(2):
                            U[(ri+ai)*m + aj][col] = 1.0
                            V[(bj)*p + (cj+bj)][col] = 1.0
                            W[(cj+bj)*n + (ri+ai)][col] = 1.0
                            col += 1
            # k=2,3: 2x2 @ 2x2
            if strassen_ok:
                for t in range(7):
                    for ai in range(2):
                        for aj in range(2):
                            U[(ri+ai)*m + (2+aj)][col] = U_terms[t][ai][aj]
                    for bi in range(2):
                        for bj in range(2):
                            V[(2+bi)*p + (cj+bj)][col] = V_terms[t][bi][bj]
                    for ci in range(2):
                        for cj2 in range(2):
                            W[(cj+cj2)*n + (ri+ci)][col] = W_terms[t][ci][cj2]
                    col += 1
            else:
                for ai in range(2):
                    for aj in range(2):
                        for bj in range(2):
                            U[(ri+ai)*m + (2+aj)][col] = 1.0
                            V[(2+bj)*p + (cj+bj)][col] = 1.0
                            W[(cj+bj)*n + (ri+ai)][col] = 1.0
                            col += 1
            # k=4: 2x1 @ 1x2
            for ai in range(2):
                for bj in range(2):
                    U[(ri+ai)*m + 4][col] = 1.0
                    V[4*p + (cj+bj)][col] = 1.0
                    W[(cj+bj)*n + (ri+ai)][col] = 1.0
                    col += 1
    
    # Verify the full decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, -1)
    V_reshaped = V_arr.reshape(m, p, -1)
    W_reshaped = W_arr.reshape(p, n, -1)
    
    # Check: for random A (n x m), B (m x p), C = A @ B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Using factors: C[i,k] = sum_r U_reshaped[i, :, r] * V_reshaped[:, :, r] * W_reshaped[:, k, r]... 
    # Actually, the factorization is: C[i,k] = sum_r sum_j U[i,j,r] * A[i,j] * V[j,k,r] * W[k,i,r]?
    # No, the standard tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 for all valid i,j,k (the multiplication tensor).
    # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_j A[i,j] * 1 * B[j,k]
    # = sum_j A[i,j] * T[i,j,k] * B[j,k] = sum_j A[i,j] * (sum_r U[i,j,r] V[j,k,r] W[k,i,r]) * B[j,k]
    # = sum_r (sum_j A[i,j] U[i,j,r]) * (sum_j B[j,k] V[j,k,r]) * W[k,i,r]
    #
    # So: C[i,k] = sum_r (U_reshaped[i,:,r] dot A[i,:]) * (V_reshaped[:,k,r] dot B[:,k]) * W_reshaped[k,i,r]
    # Wait, that's not quite right. Let me think again.
    #
    # The tensor T for matrix multiplication satisfies:
    # C[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]
    # where T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k]).
    #
    # If T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r], then:
    # C[i,k] = sum_j A[i,j] (sum_r U[i,j,r] V[j,k,r] W[k,i,r]) B[j,k]
    # = sum_r W[k,i,r] * (sum_j A[i,j] U[i,j,r]) * (sum_j B[j,k] V[j,k,r])
    #
    # So for each r:
    #   a_r = sum_j A[i,j] U[i,j,r]  (for fixed i, this is a scalar)
    #   b_r = sum_j B[j,k] V[j,k,r]  (for fixed k, this is a scalar)
    #   C[i,k] += W[k,i,r] * a_r * b_r
    #
    # Let me verify numerically.
    A_rand = np.random.rand(n, m)
    B_rand = np.random.rand(m, p)
    C_expected = A_rand @ B_rand
    C_recon = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            a_val = sum(A_rand[i, j] * U_reshaped[i, j, r] for j in range(m))
            for k in range(p):
                b_val = sum(B_rand[j, k] * V_reshaped[j, k, r] for j in range(m))
                C_recon[i, k] += W_reshaped[k, i, r] * a_val * b_val
    
    if not np.allclose(C_recon, C_expected, atol=1e-9):
        # Fallback to pure schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        # This is guaranteed correct.
    
    return {"U": U, "V": V, "W": W}
