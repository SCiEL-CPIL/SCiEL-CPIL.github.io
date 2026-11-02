"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 23
    
    # U has shape (n*m, rank) = (9, 23)
    # V has shape (m*p, rank) = (9, 23)  
    # W has shape (p*n, rank) = (9, 23)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # For 3x3: 
    # A index: i*m+j where i=row, j=col of A (i,j in 0..2)
    # B index: j*p+k where j=row, k=col of B (j,k in 0..2)
    # C index: k*n+i where k=row, i=col of C... wait
    # "W[k*n+i,r]" - so W is indexed by (k*n+i) where k is... 
    # The output factor follows the official trace convention W[k*n+i,r]
    # C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # So C has indices (i,k) where i is row, k is col
    # W[k*n+i] means: first index is k*n+i, so k is the "row" in the k*n+i sense
    # Wait, let me re-read: "W[k*n+i,r]" 
    # So W row index = k*n + i, where presumably k and i are the output indices
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the output C has entries indexed by (i,k). 
    # W[k*n+i] suggests the output index is (k, i) with k as major, i as minor?
    # Or maybe it's C[k,i] = sum_j A[k,j] B[j,i]? That would be C = A*B transposed?
    
    # Let me re-read the task: "The output factor follows the official trace convention W[k*n+i,r]"
    # In the schoolbook: W[k*n+i][column] = 1 where the loop is for i in range(n), j in range(m), k in range(p)
    # And U[i*m+j][column] = 1, V[j*p+k][column] = 1
    # So the term is A[i,j] * B[j,k] and it contributes to W[k*n+i]
    # This means W[k*n+i] corresponds to C[i,k] (since A[i,j]*B[j,k] contributes to C[i,k])
    # So W row index = k*n + i, which means: for C[i,k], the W index is k*n+i
    # i.e., W is indexed with k as the "outer" index and i as the "inner" index
    # W[k*n+i] = C[i,k] in some sense, or rather the contribution goes to position (k*n+i) in W
    
    # So: A index = i*m + j (i=row, j=col of A)
    #     B index = j*p + k (j=row, k=col of B)
    #     C/W index = k*n + i (k=col of C, i=row of C)
    
    # This means C[i,k] gets its value from W[k*n+i]
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # I'll construct a rank-23 decomposition
    # This is based on known results for 3x3 matrix multiplication tensor rank
    
    # Let me use a specific known construction.
    # The 23 terms are defined by linear combinations.
    
    # I'll use the following approach based on literature:
    # Terms 0-6: Strassen-like for the 2x2 core
    # Terms 7-22: Border terms with some sharing
    
    # Actually, let me just try to verify a known rank-23 decomposition.
    
    # For now, let me use a rank-26 approach that I can verify,
    # and then try to optimize.
    
    # Rank 26: Use 7 terms for 2x2 core, 19 for border (not 15, because of tensor structure)
    # Actually, let me just do schoolbook minus 1 by finding one redundancy.
    
    # Alternative: just return rank 23 with a known decomposition.
    
    # Let me try to construct it computationally.
    
    # The matrix multiplication tensor T has T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # for some valid i,j,k. Otherwise 0.
    
    # T is a 9x9x9 tensor. We want T = sum_{t=0}^{22} U[:,t] * V[:,t] * W[:,t] (outer products)
    
    # This is a system of 729 equations in 23*27 = 621 unknowns (entries of U,V,W).
    # Overdetermined, but the system is consistent for rank 23.
    
    # This is hard to solve directly. Let me use a known explicit decomposition.
    
    # After research, here's a rank-23 decomposition for 3x3 matrix multiplication:
    
    # I'll define the 23 terms explicitly.
    
    # Let me use a different approach: I'll use the fact that we can write
    # the decomposition in terms of specific linear combinations.
    
    # Given the complexity, let me just return a valid rank-27 (schoolbook) 
    # and note that improvement is needed.
    
    # Actually, let me try rank 26 first.
    
    # For rank 26, I can use Strassen for the top-left 2x2 and handle borders.
    # But as I noted, naive 7+15=22 doesn't work.
    
    # Let me just do schoolbook with one term removed if possible.
    
    # Actually, I recall that you can get rank 26 by using one Strassen-like 
    # identity to replace 2 schoolbook terms with 1, saving 1.
    
    # For example, for computing C00 and C11, instead of:
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    # We can't directly save here.
    
    # Let me just implement the known rank-23 decomposition from the literature.
    
    # Here's a specific construction (adapted from known results):
    
    # I'll use the following 23 rank-1 terms. Each term t has:
    # u_t: a vector in R^9 (coefficients for A)
    # v_t: a vector in R^9 (coefficients for B)
    # w_t: a vector in R^9 (coefficients for C)
    
    # The decomposition is: sum_t u_t * v_t * w_t = T
    
    # Let me define them:
    
    terms = []
    
    # I'll use a known rank-23 construction.
    # Based on the work of Landsberg and others, here's one:
    
    # The construction uses the following idea:
    # Write 3x3 as a 2x2 block plus borders, and use 7 terms for the core
    # and 16 terms for the borders (not 15, because one border term 
    # needs to be split to maintain tensor rank properties).
    
    # 7 + 16 = 23.
    
    # Core terms (Strassen for 2x2):
    # A_core = A[0:2, 0:2], B_core = B[0:2, 0:2], C_core = C[0:2, 0:2]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each M_t, the A-linear-combination, B-linear-combination, and C-contribution:
    
    # M1: A part: A00+A11 -> indices (0*3+0, 1*3+1) = (0,4) in A-indexing
    #     B part: B00+B11 -> indices (0*3+0, 1*3+1) = (0,4) in B-indexing
    #     C part: contributes to C00 and C11
    #     C00 is at W index 0*3+0=0, C11 is at W index 1*3+1=4
    #     So W part: +1 at index 0, +1 at index 4
    
    # Let me define each term:
    
    # Term 0: M1
    terms.append((
        {0: 1, 4: 1},      # A: A00 + A11 (A[0,0] and A[1,1])
        {0: 1, 4: 1},      # B: B00 + B11 (B[0,0] and B[1,1])
        {0: 1, 4: 1}       # C: C00 + C11 (W[0*3+0] and W[1*3+1])
    ))
    
    # Term 1: M2
    terms.append((
        {3: 1, 4: 1},      # A: A10 + A11 (A[1,0] and A[1,1]) -> indices 1*3+0=3, 1*3+1=4
        {0: 1},            # B: B00 -> index 0*3+0=0
        {3: 1}             # C: C10 -> W[0*3+1]=3
    ))
    
    # Term 2: M3
    terms.append((
        {0: 1},            # A: A00 -> index 0
        {3: 1, 4: -1},    # B: B10 - B11 -> indices 1*3+0=3, 1*3+1=4
        {1: 1}             # C: C01 -> W[1*3+0]=3... wait
    ))
    
    # Wait, I need to be more careful with indices.
    # A[i,j] has index i*m+j = i*3+j
    # B[j,k] has index j*p+k = j*3+k
    # C[i,k] has W index k*n+i = k*3+i
    
    # C00: i=0,k=0 -> W[0*3+0]=0
    # C01: i=0,k=1 -> W[1*3+0]=3
    # C10: i=1,k=0 -> W[0*3+1]=1
    # C11: i=1,k=1 -> W[1*3+1]=4
    
    # Let me redo:
    terms = []
    
    # Term 0: M1 = (A00+A11)(B00+B11), contributes +1 to C00 and +1 to C11
    # A: A[0,0]=idx0, A[1,1]=idx4
    # B: B[0,0]=idx0, B[1,1]=idx4
    # W: C[0,0]=W[0]=0, C[1,1]=W[4]=4
    terms.append(({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}))
    
    # Term 1: M2 = (A10+A11)B00, contributes +1 to C10
    # A: A[1,0]=idx3, A[1,1]=idx4
    # B: B[0,0]=idx0
    # W: C[1,0]=W[1]=1
    terms.append(({3:1, 4:1}, {0:1}, {1:1}))
    
    # Term 2: M3 = A00(B10-B11), contributes +1 to C01
    # A: A[0,0]=idx0
    # B: B[1,0]=idx3, B[1,1]=idx4
    # W: C[0,1]=W[3]=3
    terms.append(({0:1}, {3:1, 4:-1}, {3:1}))
    
    # Term 3: M4 = A11(B01-B00), contributes +1 to C00 and +1 to C10
    # A: A[1,1]=idx4
    # B: B[0,1]=idx1, B[0,0]=idx0
    # W: C[0,0]=W[0]=0, C[1,0]=W[1]=1
    terms.append(({4:1}, {1:1, 0:-1}, {0:1, 1:1}))
    
    # Term 4: M5 = (A00+A01)B11, contributes -1 to C00 and +1 to C01
    # A: A[0,0]=idx0, A[0,1]=idx1
    # B: B[1,1]=idx4
    # W: C[0,0]=W[0]=0, C[0,1]=W[3]=3
    terms.append(({0:1, 1:1}, {4:1}, {0:-1, 3:1}))
    
    # Term 5: M6 = (A10-A00)(B00+B01), contributes +1 to C11
    # A: A[1,0]=idx3, A[0,0]=idx0
    # B: B[0,0]=idx0, B[0,1]=idx1
    # W: C[1,1]=W[4]=4
    terms.append(({3:1, 0:-1}, {0:1, 1:1}, {4:1}))
    
    # Term 6: M7 = (A01-A11)(B10+B11), contributes +1 to C00
    # A: A[0,1]=idx1, A[1,1]=idx4
    # B: B[1,0]=idx3, B[1,1]=idx4
    # W: C[0,0]=W[0]=0
    terms.append(({1:1, 4:-1}, {3:1, 4:1}, {0:1}))
    
    # Now border terms (16 terms):
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # That's 15 terms. I need 16. 
    # The issue is that 7+15=22 < 23, so I must be missing a term.
    # Perhaps one of the border "terms" needs to be split.
    
    # Let me just add all 15 border terms and see if we get 22, 
    # and then figure out the 23rd.
    
    # Border term 7: A00*B02 -> C02
    # A: A[0,0]=idx0, B: B[0,2]=idx2, W: C[0,2]=W[2*3+0]=6
    terms.append(({0:1}, {2:1}, {6:1}))
    
    # Border term 8: A01*B12 -> C02
    # A: A[0,1]=idx1, B: B[1,2]=idx5, W: C[0,2]=W[6]=6
    terms.append(({1:1}, {5:1}, {6:1}))
    
    # Border term 9: A02*B22 -> C02
    # A: A[0,2]=idx2, B: B[2,2]=idx8, W: C[0,2]=W[6]=6
    terms.append(({2:1}, {8:1}, {6:1}))
    
    # Border term 10: A10*B02 -> C12
    # A: A[1,0]=idx3, B: B[0,2]=idx2, W: C[1,2]=W[2*3+1]=7
    terms.append(({3:1}, {2:1}, {7:1}))
    
    # Border term 11: A11*B12 -> C12
    # A: A[1,1]=idx4, B: B[1,2]=idx5, W: C[1,2]=W[7]=7
    terms.append(({4:1}, {5:1}, {7:1}))
    
    # Border term 12: A12*B22 -> C12
    # A: A[1,2]=idx5, B: B[2,2]=idx8, W: C[1,2]=W[7]=7
    terms.append(({5:1}, {8:1}, {7:1}))
    
    # Border term 13: A20*B00 -> C20
    # A: A[2,0]=idx6, B: B[0,0]=idx0, W: C[2,0]=W[0*3+2]=2
    terms.append(({6:1}, {0:1}, {2:1}))
    
    # Border term 14: A21*B10 -> C20
    # A: A[2,1]=idx7, B: B[1,0]=idx3, W: C[2,0]=W[2]=2
    terms.append(({7:1}, {3:1}, {2:1}))
    
    # Border term 15: A22*B20 -> C20
    # A: A[2,2]=idx8, B: B[2,0]=idx6, W: C[2,0]=W[2]=2
    terms.append(({8:1}, {6:1}, {2:1}))
    
    # Border term 16: A20*B01 -> C21
    # A: A[2,0]=idx6, B: B[0,1]=idx1, W: C[2,1]=W[1*3+2]=5
    terms.append(({6:1}, {1:1}, {5:1}))
    
    # Border term 17: A21*B11 -> C21
    # A: A[2,1]=idx7, B: B[1,1]=idx4, W: C[2,1]=W[5]=5
    terms.append(({7:1}, {4:1}, {5:1}))
    
    # Border term 18: A22*B21 -> C21
    # A: A[2,2]=idx8, B: B[2,1]=idx7, W: C[2,1]=W[5]=5
    terms.append(({8:1}, {7:1}, {5:1}))
    
    # Border term 19: A20*B02 -> C22
    # A: A[2,0]=idx6, B: B[0,2]=idx2, W: C[2,2]=W[2*3+2]=8
    terms.append(({6:1}, {2:1}, {8:1}))
    
    # Border term 20: A21*B12 -> C22
    # A: A[2,1]=idx7, B: B[1,2]=idx5, W: C[2,2]=W[8]=8
    terms.append(({7:1}, {5:1}, {8:1}))
    
    # Border term 21: A22*B22 -> C22
    # A: A[2,2]=idx8, B: B[2,2]=idx8, W: C[2,2]=W[8]=8
    terms.append(({8:1}, {8:1}, {8:1}))
    
    # That's 7 + 15 = 22 terms. I need one more to get 23.
    # This suggests my decomposition is incomplete or incorrect.
    
    # Let me verify: does this 22-term decomposition actually work?
    # It should give the correct C entries if the Strassen part is correct
    # and the border parts are correct.
    
    # The issue might be that the Strassen decomposition, when embedded,
    # doesn't cleanly separate from the border in the tensor rank sense.
    
    # Let me just go with 22 terms and see if it's valid. If not, I'll add a dummy.
    
    # Actually, I think the problem is that my Strassen terms are for a 2x2 
    # multiplication in isolation, but in the 3x3 context, the tensor rank
    # is not simply additive. The 22 terms I've written should actually 
    # correctly compute C = A*B, so the tensor decomposition should be valid.
    
    # Let me verify by checking: for each C[i,k], the sum of contributions
    # from all terms should equal sum_j A[i,j]*B[j,k].
    
    # If it works, then rank 22 is valid, contradicting the lower bound of 23.
    # Since the lower bound is well-established, I must have an error.
    
    # Let me double-check the Strassen formulas:
    # For 2x2: C = A*B where A,B are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify C00:
    # M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A01*B10 + A11*B01 - A11*B10
    # That's not right! C00 should be A00*B00 + A01*B10.
    
    # I have an extra term: A11*B01 - A11*B10. Let me recheck.
    
    # Actually, I think I made an error in the Strassen formula. Let me look it up properly.
    
    # Correct Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me re-expand C00 more carefully:
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M4 = A11(B01-B00) = A11*B01 - A11*B00
    # M5 = (A00+A01)B11 = A00*B11 + A01*B11
    # M7 = (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # 
    # C00 = M1 + M4 - M5 + M7
    # = (A00*B00 + A00*B11 + A11*B00 + A11*B11)
    #   + (A11*B01 - A11*B00)
    #   - (A00*B11 + A01*B11)
    #   + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    # 
    # Collect terms:
    # A00*B00: +1
    # A00*B11: +1 - 1 = 0
    # A11*B00: +1 - 1 = 0
    # A11*B11: +1 - 1 = 0
    # A11*B01: +1
    # A01*B11: -1 + 1 = 0
    # A01*B10: +1
    # A11*B10: -1
    # 
    # So C00 = A00*B00 + A11*B01 + A01*B10 - A11*B10
    # 
    # But correct C00 = A00*B00 + A01*B10
    # 
    # I have extra terms A11*B01 - A11*B10. This means my Strassen formula is wrong!
    
    # Let me look up the correct Strassen formulas again.
    
    # The correct Strassen algorithm:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b10 - b11)
    # p4 = a11(b01 - b00)
    # p5 = (a00 + a01)b11
    # p6 = (a10 - a00)(b00 + b01)
    # p7 = (a01 - a11)(b10 + b11)
    # 
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # Let me re-expand c00 = p1 + p4 - p5 + p7:
    # p1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # p4 = a11(b01-b00) = a11*b01 - a11*b00
    # p5 = (a00+a01)b11 = a00*b11 + a01*b11
    # p7 = (a01-a11)(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
    # 
    # c00 = p1 + p4 - p5 + p7
    # = [a00*b00 + a00*b11 + a11*b00 + a11*b11]
    #   + [a11*b01 - a11*b00]
    #   - [a00*b11 + a01*b11]
    #   + [a01*b10 + a01*b11 - a11*b10 - a11*b11]
    # 
    # = a00*b00 
    #   + a00*b11 - a00*b11 
    #   + a11*b00 - a11*b00
    #   + a11*b11 - a11*b11
    #   + a11*b01
    #   - a01*b11 + a01*b11
    #   + a01*b10
    #   - a11*b10
    # 
    # = a00*b00 + a11*b01 + a01*b10 - a11*b10
    
    # This is still wrong! The correct c00 = a00*b00 + a01*b10.
    
    # I must have the Strassen formulas wrong. Let me look them up properly.
    
    # After checking, the correct Strassen formulas are:
    # p1 = (a00 + a11) * (b00 + b11)
    # p2 = (a10 + a11) * b00
    # p3 = a00 * (b10 - b11)
    # p4 = a11 * (b01 - b00)
    # p5 = (a00 + a01) * b11
    # p6 = (a10 - a00) * (b00 + b01)
    # p7 = (a01 - a11) * (b10 + b11)
    # 
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # I keep getting the same result. Let me try with concrete numbers.
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[19,22],[43,50]]
    # 
    # p1 = (1+4)(5+8) = 5*13 = 65
    # p2 = (3+4)*5 = 7*5 = 35
    # p3 = 1*(7-8) = -1
    # p4 = 4*(6-5) = 4
    # p5 = (1+2)*8 = 24
    # p6 = (3-1)*(5+6) = 2*11 = 22
    # p7 = (2-4)*(7+8) = -2*15 = -30
    # 
    # c00 = p1 + p4 - p5 + p7 = 65 + 4 - 24 + (-30) = 15. But correct is 19!
    
    # So my formulas are definitely wrong. Let me find the correct ones.
    
    # The correct Strassen algorithm (from Wikipedia):
    # p1 = (a00 + a11) · (b00 + b11)
    # p2 = (a10 + a11) · b00
    # p3 = a00 · (b10 − b11)
    # p4 = a11 · (b01 − b00)
    # p5 = (a00 + a01) · b11
    # p6 = (a10 − a00) · (b00 + b01)
    # p7 = (a01 − a11) · (b10 + b11)
    # 
    # c00 = p1 + p4 − p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 − p2 + p3 + p6
    
    # With my numbers: c00 = 65 + 4 - 24 + (-30) = 15 ≠ 19.
    
    # I think the issue is with p7. Let me check: p7 = (a01 - a11)(b10 + b11) = (2-4)(7+8) = -2*15 = -30.
    
    # Hmm, let me try a different source. Maybe the formula is:
    # c00 = p1 + p4 - p5 + p7
    # but with different p definitions.
    
    # Actually, I think I have the right formulas but let me recheck with the example.
    # Maybe I'm making an arithmetic error.
    # 
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C[0,0] = 1*5 + 2*7 = 5+14 = 19 ✓
    # C[0,1] = 1*6 + 2*8 = 6+16 = 22 ✓
    # C[1,0] = 3*5 + 4*7 = 15+28 = 43 ✓
    # C[1,1] = 3*6 + 4*8 = 18+32 = 50 ✓
    # 
    # p1 = (1+4)(5+8) = 5*13 = 65
    # p2 = (3+4)*5 = 35
    # p3 = 1*(7-8) = -1
    # p4 = 4*(6-5) = 4
    # p5 = (1+2)*8 = 24
    # p6 = (3-1)*(5+6) = 22
    # p7 = (2-4)*(7+8) = -30
    # 
    # c00 = 65 + 4 - 24 + (-30) = 15 ✗ (should be 19)
    
    # The formulas I have are incorrect. Let me find the right ones.
    
    # After more research, I believe the correct formulas are:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b10 - b11)
    # p4 = a11(b01 - b00)
    # p5 = (a00 + a01)b11
    # p6 = (a10 - a00)(b00 + b01)
    # p7 = (a01 - a11)(b10 + b11)
    # 
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # These are the same as what I had. But they don't work with my example.
    
    # Wait, maybe I have a sign error. Let me try:
    # c00 = p1 - p4 + p5 - p7? No, that gives 65-4+24-(-30) = 115.
    
    # Let me try: c00 = p1 + p4 + p5 + p7? 65+4+24+(-30) = 63. No.
    
    # I'm clearly making an error somewhere. Let me just use the schoolbook method 
    # for the 2x2 part (4 terms) and see if I can get a valid decomposition.
    
    # Actually, you know what, let me just abandon the Strassen approach and use
    # a known explicit rank-23 decomposition from the literature.
    
    # Given the time I've spent, let me just return the schoolbook rank-27 
    # and note that this is the baseline. The improvement to rank 23 or 26
    # requires a correct explicit construction.
    
    # Let me just return the schoolbook decomposition.
    
    rank = 27
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
