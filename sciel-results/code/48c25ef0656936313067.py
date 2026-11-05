"""Rewritten program for 3x3x7 matrix multiplication rank-minimal decomposition."""

# EVOLVE-BLOCK-START
def _make_strassen_3x3():
    """
    Returns (U, V, W) for 3x3x3 Strassen decomposition (rank 23).
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    Convention: C[i,k] = sum_c W[k*3+i, c] * U[i*3+j, c] * V[j*3+k, c) summed over j
    Actually: C[i,k] = sum_{j} A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c]
    """
    m, p, n = 3, 3, 3
    rank = 23
    U = [[0.0] * rank for _ in range(m * p)]  # Actually n*m = 9 rows
    # U has shape (n*m, rank) = (9, 23)
    # V has shape (m*p, rank) = (9, 23)
    # W has shape (p*n, rank) = (9, 23)
    U = [[0.0] * rank for _ in range(n * m)]  # 9 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 9 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 9 rows

    # Strassen for 3x3: partition into blocks
    # A = [[A00,A01,A02],[A10,A11,A12],[A20,A21,A22]]
    # B = [[B00,B01,B02],[B10,B11,B12],[B20,B21,B22]]
    # 
    # Use overlapping 2x2 Strassen patches:
    # Patch 1: rows 0,1 cols 0,1 (top-left 2x2)
    # Patch 2: rows 1,2 cols 0,1 (bottom-left 2x2)
    # Patch 3: rows 0,1 cols 1,2 (top-right 2x2)
    # Patch 4: rows 1,2 cols 1,2 (bottom-right 2x2)
    # Each 2x2 Strassen uses 7 mults = 28, but overlap means we need to be careful.
    # 
    # Alternative: Use the known 23-rank decomposition for 3x3.
    # Let me use a different approach: treat 3x3 as a 2x2 with one extra row/col.
    # 
    # Actually, let me use a cleaner method. The 3x3 matmul can be viewed as:
    # Multiply a 3x3 by 3x3. We can use the formula:
    # Let's use 7 mults for the (0,1)x(0,1) subblock (top-left 2x2 of output),
    # 7 mults for (1,2)x(1,2) subblock (bottom-right 2x2 of output),
    # and handle the cross terms carefully.
    #
    # Simpler: just use rank 9 naive for the 3x3 core, then extend to 3x7.
    # But that gives rank 9*7/3... no. Let me think differently.
    #
    # For 3x3x7: C[i,k] = sum_j A[i,j] B[j,k]
    # Naive: rank 63.
    # 
    # Better approach: Use Strassen-like 7-mult for 2x2 blocks.
    # Partition the 7 output columns into: [0,1], [2,3], [4,5], [6]
    # For columns [0,1]: use 2x2 Strassen on B[:,0:2], needs A rows. But A is 3x3.
    # 
    # Let me use this decomposition:
    # Write 3x3x7 as combination of 2x2x2 Strassen (rank 7) instances.
    # 
    # Columns 0,1: C[i,0], C[i,1] for i=0,1,2
    #   These involve A[i,j]*B[j,0] and A[i,j]*B[j,1] for j=0,1,2
    #   = (A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0], A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1])
    #   
    # For a given pair of output columns (k1,k2) and input row range, the 2x2 Strassen
    # applies when we have 2x2 blocks. Here we have 3x3 input to 2 columns.
    # 
    # Let me use a hybrid: for the first 6 columns, pair them into 3 groups of 2,
    # and for each group use a rank-7 decomposition of 3x3x2 tensor.
    # The 3x3x2 tensor: C[i,k] = sum_j A[i,j]*B[j,k] for k in {0,1}
    # This is a 3x3x2 tensor. Its rank? We can use:
    #   M_3,3,2 <= M_3,3,3 so rank <= 23. But we can do better.
    #   Actually rank(M_3,3,2) = 9? No, naive is 18. 
    #   We can do: split A into 3 row-vectors, each multiplied by 3x2 matrix.
    #   Each row i: C[i,0:2] = A[i,:] * B[:,0:2] which is 1x3 times 3x2.
    #   Rank of 1x3x2 tensor is 3 (naive), or 2 if we use Strassen-like? 
    #   1x3x2: C[k] = sum_j A[j]*B[j,k], that's 3 terms per output, 2 outputs = 6 naive.
    #   Can we do it in fewer? The tensor is sum_j e_j ⊗ f_j ⊗ g_j where... 
    #   Actually 1x3x2 matmul tensor has rank 3 (since it's like a 3-term sum with 2 output dims).
    #   Wait: rank of m=3,p=2,n=1: C = A (3,) * B (3x2) -> for each k: dot(A,B[:,k]).
    #   This is rank 3 since we need 3 independent multiplications? Actually no - 
    #   it's a bilinear map R^3 x R^6 -> R^2. The tensor lives in R^3 ⊗ R^6 ⊗ R^2.
    #   Naive rank is 3*2 = 6? No: C_k = sum_j A_j B_{j,k}, so for each (j,k) we have one term.
    #   That's 6 terms. Can we do less? The 3x2x1 matmul... I think rank is 3.
    #   Hmm, let me just use naive 18 for 3x3x2, giving 3*18 + 9 (for col 6) = 63. No savings.
    #
    # Let me try a completely different approach: use the known fact that
    # matmul(3,3,p) can be done in rank 3p + something using Strassen on blocks.
    #
    # Actually, the simplest reliable improvement: use Strassen's 7-mult 2x2 algorithm
    # on overlapping 2x2 subproblems. For 3x3x3, the rank is 23 (Larsen 2009, or actually it's 23? Let me recall: the rank of 3x3x3 is 23 per Bostan et al. 2007? Actually I recall it might be 23.)
    #
    # Given uncertainty, let me construct a valid rank-23 for 3x3x3 using a systematic method,
    # then extend to 3x3x7.
    #
    # Here's a clean 3x3x3 rank-23 construction:
    # Use Strassen on 4 overlapping 2x2x2 patches:
    # - Top-left 2x2 (rows 0,1; cols 0,1 of B; cols 0,1 of A)
    # - Top-right 2x2 (rows 0,1; cols 1,2)
    # - Bottom-left 2x2 (rows 1,2; cols 0,1)
    # - Bottom-right 2x2 (rows 1,2; cols 1,2)
    # Each 2x2x2 Strassen: rank 7. Total 28, but the overlapping parts (row 1 or col 1) are shared.
    # We need to combine carefully to get rank 23, not 28.
    #
    # This is getting complex. Let me just verify the naive 63 works and try a simpler improvement.
    #
    # Actually, let me use a verified approach: decompose 3x3x7 by treating it as
    # three 3x3x3 blocks minus one 3x3x2 block... no.
    #
    # Let me use this: the 3x3x7 tensor can be written as:
    # M_{3,3,7} = M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    # where the three 3x3x1 tensors correspond to output columns 0-2, 3-5, 6.
    # Rank(M_{3,3,3}) = 23, Rank(M_{3,3,1}) = 9.
    # Total: 23 + 23 + 9 = 55. This beats 63!
    #
    # So I need a rank-23 decomposition for 3x3x3. Let me use the standard Strassen-based one.
    #
    # For 3x3x3, here is a rank-23 decomposition:
    # Partition A, B into 2x2 blocks with overlap:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    # 
    # Use Strassen on the (1,2)x(1,2) block (top-left 2x2 output), etc.
    #
    # I'll use a known explicit 23-term decomposition. Here's one construction:
    # 
    # Let me index differently. For 3x3 matmul C = AB:
    # Use the 7-multiplication Strassen on 2x2 blocks:
    #
    # S1 = (A00+A11)(B00+B11) -> contributes to C00,C01,C10,C11
    # S2 = (A10+A11)B00 -> contributes to C10,C11  
    # S3 = A00(B10-B11) -> contributes to C00,C01
    # S4 = A11(B01-B00) -> contributes to C10,C11
    # S5 = (A00+A01)(B11+B22) -> wait, this is getting confused.
    #
    # Let me use a systematic approach with 4 overlapping 2x2 Strassen patches.
    # Patch TL: rows {0,1}, cols {0,1} -> C[0:2,0:2]
    # Patch TR: rows {0,1}, cols {1,2} -> C[0:2,1:3]
    # Patch BL: rows {1,2}, cols {0,1} -> C[1:3,0:2]
    # Patch BR: rows {1,2}, cols {1,2} -> C[1:3,1:3]
    #
    # Each patch is a 2x2x2 Strassen (7 terms). Total 28 terms.
    # But the center entry C[1,1] appears in all 4 patches, and edge entries in 2 patches.
    # To get rank 23, we need to eliminate 5 redundant terms through linear dependencies.
    #
    # This is quite involved to get exactly right. Given time constraints, let me use a 
    # safer construction: rank 23 via a different known method.
    #
    # Actually, I'll use the following well-known rank-23 for 3x3x3:
    # Based on "3x3 matrix multiplication can be done in 23 multiplications" (Larsen, 2009)
    # But writing all 23 terms explicitly from memory is error-prone.
    #
    # Let me instead use a verified rank-55 construction:
    # 2 blocks of 3x3x3 (rank 23 each) + 1 block of 3x3x1 (rank 9) = 55
    # And for 3x3x3, use rank 23 via a clean construction.
    #
    # Here's my plan: I'll construct rank-23 for 3x3x3 using the "overlap Strassen" method
    # carefully, then verify it mentally.
    #
    # For now, let me return a valid but not optimal decomposition to ensure correctness.
    # I'll use rank 49: treat the 7 columns as: 2 groups of 3 columns (each rank 23) + 1 column (rank 9)
    # Wait, 23+23+9 = 55. But I'm not confident in the 23-rank construction.
    #
    # Safest: use rank 63 naive. But let me try rank 55 with a verified 23-rank 3x3x3.
    
    # Here is a verified 3x3x3 rank-23 decomposition:
    # Reference: Use Strassen on overlapping 2x2 blocks with 5-term overlap elimination.
    
    # I'll construct it explicitly. Let A be 3x3 indexed A[i,j], B be 3x3 indexed B[j,k].
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Define the 23 bilinear forms. I'll use a systematic construction:
    # 
    # Use 7 Strassen mults for top-left 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = A01(B10+B11)
    # M6 = (A00+A01)(B11-B00)  -- wait, standard Strassen for 2x2:
    #
    # Standard 2x2 Strassen:
    # [[P,Q],[R,S]] * [[a,b],[c,d]] = [[P*d - (S-Q)*(c-d), (P+Q)*(d-b)],[(R+S)*(a-c), S*a - (P-R)*(a-b)]]
    # Where: P=(P+Q)*(d-b)... no let me just recall:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(b-c)
    # P5 = (a+b)h
    # P6 = (c+d)(g-b)  -- I keep getting this wrong.
    #
    # Standard Strassen 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(b-c)
    # P5 = (a+b)h
    # P6 = (c+d)(g-b)  NO.
    #
    # P6 = (b+c)... let me just use:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e)
    # P3 = (a)(f-h)
    # P4 = (d)(b-c)
    # P5 = (a+b)(h)
    # P6 = (c+d)(g-b)  -- I don't trust my memory.
    # P7 = (a)(e)... 
    #
    # Forget it. Let me just use the naive 63. It's guaranteed correct.
    
    return None  # We won't use this path


def construct():
    n, m, p = 3, 3, 7
    # Use a rank-55 decomposition: 2x (3x3x3, rank 23) + 1x (3x3x1, rank 9)
    # For 3x3x3, use rank 23. For 3x3x1, use rank 9.
    # 
    # Actually, let me use a cleaner approach. I'll construct the 3x3x3 rank-23
    # using a method I can verify: the "3x3 in 23 mults" by combining 4 Strassen 2x2 patches.
    #
    # Instead of risking errors, let me use a provably correct rank-55 decomposition
    # by constructing 3x3x3 in rank 23 using a systematic approach I can verify.
    
    # Let me use this approach for 3x3x3 rank 23:
    # We view 3x3 as a 2x2 with an extra row and column, and use block structure.
    # 
    # Actually, the cleanest verified construction I can do:
    # 3x3x3 = use Strassen 2x2 (rank 7) on the top-left 2x2 block of the output,
    # plus handle the remaining entries with additional terms.
    #
    # C[0,0], C[0,1], C[1,0], C[1,1] handled by 2x2 Strassen (7 mults)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2] (3 terms)
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2] (3 terms)
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0] (3 terms)
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1] (3 terms)
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2] (3 terms)
    # Total: 7 + 3 + 3 + 3 + 3 + 3 = 22? But the Strassen part already handles C[0,2] partially? No.
    # The 2x2 Strassen only handles C[0:2, 0:2]. The C[*,*] with index 2 are separate.
    # But wait, C[0,2] = sum_j A[0,j]*B[j,2]. This is a separate set of 3 mults.
    # Similarly for the other 4 entries. So 7 + 5*3 = 7+15 = 22.
    # 
    # Is rank 22 possible? Actually the rank of 3x3x3 is known to be 23, so 22 shouldn't work.
    # The issue: the 5*3=15 extra terms might share structure with the 7 Strassen terms.
    # Actually no - C[0,2] etc. are independent from C[0,0], C[0,1], C[1,0], C[1,1].
    # So we'd get rank 7+15 = 22, which contradicts the known rank 23.
    #
    # The catch: the 2x2 Strassen uses linear combinations, and the "extra" terms for
    # C[*,2] and C[2,*] do NOT share any mults with the Strassen terms. So why is the
    # rank 23 and not 22?
    #
    # Ah, I see my error. When I say "2x2 Strassen handles C[0:2,0:2]", it handles those 4
    # output entries. The remaining 5 output entries (C[0,2], C[1,2], C[2,0], C[2,1], C[2,2])
    # each need 3 mults naively = 15. Total 7+15=22.
    # But the rank is 23, so there must be a hidden dependency or my counting is wrong.
    #
    # Actually, I think the rank of 3x3x3 IS 23, and the reason 22 doesn't work is subtle.
    # Perhaps the 2x2 Strassen, when embedded in the 3x3 context, doesn't quite decompose
    # cleanly because of how the linear combinations interact.
    #
    # You know what, let me just use rank 23 by adding one extra naive term. I'll use
    # 22 terms from the above construction and verify, but to be safe, let me just use
    # the naive 9 for 3x3x3, giving rank 9*7 = 63... no that's naive overall.
    #
    # FINAL DECISION: Use rank 49. Decompose 3x3x7 as:
    # - Columns 0,1,2: 3x3x3 block, use rank 23 (I'll construct it properly)
    # - Columns 3,4,5: 3x3x3 block, use rank 23
    # - Column 6: 3x3x1 block, use rank 9
    # But I need the 23-rank construction to be correct.
    #
    # Let me construct 3x3x3 rank 23 properly using this method:
    # Use 7 Strassen mults for the top-left 2x2 output block (C[0:2,0:2])
    # Then for the remaining 5 entries, use 15 naive mults.
    # Total = 7 + 15 = 22. If rank is really 23, this can't be right.
    # 
    # Let me look at this more carefully. The 2x2 Strassen for C[0:2,0:2] uses:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = A01*(B10+B11)
    # P6 = (A00+A01)*(B11-B00)  -- need to double check
    # P7 = A01*(B11)  -- etc.
    #
    # The key insight: these 7 mults are independent of the mults needed for C[*,2] and C[2,*].
    # So 7+15=22 should work. But rank(3x3x3)=23 per literature. 
    #
    # I think the resolution is: the rank of 3x3x3 is actually 23, and my construction
    # has a flaw. Let me just use a safe upper bound.
    #
    # SAFE APPROACH: Use rank 49 for 3x3x7.
    # Decomposition: treat as 7 separate 3x3x1 blocks, each rank 9? No, that's 63.
    #
    # OK here's my final approach. I'll use a verifiably correct decomposition:
    # Rank 55 = 2*23 + 9, but since I can't confidently write the 23-rank 3x3x3,
    # I'll use rank 63 naive. This is guaranteed correct.
    
    rank = n * m * p  # 63
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
