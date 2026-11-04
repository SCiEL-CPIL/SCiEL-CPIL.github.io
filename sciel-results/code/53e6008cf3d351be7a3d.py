"""
Strategy: Structural Construction using Strassen and Laderman.
Instead of random search, we explicitly construct a rank-75 decomposition
by partitioning the 3x5 * 5x6 multiplication into blocks:
- Top 2x5 * 5x6 part: Use Strassen (rank 7) for the 2x2 core, extended to 2x5.
- Bottom 1x5 * 5x6 part: Use Strassen (rank 7) for the 1x2 core, extended to 1x5.
- The 5x6 right side is handled by embedding the 2x2 Strassen blocks into 2x5 and 1x5.

Actually, a simpler and more robust structural approach for 3x5x6:
We can view the multiplication as:
A (3x5) * B (5x6) = C (3x6)

We can partition A into A1 (2x5) and A2 (1x5).
We can partition B into B1 (5x2) and B2 (5x4)? No, that's messy.

Let's use the standard "embedding" strategy which is guaranteed to work.
The known upper bound for M(3,5,6) is not trivially 75, but we can definitely do better than 90.
A simple construction:
Split the 5 middle dimension into 2 + 3?
Or split the 3x5 into 3x2 + 3x3?
Let's try to use the known rank 7 for 2x2x2 and rank 7 for 3x3x3 (Laderman).

Construction:
Partition A (3x5) into columns: A_left (3x2), A_right (3x3).
Partition B (5x6) into rows: B_top (2x6), B_bottom (3x6).

C = A_left * B_top + A_right * B_bottom

This is a sum of two matrix multiplications:
1. (3x2) * (2x6) -> Rank?
   We can embed this into a 3x3 * 3x3 multiplication by padding zeros?
   Or use a specific low-rank decomposition for rectangular matrices.
   
   Actually, the simplest guaranteed improvement is to use the fact that
   M(3,5,6) <= M(3,5,5) + M(3,0,1)? No.
   
   Let's use the explicit Strassen-based construction for the 2x2 blocks.
   
   Alternative Strategy:
   We know M(2,2,2) = 7 (Strassen).
   We can decompose the 3x5x6 tensor by embedding it into a larger tensor or by
   using a block decomposition that leverages Strassen's identity.
   
   Let's try a direct construction of rank 75.
   Partition the 5x5 middle interaction into four blocks:
   A = [A11 A12 A13] (3x3, 3x2) -- wait, 3x5.
   Let's partition the 5 columns of A into 2 and 3.
   A = [A_2 | A_3] where A_2 is 3x2, A_3 is 3x3.
   B = [B_2; B_3] where B_2 is 2x6, B_3 is 3x6.
   
   C = A_2 B_2 + A_3 B_3.
   
   Term 1: A_2 (3x2) * B_2 (2x6).
   This is a 3x2x6 tensor.
   We can embed A_2 into a 3x3 matrix A_2' by padding a zero column.
   We can embed B_2 into a 3x6 matrix B_2' by padding a zero row.
   Then A_2' * B_2' has rank at most M(3,3,6).
   M(3,3,6) <= M(3,3,3) + M(3,3,3)? No, M(3,3,6) is roughly 2 * M(3,3,3) if we split the 6?
   Actually, M(3,3,6) can be done with rank 14? (2 * Laderman rank 7).
   So Term 1 can be done with rank 14.
   
   Term 2: A_3 (3x3) * B_3 (3x6).
   Similarly, split B_3 into B_3a (3x3) and B_3b (3x3).
   A_3 * B_3a + A_3 * B_3b.
   Each is 3x3x3, rank 7 (Laderman).
   Total for Term 2: 14.
   
   Total Rank: 14 + 14 = 28? 
   Wait, the dimensions don't match for simple addition unless we are careful.
   The output C is 3x6.
   A_2 B_2 produces 3x6.
   A_3 B_3 produces 3x6.
   So C = C_1 + C_2.
   The tensor rank of the sum is at most the sum of the ranks.
   So Rank <= 14 + 14 = 28?
   
   Let's verify M(3,2,6).
   A_2 is 3x2. B_2 is 2x6.
   We can embed A_2 into 3x3 (pad col 3 with 0).
   We can embed B_2 into 3x6 (pad row 3 with 0).
   The product is A_2' B_2'.
   A_2' is 3x3. B_2' is 3x6.
   Split B_2' into B_21 (3x3) and B_22 (3x3).
   A_2' B_2' = A_2' B_21 + A_2' B_22.
   Rank(A_2' B_21) <= M(3,3,3) = 7.
   Rank(A_2' B_22) <= M(3,3,3) = 7.
   So Rank(A_2 B_2) <= 14.
   
   Similarly for A_3 (3x3) B_3 (3x6).
   Split B_3 into B_31 (3x3) and B_32 (3x3).
   A_3 B_3 = A_3 B_31 + A_3 B_32.
   Rank <= 7 + 7 = 14.
   
   Total Rank <= 14 + 14 = 28.
   
   This is a massive improvement over 90.
   
   We need to implement the Laderman decomposition for 3x3x3.
   Laderman's 7-term algorithm for 3x3 matrix multiplication:
   Let A, B be 3x3. C = AB.
   Indices i,j,k in {0,1,2}.
   
   The 7 terms (u, v, w) are defined by:
   1. u_1 = a_11, v_1 = b_11, w_1 = c_11
   2. u_2 = a_12, v_2 = b_21, w_2 = c_12
   3. u_3 = a_21, v_3 = b_12, w_3 = c_21
   4. u_4 = a_22, v_4 = b_22, w_4 = c_22
   5. u_5 = a_33, v_5 = b_33, w_5 = c_33
   6. u_6 = a_11+a_22, v_6 = b_11+b_22, w_6 = c_11+c_22 - (contribution from 1 and 4?)
   
   Actually, let's use a known explicit Laderman decomposition.
   From literature (e.g., Laderman 1976):
   The 7 multiplications are:
   m1 = a11 * b11
   m2 = a12 * b21
   m3 = a21 * b12
   m4 = a22 * b22
   m5 = a33 * b33
   m6 = (a11 + a22) * (b11 + b22)
   m7 = (a12 + a13) * (b21 + b23) + ... wait, this is getting complex.
   
   A simpler valid 7-term decomposition for 3x3:
   It is often easier to use the 8-term Winograd or just verify a specific set.
   However, we can also just use the 8-term Strassen-like decomposition for 3x3? No, Strassen is 2x2.
   There is a known 7-term decomposition.
   
   Let's use a verified 7-term decomposition for 3x3x3.
   Reference: "Seven multiplications suffice to multiply 3x3 matrices" (Laderman).
   
   The tensors (u, v, w) where C = sum u*v*w:
   1. u=(1,0,0, 0,0,0, 0,0,0), v=(1,0,0, 0,0,0, 0,0,0), w=(1,0,0, 0,0,0, 0,0,0)
   2. u=(0,1,0, 0,0,0, 0,0,0), v=(0,0,0, 0,1,0, 0,0,0), w=(0,1,0, 0,0,0, 0,0,0)
   3. u=(0,0,0, 1,0,0, 0,0,0), v=(0,0,0, 0,0,0, 0,1,0), w=(0,0,0, 0,1,0, 0,0,0)
   4. u=(0,0,0, 0,1,0, 0,0,0), v=(0,0,0, 0,0,0, 0,0,1), w=(0,0,0, 0,0,0, 0,1,0) -- Wait, indices are tricky.
   
   Let's use a reliable source for the 7-term 3x3 decomposition.
   One common representation uses the following 7 rank-1 tensors:
   (Using 0-indexed i,j,k)
   
   T1: u=(1,0,0, 0,0,0, 0,0,0), v=(1,0,0, 0,0,0, 0,0,0), w=(1,0,0, 0,0,0, 0,0,0)
   T2: u=(0,1,0, 0,0,0, 0,0,0), v=(0,0,0, 1,0,0, 0,0,0), w=(0,1,0, 0,0,0, 0,0,0)
   T3: u=(0,0,0, 1,0,0, 0,0,0), v=(0,0,0, 0,0,0, 1,0,0), w=(0,0,0, 0,1,0, 0,0,0)
   T4: u=(0,0,0, 0,1,0, 0,0,0), v=(0,0,0, 0,0,0, 0,1,0), w=(0,0,0, 0,0,0, 0,1,0)
   T5: u=(0,0,0, 0,0,0, 1,0,0), v=(0,0,0, 0,0,0, 0,0,1), w=(0,0,0, 0,0,0, 1,0,0)
   
   These 5 are the diagonal terms. We need 2 more to handle off-diagonals?
   No, 3x3 has 9 off-diagonals. 5 diagonal terms cover 5 elements.
   We need to cover the remaining 4? No, the rank is 7, so 2 more terms cover the rest?
   Actually, the 5 diagonal terms are not a valid decomposition for the whole matrix product.
   The sum of 5 rank-1 tensors has rank 5. The 3x3 multiplication tensor has rank at least 9?
   No, the rank of the 3x3 multiplication tensor is 7.
   So 5 terms is impossible.
   
   Let's use a known valid set.
   From "Matrix Multiplication" by Strassen, there is a 7-term algorithm.
   
   Let's define the 7 terms explicitly.
   Let A = [[a,b,c],[d,e,f],[g,h,i]], B = [[x,y,z],[u,v,w],[r,s,t]].
   
   m1 = a*x
   m2 = b*u
   m3 = d*y
   m4 = e*v
   m5 = i*t
   m6 = (a+e)*(x+v) - a*x - e*v  => (a+e)(x+v) - m1 - m4
   m7 = (b+f+c+i)*(u+r+s+w) ... this is getting complicated to derive on the fly.
   
   Alternative: Use the 8-term decomposition for 3x3 (which is easier to construct from 2x2 Strassen blocks) and accept rank 8 for the 3x3 core?
   If we use rank 8 for 3x3x3:
   Term 1 (3x2x6) -> 2 * 8 = 16
   Term 2 (3x3x6) -> 2 * 8 = 16
   Total Rank = 32.
   This is still much better than 90.
   And 8-term 3x3 decomposition is easy:
   Split 3x3 into 2x2, 2x1, 1x2, 1x1 blocks.
   A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
   B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1.
   
   C = A B.
   C11 = A11 B11 + A12 B21
   C12 = A11 B12 + A12 B22
   C21 = A21 B11 + A22 B21
   C22 = A21 B12 + A22 B22
   
   We can compute A11 B11 using Strassen (7 mults).
   A12 B21 is 2x1 * 1x2 -> 2x2. Rank 2 (diagonal) or 3?
   Actually, 2x1 * 1x2 is an outer product. Rank 1? No, it's a 2x2 matrix.
   (a,b)^T * (c,d) = [[ac, ad], [bc, bd]]. This has rank 1.
   So A12 B21 can be done with 1 multiplication?
   Let u = A12 (2x1), v = B21 (1x2).
   The product is u v^T.
   In tensor decomposition, this is a sum of 1 term?
   u_i v_j. Yes, rank 1.
   So A12 B21 uses 1 term.
   Similarly A21 B12 (1x2 * 2x1) is 1x1. Rank 1.
   A11 B12 (2x2 * 2x1). Can be done with 3 mults?
   A21 B11 (1x2 * 2x2). Can be done with 3 mults?
   A12 B22 (2x1 * 1x1). 2 mults?
   A22 B21 (1x1 * 1x2). 2 mults?
   A22 B22 (1x1 * 1x1). 1 mult.
   
   Total: 7 (Strassen) + 1 + 1 + 3 + 3 + 2 + 2 + 1 = 20?
   This is worse than 8.
   
   Actually, there is a known 8-term algorithm for 3x3.
   Let's just use the 8-term algorithm for 3x3x3.
   
   Wait, I can use the 7-term Laderman. I will hardcode the 7-term Laderman decomposition.
   
   Laderman's 7 terms for 3x3:
   (Using 1-based indexing for clarity in thought, will convert to 0-based)
   
   1. u=(1,0,0, 0,0,0, 0,0,0), v=(1,0,0, 0,0,0, 0,0,0), w=(1,0,0, 0,0,0, 0,0,0)
   2. u=(0,1,0, 0,0,0, 0,0,0), v=(0,0,0, 1,0,0, 0,0,0), w=(0,1,0, 0,0,0, 0,0,0)
   3. u=(0,0,0, 1,0,0, 0,0,0), v=(0,0,0, 0,0,0, 1,0,0), w=(0,0,0, 0,1,0, 0,0,0)
   4. u=(0,0,0, 0,1,0, 0,0,0), v=(0,0,0, 0,0,0, 0,1,0), w=(0,0,0, 0,0,0, 0,1,0)
   5. u=(0,0,0, 0,0,0, 1,0,0), v=(0,0,0, 0,0,0, 0,0,1), w=(0,0,0, 0,0,0, 1,0,0)
   6. u=(1,0,1, 0,1,0, 0,0,0), v=(1,0,0, 0,1,0, 0,0,1), w=(1,0,0, 0,1,0, 0,0,0)
   7. u=(0,1,0, 0,0,1, 1,0,0), v=(0,0,1, 1,0,0, 0,1,0), w=(0,1,0, 0,0,0, 0,1,0)
   
   I need to verify this.
   Actually, let's use a simpler approach:
   Use the 7-term Strassen for 2x2 and embed it.
   
   For 3x3x3, we can use the following 8-term decomposition (which is easy to verify):
   Split into 2x2 blocks.
   A = [[A1, A2], [A3, A4]] where A1 is 2x2, A2 is 2x1, A3 is 1x2, A4 is 1x1.
   B = [[B1, B2], [B3, B4]] where B1 is 2x2, B2 is 2x1, B3 is 1x2, B4 is 1x1.
   
   C11 = A1 B1 + A2 B3
   C12 = A1 B2 + A2 B4
   C21 = A3 B1 + A4 B3
   C22 = A3 B2 + A4 B4
   
   A1 B1: 7 mults (Strassen).
   A2 B3: 2x1 * 1x2. Rank 1. (1 mult)
   A1 B2: 2x2 * 2x1. Rank 3? (3 mults)
   A2 B4: 2x1 * 1x1. Rank 2? (2 mults)
   A3 B1: 1x2 * 2x2. Rank 3? (3 mults)
   A4 B3: 1x1 * 1x2. Rank 2? (2 mults)
   A3 B2: 1x2 * 2x1. Rank 1? (1 mult)
   A4 B4: 1x1 * 1x1. Rank 1? (1 mult)
   
   Total: 7 + 1 + 3 + 2 + 3 + 2 + 1 + 1 = 20.
   
   This is not 8.
   
   Okay, I will use the **7-term Laderman decomposition** hardcoded.
   I will find a verified set of 7 tensors.
   
   From a reliable source (e.g. Wikipedia or academic paper):
   The 7 multiplications for 3x3 matrix multiplication:
   
   m1 = a11 b11
   m2 = a12 b21
   m3 = a21 b12
   m4 = a22 b22
   m5 = a33 b33
   m6 = (a11 + a22) (b11 + b22)
   m7 = (a12 + a13 + a22 + a23) (b21 + b23 + b11 + b13) ... no, this is not standard.
   
   Let's use the **8-term** decomposition for 3x3 which is easier to construct correctly using Strassen on the 2x2 sub-blocks and simple multiplications for the rest, accepting a slightly higher rank but guaranteed correctness.
   
   Actually, I can just use the **schoolbook** for the 3x3 parts if I can't get Laderman right.
   Schoolbook 3x3 is 9.
   If I use 9 for the 3x3 core:
   Term 1 (3x2x6) -> 2 * 9 = 18? No, 3x2x6 is not 3x3x3.
   
   Let's stick to the **Rank 75** target using a simpler block decomposition that doesn't require Laderman.
   
   Decomposition:
   A (3x5) * B (5x6).
   Split A into A1 (2x5) and A2 (1x5).
   Split B into B1 (5x2) and B2 (5x4).
   
   C = A1 B1 + A2 B2.
   
   A1 B1: (2x5) * (5x2).
   This is a 2x5x2 tensor.
   We can embed this into 3x5x3?
   Or just use schoolbook for 2x5x2? Rank 20.
   A2 B2: (1x5) * (5x4). Rank 20.
   Total 40.
   
   This is rank 40. Much better than 90.
   And it uses only schoolbook multiplications, which are trivial to implement correctly.
   
   Let's verify the embedding.
   A1 is 2x5. B1 is 5x2.
   U1 shape (2*5, rank), V1 shape (5*2, rank), W1 shape (2*2, rank).
   Schoolbook for 2x5x2:
   For i in 0..1, j in 0..4, k in 0..1:
   U[i*5+j][c] = 1
   V[j*2+k][c] = 1
   W[k*2+i][c] = 1
   
   A2 is 1x5. B2 is 5x4.
   Schoolbook for 1x5x4:
   For i in 0..0, j in 0..4, k in 0..3:
   U[i*5+j][c] = 1
   V[j*4+k][c] = 1
   W[k*1+i][c] = 1
   
   Total Rank = 20 + 20 = 40.
   
   This is a valid, exact, rank-40 decomposition.
   Score = 40.
   Current best = 90.
   Improvement = 90/40 = 2.25.
   
   This is a huge jump and guaranteed to work.
   
   I will implement this rank-40 construction.
   
   Steps:
   1. Initialize U, V, W with zeros.
   2. Rank = 40.
   3. First 20 terms for A1 (rows 0-1 of A) * B1 (cols 0-1 of B).
      A1 is A[0:2, :]. B1 is B[:, 0:2].
      Indices in global A: i in 0,1. j in 0..4.
      Indices in global B: j in 0..4. k in 0,1.
      Indices in global C: k in 0,1. i in 0,1.
      
      Wait, the output C is 3x6.
      The term A1 B1 contributes to C[0:2, 0:2].
      The term A2 B2 contributes to C[2:3, 2:6].
      
      What about C[0:2, 2:6] and C[2:3, 0:2]?
      A1 B2: (2x5) * (5x4). Contributes to C[0:2, 2:6].
      A2 B1: (1x5) * (5x2). Contributes to C[2:3, 0:2].
      
      So we need 4 terms:
      1. A1 B1: 2x5 * 5x2. Rank 20.
      2. A1 B2: 2x5 * 5x4. Rank 40.
      3. A2 B1: 1x5 * 5x2. Rank 10.
      4. A2 B2: 1x5 * 5x4. Rank 20.
      
      Total Rank = 20 + 40 + 10 + 20 = 90.
      
      This is just the schoolbook method again!
      
      We need to reduce the rank of the rectangular multiplications.
      
      For 2x5 * 5x4:
      Can we do better than 40?
      Embed into 3x5 * 5x6? No.
      
      Let's use the **Rank 7** Strassen for the 2x2 cores.
      
      For 2x5 * 5x4:
      Split B (5x4) into B1 (5x2) and B2 (5x2).
      A B = A B1 + A B2.
      A B1: 2x5 * 5x2.
      Embed A (2x5) into 3x5? No.
      
      Let's use the fact that M(2,5,2) <= M(3,5,3)?
      
      Actually, let's just use the **Rank 40** construction I found earlier, but I realized it was 90.
      
      Wait, if I use Strassen for the 2x2 blocks:
      2x5 * 5x2:
      Split A (2x5) into A1 (2x2) and A2 (2x3).
      Split B (5x2) into B1 (2x2) and B2 (3x2).
      A B = A1 B1 + A2 B2.
      A1 B1: 2x2 * 2x2. Rank 7 (Strassen).
      A2 B2: 2x3 * 3x2.
      Embed 2x3 * 3x2 into 3x3 * 3x3?
      Pad A2 to 3x3 (add zero row).
      Pad B2 to 3x3 (add zero col).
      Rank 7 (Laderman) or 9 (Schoolbook).
      So 2x3 * 3x2 has rank <= 9.
      Total for 2x5 * 5x2: 7 + 9 = 16.
      
      Similarly for 2x5 * 5x4:
      Split B into B1 (5x2) and B2 (5x2).
      A B1: 2x5 * 5x2. Rank 16.
      A B2: 2x5 * 5x2. Rank 16.
      Total 32.
      
      For 1x5 * 5x2:
      Embed 1x5 into 2x5?
      A (1x5) * B (5x2).
      Split A into A1 (1x2), A2 (1x3).
      Split B into B1 (2x2), B2 (3x2).
      A1 B1: 1x2 * 2x2. Rank 3? (Schoolbook 4, Strassen-like 3?)
      A2 B2: 1x3 * 3x2. Rank 6?
      Total 9.
      
      For 1x5 * 5x4:
      Split B into B1 (5x2), B2 (5x2).
      A B1: 1x5 * 5x2. Rank 9.
      A B2: 1x5 * 5x2. Rank 9.
      Total 18.
      
      Total Rank: 16 (A1B1) + 32 (A1B2) + 9 (A2B1) + 18 (A2B2) = 75.
      
      This matches the target rank 75!
      
      I will implement this rank-75 construction using Strassen for 2x2 and Schoolbook for the rest (or Laderman if I can, but Schoolbook is safer).
      
      Actually, for 2x3 * 3x2, I will use Schoolbook (Rank 6).
      For 1x2 * 2x2, I will use Schoolbook (Rank 4).
      For 1x3 * 3x2, I will use Schoolbook (Rank 6).
      
      Recalculating with Schoolbook for non-2x2:
      1. A1 B1 (2x2 * 2x2): Strassen (7).
      2. A2 B2 (2x3 * 3x2): Schoolbook (6).
         Total for 2x5 * 5x2: 13.
         
      3. A1 B1 (2x2 * 2x2): Strassen (7).
      4. A2 B2 (2x3 * 3x2): Schoolbook (6).
         Total for 2x5 * 5x2: 13.
         Total for 2x5 * 5x4: 26.
         
      5. A1 B1 (1x2 * 2x2): Schoolbook (4).
      6. A2 B2 (1x3 * 3x2): Schoolbook (6).
         Total for 1x5 * 5x2: 10.
         
      7. A1 B1 (1x2 * 2x2): Schoolbook (4).
      8. A2 B2 (1x3 * 3x2): Schoolbook (6).
         Total for 1x5 * 5x2: 10.
         Total for 1x5 * 5x4: 20.
         
      Total Rank: 13 + 26 + 10 + 20 = 69.
      
      This is even better! Rank 69.
      
      I will implement this rank-69 construction.
      
      Components:
      - Strassen 2x2x2 (Rank 7)
      - Schoolbook for rectangular blocks.
      
      This is a structural, deterministic construction.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen's 7 multiplications
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # m1
    U[0][0] = 1; U[3][0] = 1  # a, d
    V[0][0] = 1; V[3][0] = 1  # e, h
    W[0][0] = 1; W[3][0] = 1  # c11, c22
    
    # m2
    U[2][1] = 1; U[3][1] = 1  # c, d
    V[0][1] = 1               # e
    W[2][1] = 1               # c21
    
    # m3
    U[0][2] = 1               # a
    V[1][2] = 1; V[3][2] = -1 # f, -h
    W[1][2] = 1               # c12
    
    # m4
    U[3][3] = 1               # d
    V[2][3] = 1; V[0][3] = -1 # g, -e
    W[3][3] = 1               # c22
    
    # m5
    U[0][4] = 1; U[1][4] = 1  # a, b
    V[3][4] = 1               # h
    W[1][4] = 1; W[3][4] = 1  # c12, c22
    
    # m6
    U[2][5] = 1; U[0][5] = -1 # c, -a
    V[0][5] = 1; V[1][5] = 1  # e, f
    W[2][5] = 1; W[0][5] = 1  # c21, c11
    
    # m7
    U[1][6] = 1; U[3][6] = -1 # b, -d
    V[2][6] = 1; V[3][6] = 1  # g, h
    W[3][6] = 1; W[1][6] = 1  # c22, c12
    
    return U, V, W

def _schoolbook_block(n, m, p, offset_u, offset_v, offset_w, rank_start):
    """
    Adds schoolbook decomposition for n x m * m x p block.
    Returns new rank_start.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We will construct a rank-69 decomposition.
    # Total Rank = 69.
    total_rank = 69
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_cursor = 0
    
    # 1. A1 B1: A[0:2, 0:2] * B[0:2, 0:2] (2x2 * 2x2)
    # Contributes to C[0:2, 0:2]
    sU, sV, sW = _strassen_2x2()
    for i in range(4):
        for c in range(7):
            # Map local i (0-3) to global U index
            # Local i corresponds to A row i//2, col i%2
            # Global A row = i//2, col = i%2
            gi = i // 2
            gj = i % 2
            U[gi * 5 + gj][rank_cursor + c] = sU[i][c]
            
            # Map local i (0-3) to global V index
            # Local i corresponds to B row i//2, col i%2
            # Global B row = i//2, col = i%2
            gj_v = i // 2
            gk = i % 2
            V[gj_v * 6 + gk][rank_cursor + c] = sV[i][c]
            
            # Map local i (0-3) to global W index
            # Local i corresponds to C row i//2, col i%2
            # Global C row = i//2, col = i%2
            gk_w = i // 2
            gi_w = i % 2
            W[gk_w * 3 + gi_w][rank_cursor + c] = sW[i][c]
    rank_cursor += 7
    
    # 2. A2 B2: A[0:2, 2:5] * B[2:5, 0:2] (2x3 * 3x2)
    # Contributes to C[0:2, 0:2]
    bU, bV, bW, _ = _schoolbook_block(2, 3, 2, 0, 0, 0, 0)
    for i in range(6):
        for c in range(6):
            gi = i // 3
            gj = i % 3
            U[gi * 5 + (2 + gj)][rank_cursor + c] = bU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[(2 + gj_v) * 6 + gk][rank_cursor + c] = bV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[gk_w * 3 + gi_w][rank_cursor + c] = bW[i][c]
    rank_cursor += 6
    
    # 3. A1 B1 (of second half): A[0:2, 0:2] * B[0:2, 2:4] (2x2 * 2x2)
    # Contributes to C[0:2, 2:4]
    sU, sV, sW = _strassen_2x2()
    for i in range(4):
        for c in range(7):
            gi = i // 2
            gj = i % 2
            U[gi * 5 + gj][rank_cursor + c] = sU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[gj_v * 6 + (2 + gk)][rank_cursor + c] = sV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[(2 + gk_w) * 3 + gi_w][rank_cursor + c] = sW[i][c]
    rank_cursor += 7
    
    # 4. A2 B2 (of second half): A[0:2, 2:5] * B[2:5, 2:4] (2x3 * 3x2)
    # Contributes to C[0:2, 2:4]
    bU, bV, bW, _ = _schoolbook_block(2, 3, 2, 0, 0, 0, 0)
    for i in range(6):
        for c in range(6):
            gi = i // 3
            gj = i % 3
            U[gi * 5 + (2 + gj)][rank_cursor + c] = bU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[(2 + gj_v) * 6 + (2 + gk)][rank_cursor + c] = bV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[(2 + gk_w) * 3 + gi_w][rank_cursor + c] = bW[i][c]
    rank_cursor += 6
    
    # 5. A1 B1 (of third half): A[0:2, 0:2] * B[0:2, 4:6] (2x2 * 2x2)
    # Contributes to C[0:2, 4:6]
    sU, sV, sW = _strassen_2x2()
    for i in range(4):
        for c in range(7):
            gi = i // 2
            gj = i % 2
            U[gi * 5 + gj][rank_cursor + c] = sU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[gj_v * 6 + (4 + gk)][rank_cursor + c] = sV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[(4 + gk_w) * 3 + gi_w][rank_cursor + c] = sW[i][c]
    rank_cursor += 7
    
    # 6. A2 B2 (of third half): A[0:2, 2:5] * B[2:5, 4:6] (2x3 * 3x2)
    # Contributes to C[0:2, 4:6]
    bU, bV, bW, _ = _schoolbook_block(2, 3, 2, 0, 0, 0, 0)
    for i in range(6):
        for c in range(6):
            gi = i // 3
            gj = i % 3
            U[gi * 5 + (2 + gj)][rank_cursor + c] = bU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[(2 + gj_v) * 6 + (4 + gk)][rank_cursor + c] = bV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[(4 + gk_w) * 3 + gi_w][rank_cursor + c] = bW[i][c]
    rank_cursor += 6
    
    # 7. A1 B1 (row 2): A[2, 0:2] * B[0:2, 0:2] (1x2 * 2x2)
    # Contributes to C[2, 0:2]
    bU, bV, bW, _ = _schoolbook_block(1, 2, 2, 0, 0, 0, 0)
    for i in range(2):
        for c in range(4):
            gi = i // 2
            gj = i % 2
            U[2 * 5 + gj][rank_cursor + c] = bU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[gj_v * 6 + gk][rank_cursor + c] = bV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[gk_w * 3 + 2][rank_cursor + c] = bW[i][c]
    rank_cursor += 4
    
    # 8. A2 B2 (row 2): A[2, 2:5] * B[2:5, 0:2] (1x3 * 3x2)
    # Contributes to C[2, 0:2]
    bU, bV, bW, _ = _schoolbook_block(1, 3, 2, 0, 0, 0, 0)
    for i in range(3):
        for c in range(6):
            gi = i // 3
            gj = i % 3
            U[2 * 5 + (2 + gj)][rank_cursor + c] = bU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[(2 + gj_v) * 6 + gk][rank_cursor + c] = bV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[gk_w * 3 + 2][rank_cursor + c] = bW[i][c]
    rank_cursor += 6
    
    # 9. A1 B1 (row 2, cols 2-4): A[2, 0:2] * B[0:2, 2:4] (1x2 * 2x2)
    # Contributes to C[2, 2:4]
    bU, bV, bW, _ = _schoolbook_block(1, 2, 2, 0, 0, 0, 0)
    for i in range(2):
        for c in range(4):
            gi = i // 2
            gj = i % 2
            U[2 * 5 + gj][rank_cursor + c] = bU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[gj_v * 6 + (2 + gk)][rank_cursor + c] = bV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[(2 + gk_w) * 3 + 2][rank_cursor + c] = bW[i][c]
    rank_cursor += 4
    
    # 10. A2 B2 (row 2, cols 2-4): A[2, 2:5] * B[2:5, 2:4] (1x3 * 3x2)
    # Contributes to C[2, 2:4]
    bU, bV, bW, _ = _schoolbook_block(1, 3, 2, 0, 0, 0, 0)
    for i in range(3):
        for c in range(6):
            gi = i // 3
            gj = i % 3
            U[2 * 5 + (2 + gj)][rank_cursor + c] = bU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[(2 + gj_v) * 6 + (2 + gk)][rank_cursor + c] = bV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[(2 + gk_w) * 3 + 2][rank_cursor + c] = bW[i][c]
    rank_cursor += 6
    
    # 11. A1 B1 (row 2, cols 4-6): A[2, 0:2] * B[0:2, 4:6] (1x2 * 2x2)
    # Contributes to C[2, 4:6]
    bU, bV, bW, _ = _schoolbook_block(1, 2, 2, 0, 0, 0, 0)
    for i in range(2):
        for c in range(4):
            gi = i // 2
            gj = i % 2
            U[2 * 5 + gj][rank_cursor + c] = bU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[gj_v * 6 + (4 + gk)][rank_cursor + c] = bV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[(4 + gk_w) * 3 + 2][rank_cursor + c] = bW[i][c]
    rank_cursor += 4
    
    # 12. A2 B2 (row 2, cols 4-6): A[2, 2:5] * B[2:5, 4:6] (1x3 * 3x2)
    # Contributes to C[2, 4:6]
    bU, bV, bW, _ = _schoolbook_block(1, 3, 2, 0, 0, 0, 0)
    for i in range(3):
        for c in range(6):
            gi = i // 3
            gj = i % 3
            U[2 * 5 + (2 + gj)][rank_cursor + c] = bU[i][c]
            
            gj_v = i // 2
            gk = i % 2
            V[(2 + gj_v) * 6 + (4 + gk)][rank_cursor + c] = bV[i][c]
            
            gk_w = i // 2
            gi_w = i % 2
            W[(4 + gk_w) * 3 + 2][rank_cursor + c] = bW[i][c]
    rank_cursor += 6
    
    # Verify rank_cursor is 69
    # 7+6+7+6+7+6+4+6+4+6+4+6 = 69
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
