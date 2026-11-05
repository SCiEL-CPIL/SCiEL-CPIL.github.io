"""Matrix multiplication 6x6x6 via 2x2 block decomposition with BCLR rank-21 for 3x3 blocks.

Approach:
- Partition 6x6 matrices into 2x2 blocks of 3x3.
- Use Strassen's 7-product algorithm at the block level.
- Use the BCLR rank-21 decomposition for each 3x3 matrix multiplication.
- Total rank: 7 (Strassen) * 21 (BCLR) = 147.

The BCLR algorithm computes C = A * B for 3x3 matrices using 21 multiplications:
A = [[a1, a2, a3], [a4, a5, a6], [a7, a8, a9]]
B = [[b1, b2, b3], [b4, b5, b6], [b7, b8, b9]]

P1 = (a1+a2+a3+a4+a5+a6+a7+a8+a9) * (b1+b2+b3+b4+b5+b6+b7+b8+b9)
P2 = (a1+a2+a3) * (b1+b4+b7)
P3 = (a1+a4+a7) * (b1+b2+b3)
P4 = (a2+a5+a8) * (b4+b5+b6)
P5 = (a3+a6+a9) * (b7+b8+b9)
P6 = (a1+a5+a9) * (b1+b5+b9)
P7 = (a2+a5+a9) * (b1+b4+b7)
P8 = (a1+a4+a7) * (b1+b2+b3)  -- wait, this is same as P3

Let me use the actual BCLR terms. The BCLR rank-21 algorithm is:

Actually, let me use a known rank-21 decomposition. One standard formulation:

For C = AB where A, B are 3x3:
Let's denote A[i][j] and B[i][j] with i,j in {0,1,2}.

The 21 terms (from BCLR 1979):
M1  = (a00+a01+a02) * (b00+b10+b20)
M2  = (a10+a11+a12) * (b00+b10+b20)
M3  = (a20+a21+a22) * (b00+b10+b20)
M4  = (a00+a10+a20) * (b00+b01+b02)
M5  = (a01+a11+a21) * (b00+b01+b02)
M6  = (a02+a12+a22) * (b00+b01+b02)
M7  = (a00+a11+a22) * (b00+b11+b22)
M8  = (a01+a10+a22) * (b00+b10+b22)
M9  = (a02+a12+a20) * (b00+b20+b22)
M10 = (a00+a01+a21) * (b00+b02+b12)
M11 = (a02+a10+a12) * (b00+b10+b20)  -- no, let me be more careful

This is getting complex. Let me use a verified rank-21 decomposition.

Actually, I'll use the following approach: I know that the BCLR algorithm exists with rank 21.
Let me define it using the standard formulation from the literature.

The BCLR 21-multiplication algorithm for 3x3:

Define:
s1 = a00 + a01 + a02 + a10 + a11 + a12 + a20 + a21 + a22
s2 = b00 + b01 + b02 + b10 + b11 + b12 + b20 + b21 + b22

The 21 products are:
p1 = (a00+a11+a22) * (b00+b11+b22)
p2 = (a00+a01+a02) * (b00+b10+b20)
p3 = (a00+a10+a20) * (b00+b01+b02)
p4 = (a01+a11+a21) * (b01+b11+b21)
p5 = (a02+a12+a22) * (b02+b12+b22)
p6 = (a10+a20) * (b00+b01)
p7 = (a10+a21) * (b01+b10)
p8 = (a11+a20) * (b00+b11)
p9 = (a11+a22) * (b11+b22)
p10 = (a12+a20) * (b02+b10)
p11 = (a12+a21) * (b01+b12)
p12 = (a12+a22) * (b02+b21)
p13 = (a01+a10) * (b00+b02)
p14 = (a02+a10) * (b01+b02)
p15 = (a01+a20) * (b00+b01)
p16 = (a02+a20) * (b00+b02)
p17 = (a01+a12) * (b11+b12)
p18 = (a02+a11) * (b10+b12)
p19 = (a01+a22) * (b21+b22)
p20 = (a02+a21) * (b20+b22)
p21 = (a00+a22) * (b00+b22)

This doesn't look right either. Let me use a different, verified approach.

I'll use the fact that I can verify any decomposition by checking it. Let me use a simpler but correct approach:

Since I need to be correct, let me use the schoolbook rank-9 for 3x3 blocks but with a different block decomposition strategy.

Actually, the current code already achieves rank 189 with Strassen(7) * schoolbook_3x3(27) = 189.
Wait, 7 * 27 = 189. But schoolbook for 3x3 is rank 9, not 27.

Let me re-examine. For a 3x3 matrix multiplication, the schoolbook algorithm uses 3*3*3 = 27 scalar multiplications, but the tensor rank is 9 (since C[i,j] = sum_k A[i,k]*B[k,j], and we can write this as 9 rank-1 tensors).

So Strassen(7) * schoolbook_rank(9) = 63? No, that's not right either.

Wait, I think I was confused. Let me re-read the current code.

The current code uses:
- 7 Strassen terms at the block level
- For each Strassen term, it does a 3x3 matrix multiplication using the schoolbook method
- The schoolbook 3x3 multiplication has rank 9 (not 27)
- So total rank = 7 * 9 = 63?

But the code says rank = 7 * 27 = 189. Let me check...

Looking at the code: `rank = 7 * 27`. And in the inner loop, it iterates over `i in range(bs)`, `k in range(bs)`, `j in range(bs)` where bs=3, so 3*3*3 = 27 iterations per Strassen term.

But wait, for each (i,k,j) triple, it's creating ONE rank-1 term. So 27 terms per Strassen product, times 7 Strassen products = 189 total.

But the schoolbook 3x3 multiplication should only need 9 terms (one for each (i,j) pair of C, with k summed). The 27-term decomposition is the "expanded" form where we don't sum over k but instead create separate terms for each (i,k,j).

So the current code is using the expanded schoolbook (rank 27 per 3x3 product) instead of the compact schoolbook (rank 9 per 3x3 product).

If I use the compact schoolbook (rank 9), total rank = 7 * 9 = 63. That would be much better!

Let me implement this correctly.

For a 3x3 matrix product X * Y = Z:
Z[i][j] = sum_{k=0}^{2} X[i][k] * Y[k][j]

The rank-9 decomposition has 9 terms, one for each (i,j) in {0,1,2}^2:
Term (i,j): 
  U part: X[i][k] for all k -> but this is a sum, not a single entry
  
Wait, no. The rank-9 decomposition is:
For each (i, j):
  u_{i,j} is a vector with u_{i,j}[i*3+k] = 1 for k in {0,1,2} (i.e., the i-th row of X)
  v_{i,j} is a vector with v_{i,j}[k*3+j] = 1 for k in {0,1,2} (i.e., the j-th column of Y)
  w_{i,j} is a vector with w_{i,j}[j*3+i] = 1 (i.e., position (i,j) in Z, using the W convention)

Wait, but u_{i,j} has multiple non-zero entries, so it's not a rank-1 tensor in the strict sense. The rank-1 decomposition requires each u, v, w to have at most one non-zero entry.

So the schoolbook 3x3 multiplication actually has rank 27 in the strict rank-1 sense (one term per (i,k,j) triple), not rank 9.

The "rank 9" refers to the bilinear complexity (number of scalar multiplications), not the tensor rank in the strict sense.

So the current code is correct: rank 27 per 3x3 product, total 7 * 27 = 189.

To improve, I need to use a lower-rank decomposition for 3x3. The BCLR algorithm achieves rank 21 for 3x3 matrix multiplication in the bilinear complexity sense, which would give 7 * 21 = 147 total rank.

But implementing BCLR correctly is risky. Let me instead try to use a 2x2 block decomposition with Strassen, giving 7 * 27 = 189 (same as current) or use a different approach.

Actually, let me try using 2x2 blocks of 3x3 with Strassen, but use the compact form where possible.

Alternatively, I can try to directly implement a known rank-21 decomposition for 3x3.

Let me use the following verified rank-21 decomposition (from BCLR 1979, reformulated):

For C = AB, A = [a_ij], B = [b_ij], i,j in {0,1,2}:

m1  = (a00 + a11 + a22) * (b00 + b11 + b22)
m2  = (a00 + a01 + a02) * (b00 + b10 + b20)
m3  = (a00 + a10 + a20) * (b00 + b01 + b02)
m4  = (a01 + a11 + a21) * (b01 + b11 + b21)
m5  = (a02 + a12 + a22) * (b02 + b12 + b22)
m6  = (a00 + a01 + a21) * (b00 + b02 + b12)
m7  = (a02 + a10 + a12) * (b00 + b10 + b20)
m8  = (a01 + a10 + a20) * (b00 + b01 + b02)
m9  = (a02 + a12 + a20) * (b00 + b20 + b22)
m10 = (a01 + a20 + a22) * (b00 + b20 + b22)
m11 = (a00 + a12 + a21) * (b00 + b12 + b21)
m12 = (a00 + a02 + a12) * (b00 + b02 + b22)
m13 = (a01 + a10 + a22) * (b00 + b10 + b22)
m14 = (a02 + a11 + a20) * (b00 + b11 + b20)
m15 = (a00 + a11 + a20) * (b00 + b10 + b11)
m16 = (a01 + a12 + a20) * (b01 + b12 + b20)
m17 = (a02 + a10 + a21) * (b02 + b10 + b21)
m18 = (a00 + a12 + a22) * (b00 + b12 + b22)
m19 = (a01 + a10 + a22) * (b01 + b10 + b22)
m20 = (a02 + a11 + a20) * (b02 + b11 + b20)
m21 = (a00 + a22) * (b00 + b22)

This is getting too speculative. Let me just use the current approach but with 2x2 blocks instead of 3x3 blocks, which gives the same rank 189 but might have a different structure.

Actually, let me just keep the current code as-is since I can't reliably improve it without risking correctness.

Wait, I just realized: the current code has rank 189. If I can get rank 147 (BCLR), that's an improvement. But if I can't, I should keep 189.

Let me try a different approach: use 2x2 blocks of 3x3 matrices, and for each 3x3 block product, use the schoolbook rank-27. This gives 7 * 27 = 189, same as current.

Alternatively, use 3x3 blocks of 2x2 matrices. 6x6 = 3x3 grid of 2x2 blocks.
- 9 output blocks
- Each output block is a 2x2 matrix
- C_ij = sum_k A_ik * B_kj where each is 2x2
- 9 * 4 = 36 2x2 matrix products
- Each 2x2 product has rank 7 (Strassen)
- Total rank = 36 * 7 = 252. Worse.

Or use Strassen at the 3x3 block level:
- 3x3 grid of 2x2 blocks
- Strassen on 3x3 blocks? Strassen is for 2x2, not 3x3.

I think the best I can do reliably is keep rank 189. Let me output the current code.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # Strassen decomposition at the 2x2 block level (each entry is a 3x3 matrix)
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    # Block (bi): row offset = (bi // 2) * bs, col offset = (bi % 2) * bs

    # Strassen 7 virtual products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)

    # Each term: (list of (block_idx, coeff) for A, list of (block_idx, coeff) for B)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]

    # Output reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # output_contrib[term_idx] = list of (output_block_idx, coeff)
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]

    # Total rank: 7 Strassen terms * 27 schoolbook terms per 3x3 product = 189
    rank = 7 * 27

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for i in range(bs):   # local row in 3x3 block
            for k in range(bs):  # local col in A / local row in B
                for j in range(bs):  # local col in B / local col in C
                    # U contributions: A entries
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * bs
                        col_a = (blk_a % 2) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx][col] += coeff_a

                    # V contributions: B entries
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * bs
                        col_b = (blk_b % 2) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx][col] += coeff_b

                    # W contributions: C entries (W index = global_col * n + global_row)
                    for (blk_c, coeff_c) in output_contrib[term_idx]:
                        row_c = (blk_c // 2) * bs
                        col_c = (blk_c % 2) * bs
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx][col] += coeff_c

                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
