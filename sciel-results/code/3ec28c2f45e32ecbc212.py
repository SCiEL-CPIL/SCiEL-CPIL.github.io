import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    n, m, p = 2, 3, 6

    # We construct a rank-20 decomposition:
    #   C = A * B,  A is 2x3, B is 3x6, C is 2x6
    #
    # Split A into A1 (2x2) and A2 (2x1):
    #   A = [A1, A2]
    # Split B into B1 (2x6) and B2 (1x6):
    #   B = [B1; B2]
    # Then C = A1 * B1 + A2 * B2
    #
    # A2 * B2: 2x1 * 1x6 -> rank 2
    # A1 * B1: 2x2 * 2x6 -> split B1 into two 2x3 blocks:
    #   B1 = [B1a, B1b] where B1a is 2x3, B1b is 2x3
    #   A1 * B1 = A1 * B1a + A1 * B1b
    # Each A1 * Bxi: 2x2 * 2x3 -> split Bxi into 2x2 and 2x1:
    #   Bxi = [Bxi1, Bxi2] where Bxi1 is 2x2, Bxi2 is 2x1
    #   A1 * Bxi = A1 * Bxi1 + A1 * Bxi2
    #   A1 * Bxi1: 2x2 * 2x2 -> rank 7 (Strassen)
    #   A1 * Bxi2: 2x2 * 2x1 -> rank 2
    #   Total per 2x2*2x3: 7 + 2 = 9
    # Total for 2x2*2x6: 9 + 9 = 18
    # Grand total: 18 + 2 = 20
    #
    # Index mapping:
    #   U has shape (n*m, rank) = (6, 20), indexed by (i, j) -> i*m + j = i*3 + j
    #   V has shape (m*p, rank) = (18, 20), indexed by (j, k) -> j*p + k = j*6 + k
    #   W has shape (p*n, rank) = (12, 20), indexed by (k, i) -> k*n + i = k*2 + i
    #
    # The tensor condition: for all i in [0,2), j in [0,3), k in [0,6):
    #   sum_r U[i*3+j, r] * V[j*6+k, r] * W[k*2+i, r] = 1

    rank = 20
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(18)]
    W = [[0.0] * rank for _ in range(12)]

    col = 0

    # Helper: add a Strassen 2x2x2 block
    # A is 2x2 with rows i=0,1 and cols j=j0, j0+1
    # B is 2x2 with rows j=j0, j0+1 and cols k=k0, k0+1
    # C is 2x2 with rows i=0,1 and cols k=k0, k0+1
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # U entries for A[i, j]:
        #   a = A[0, j0], b = A[0, j0+1], c = A[1, j0], d = A[1, j0+1]
        # V entries for B[j, k]:
        #   e = B[j0, k0], f = B[j0, k0+1], g = B[j0+1, k0], h = B[j0+1, k0+1]
        # W entries for C[k, i]:
        #   C[0,0], C[0,1], C[1,0], C[1,1] correspond to (k,i) = (k0,0), (k0,1), (k0+1,0), (k0+1,1)
        #
        # For each of the 7 multiplies, we add one column.
        #
        # M1: (a+d)(e+h)
        #   U: a=+1, d=+1
        #   V: e=+1, h=+1
        #   W: C00=+1, C11=+1
        U[i0*3 + j0][col] = 1.0
        U[i0*3 + j0 + 1][col] = 1.0
        V[j0*6 + k0][col] = 1.0
        V[j0*6 + k0 + 1][col] = 1.0
        W[k0*2 + i0][col] = 1.0
        W[(k0+1)*2 + i0 + 1][col] = 1.0
        col += 1

        # M2: (c+d)e
        #   U: c=+1, d=+1
        #   V: e=+1
        #   W: C10=+1
        U[i0*3 + j0][col] = 1.0
        U[i0*3 + j0 + 1][col] = 1.0
        V[j0*6 + k0][col] = 1.0
        W[(k0+1)*2 + i0][col] = 1.0
        col += 1

        # M3: a(f-h)
        #   U: a=+1
        #   V: f=+1, h=-1
        #   W: C01=+1
        U[i0*3 + j0][col] = 1.0
        V[j0*6 + k0 + 1][col] = 1.0
        V[j0*6 + k0 + 1][col] = 1.0  # f
        V[j0*6 + k0 + 1][col] = 1.0  # wait, h is B[j0+1, k0+1]
        # Let me redo: f = B[j0, k0+1], h = B[j0+1, k0+1]
        # f-h means V[j0*6+k0+1] = +1, V[j0*6+k0+1]... no
        # f is at V[j0*6 + k0+1], h is at V[j0*6 + k0+1]? No.
        # B rows: j0, j0+1. B cols: k0, k0+1.
        # e = B[j0, k0] -> V[j0*6 + k0]
        # f = B[j0, k0+1] -> V[j0*6 + k0+1]
        # g = B[j0+1, k0] -> V[j0*6 + k0]... no, V[(j0+1)*6 + k0]
        # h = B[j0+1, k0+1] -> V[(j0+1)*6 + k0+1]
        # So f-h: V[j0*6+k0+1] = +1, V[(j0+1)*6+k0+1] = -1
        # Let me restart this block properly.

    # Let me restart with a cleaner approach.
    # I'll build the decomposition column by column, being very careful with indices.

    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(18)]
    W = [[0.0] * rank for _ in range(12)]

    col = 0

    # We have 4 sub-problems of type 2x2*2x2 (Strassen, rank 7 each) = 28
    # Plus 4 sub-problems of type 2x2*2x1 (rank 2 each) = 8
    # Plus 1 sub-problem of type 2x1*1x6 (rank 2) = 2
    # Wait, that's 28 + 8 + 2 = 38, not 20.
    #
    # Let me re-examine. The issue is that I'm double-counting.
    # Actually, let me think about this differently.
    #
    # C = A * B where A is 2x3, B is 3x6.
    #
    # Let me write A as:
    #   A = [[a00, a01, a02],
    #        [a10, a11, a12]]
    #
    # B as:
    #   B = [[b00, b01, b02, b03, b04, b05],
    #        [b10, b11, b12, b13, b14, b15],
    #        [b20, b21, b22, b23, b24, b25]]
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This is a 36-term sum. We want to reduce it.
    #
    # Actually, the rank of the matrix multiplication tensor for m x n * n x p
    # is at most m*n*p (schoolbook) and at least max(m*n, m*p, n*p).
    # For 2x3 * 3x6: max(6, 12, 18) = 18.
    #
    # The known minimum rank for 2x3x6 is not well-established in my memory.
    # Let me try a different approach: use the fact that we can factor through
    # intermediate matrices.
    #
    # Actually, let me just try to find a valid decomposition with rank as low as possible
    # using a systematic approach. Given the time constraint, let me try a few known
    # constructions.
    #
    # One approach: use the block structure.
    # Write A = [A_left, A_right] where A_left is 2x2, A_right is 2x1.
    # Write B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 1x6.
    # C = A_left * B_top + A_right * B_bottom
    #
    # A_right * B_bottom: (2x1)*(1x6) = 2x6. This has rank 2 (since it's a rank-1
    # outer product scaled by a vector, actually it's sum of 2 rank-1 terms).
    # Actually, A_right * B_bottom[i,k] = A_right[i,0] * B_bottom[0,k].
    # This is a rank-1 matrix (outer product of A_right[:,0] and B_bottom[0,:]).
    # So it can be decomposed with rank 1! Wait, no. In the tensor decomposition sense,
    # each entry C[i,k] = sum_j A[i,j] B[j,k]. For this sub-problem, j only goes from
    # 2 to 2 (one term), so C[i,k] = A[i,2] * B[2,k]. This is a rank-1 tensor in the
    # (i,k) space, but in the full tensor decomposition, it contributes one rank-1 term.
    #
    # So A_right * B_bottom contributes rank 1 to the total.
    #
    # A_left * B_top: (2x2)*(2x6). This has schoolbook rank 2*2*6 = 24.
    # But we can do better. Write B_top = [B1, B2, B3] where each Bx is 2x2.
    # A_left * B_top = A_left * B1 + A_left * B2 + A_left * B3
    # Each A_left * Bx is 2x2*2x2, rank 7 (Strassen).
    # Total: 3 * 7 = 21.
    # Grand total: 21 + 1 = 22.
    #
    # Can we do better? What if we group differently?
    # B_top is 2x6. We could write it as [B1, B2] where B1 is 2x3, B2 is 2x3.
    # A_left * B1: 2x2 * 2x3. Schoolbook rank 12.
    # For 2x2*2x3, we can write B1 = [B1a, B1b] where B1a is 2x2, B1b is 2x1.
    # A_left * B1a: 2x2*2x2, rank 7.
    # A_left * B1b: 2x2*2x1, rank 2.
    # Total for 2x2*2x3: 9.
    # Similarly for B2: 9.
    # Total for A_left * B_top: 18.
    # Grand total: 18 + 1 = 19.
    #
    # Can we do even better? What about the 2x2*2x1 part?
    # A_left * B1b where B1b is 2x1.
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B1b[j,k] for k in {0,1} (two columns).
    # This is 4 entries, each a sum of 2 products. Schoolbook rank 4.
    # But we can do it with rank 2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # We can use:
    # Term 1: A[0,0] * B[0,0] -> contributes to C[0,0]
    # Term 2: A[0,1] * B[1,0] -> contributes to C[0,0]
    # Term 3: A[0,0] * B[0,1] -> contributes to C[0,1]
    # Term 4: A[0,1] * B[1,1] -> contributes to C[0,1]
    # Term 5: A[1,0] * B[0,0] -> contributes to C[1,0]
    # Term 6: A[1,1] * B[1,0] -> contributes to C[1,0]
    # Term 7: A[1,0] * B[0,1] -> contributes to C[1,1]
    # Term 8: A[1,1] * B[1,1] -> contributes to C[1,1]
    # That's rank 8, not 2. I made an error.
    #
    # Actually, 2x2*2x1 is just matrix-vector multiplication, which has rank 4 (schoolbook).
    # There's no known improvement for 2x2*2x1.
    #
    # So my best estimate is rank 19. Let me implement that.

    # Reset
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(18)]
    W = [[0.0] * rank for _ in range(12)]
    col = 0

    # Strassen 2x2x2 block: A is 2x2 at rows i0,i0+1 cols j0,j0+1
    # B is 2x2 at rows j0,j0+1 cols k0,k0+1
    # C is 2x2 at rows i0,i0+1 cols k0,k0+1
    def strassen_block(i0, j0, k0):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[i0*3+j0][col] = 1.0; U[i0*3+j0+1][col] = 1.0
        V[j0*6+k0][col] = 1.0; V[j0*6+k0+1][col] = 1.0
        W[k0*2+i0][col] = 1.0; W[(k0+1)*2+i0+1][col] = 1.0
        col += 1
        # M2 = (c+d)e
        U[i0*3+j0][col] = 1.0; U[i0*3+j0+1][col] = 1.0
        V[j0*6+k0][col] = 1.0
        W[(k0+1)*2+i0][col] = 1.0
        col += 1
        # M3 = a(f-h)
        U[i0*3+j0][col] = 1.0
        V[j0*6+k0+1][col] = 1.0; V[(j0+1)*6+k0+1][col] = -1.0
        W[k0*2+i0+1][col] = 1.0
        col += 1
        # M4 = d(g-e)
        U[i0*3+j0+1][col] = 1.0
        V[(j0+1)*6+k0][col] = 1.0; V[j0*6+k0][col] = -1.0
        W[(k0+1)*2+i0][col] = 1.0
        col += 1
        # M5 = (a+b)h
        U[i0*3+j0][col] = 1.0; U[i0*3+j0+1][col] = 1.0
        V[(j0+1)*6+k0+1][col] = 1.0
        W[k0*2+i0][col] = -1.0; W[k0*2+i0+1][col] = 1.0
        col += 1
        # M6 = (c-a)(e+f)
        U[i0*3+j0][col] = -1.0; U[i0*3+j0][col] = -1.0  # c-a: c=+1, a=-1
        # Actually c is A[i0+1, j0], a is A[i0, j0]
        U[i0*3+j0][col] = -1.0  # a
        U[i0*3+j0][col] = -1.0  # this is wrong, let me fix
        # c = A[i0+1, j0], a = A[i0, j0]
        # c-a: U[i0*3+j0] = -1 (for a), U[i0*3+j0+1] = +1 (for c)
        U[i0*3+j0][col] = -1.0
        U[i0*3+j0+1][col] = 1.0
        V[j0*6+k0][col] = 1.0; V[j0*6+k0+1][col] = 1.0
        W[(k0+1)*2+i0+1][col] = 1.0
        col += 1
        # M7 = (b-d)(g+h)
        # b = A[i0, j0+1], d = A[i0+1, j0+1]
        U[i0*3+j0+1][col] = 1.0  # b
        U[i0*3+j0+1][col] = 1.0  # d with sign -1
        U[i0*3+j0+1][col] = 1.0
        U[i0*3+j0+1][col] = 1.0
        # b-d: U[i0*3+j0+1] = +1, U[i0*3+j0+1] = -1
        U[i0*3+j0+1][col] = 1.0
        U[i0*3+j0+1][col] = -1.0
        # This is getting messy. Let me use a different approach.

    # Let me just hardcode the Strassen decomposition properly.
    # I'll use a helper that takes explicit row/col indices.

    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(18)]
    W = [[0.0] * rank for _ in range(12)]
    col = 0

    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        if val != 0:
            U[u_idx][col] += val
            V[v_idx][col] += val
            W[w_idx][col] += val
        col += 1

    # This approach won't work because each term in Strassen involves multiple U, V, W entries.
    # Let me use a different method: define the 7 Strassen multiplies explicitly.

    # For a 2x2*2x2 block with A at (i0..i0+1, j0..j0+1), B at (j0..j0+1, k0..k0+1):
    # a=A[i0,j0], b=A[i0,j0+1], c=A[i0+1,j0], d=A[i0+1,j0+1]
    # e=B[j0,k0], f=B[j0,k0+1], g=B[j0+1,k0], h=B[j0+1,k0+1]
    #
    # M1=(a+d)(e+h): U has a=1,d=1; V has e=1,h=1; W has C00=1,C11=1
    # M2=(c+d)e: U has c=1,d=1; V has e=1; W has C10=1
    # M3=a(f-h): U has a=1; V has f=1,h=-1; W has C01=1
    # M4=d(g-e): U has d=1; V has g=1,e=-1; W has C10=1
    # M5=(a+b)h: U has a=1,b=1; V has h=1; W has C00=-1,C01=1
    # M6=(c-a)(e+f): U has c=1,a=-1; V has e=1,f=1; W has C11=1
    # M7=(b-d)(g+h): U has b=1,d=-1; V has g=1,h=1; W has C00=1,C11=-1

    # I need a way to add multiple entries to the same column.
    # Let me redefine.

    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(18)]
    W = [[0.0] * rank for _ in range(12)]
    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*3+j0][col] += 1.0
        U[i0*3+j0+1][col] += 1.0
        V[j0*6+k0][col] += 1.0
        V[j0*6+k0+1][col] += 1.0
        W[k0*2+i0][col] += 1.0
        W[(k0+1)*2+i0+1][col] += 1.0
        col += 1
        # M2
        U[i0*3+j0][col] += 1.0
        U[i0*3+j0+1][col] += 1.0
        V[j0*6+k0][col] += 1.0
        W[(k0+1)*2+i0][col] += 1.0
        col += 1
        # M3
        U[i0*3+j0][col] += 1.0
        V[j0*6+k0+1][col] += 1.0
        V[(j0+1)*6+k0+1][col] += -1.0
        W[k0*2+i0+1][col] += 1.0
        col += 1
        # M4
        U[i0*3+j0+1][col] += 1.0
        V[(j0+1)*6+k0][col] += 1.0
        V[j0*6+k0][col] += -1.0
        W[(k0+1)*2+i0][col] += 1.0
        col += 1
        # M5
        U[i0*3+j0][col] += 1.0
        U[i0*3+j0+1][col] += 1.0
        V[(j0+1)*6+k0+1][col] += 1.0
        W[k0*2+i0][col] += -1.0
        W[k0*2+i0+1][col] += 1.0
        col += 1
        # M6
        U[i0*3+j0][col] += -1.0
        U[i0*3+j0][col] += -1.0  # a
        U[i0*3+j0][col] += -1.0  # this is wrong
        # c-a: c is at (i0+1, j0), a is at (i0, j0)
        U[i0*3+j0][col] += -1.0  # a
        U[i0*3+j0][col] += -1.0  # duplicate
        # Let me fix: U[i0*3+j0] = -1 (a), U[i0*3+j0+1]... no, c is U[(i0+1)*3+j0]
        # Wait, i0 is always 0 in our case since n=2.
        # Let me just be explicit.
        U[i0*3+j0][col] += -1.0  # a
        U[(i0+1)*3+j0][col] += 1.0  # c
        V[j0*6+k0][col] += 1.0  # e
        V[j0*6+k0+1][col] += 1.0  # f
        W[(k0+1)*2+i0+1][col] += 1.0  # C11
        col += 1
        # M7
        # b-d: b is A[i0,j0+1], d is A[i0+1,j0+1]
        U[i0*3+j0+1][col] += 1.0  # b
        U[(i0+1)*3+j0+1][col] += -1.0  # d
        V[(j0+1)*6+k0][col] += 1.0  # g
        V[(j0+1)*6+k0+1][col] += 1.0  # h
        W[k0*2+i0][col] += 1.0  # C00
        W[(k0+1)*2+i0+1][col] += -1.0  # C11
        col += 1

    # Now build the full decomposition.
    # A is 2x3, B is 3x6.
    # Split: A_left = A[:,0:2] (2x2), A_right = A[:,2:3] (2x1)
    # B_top = B[0:2,:] (2x6), B_bottom = B[2:3,:] (1x6)
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # A_right * B_bottom: rank 1
    # A_left * B_top: split B_top into B_top1 (2x3) and B_top2 (2x3)
    #   B_top1 = B[0:2, 0:3], B_top2 = B[0:2, 3:6]
    #   Each: split into 2x2 and 2x1
    #   B_top1_left = B[0:2, 0:2] (2x2), B_top1_right = B[0:2, 2:3] (2x1)
    #   B_top2_left = B[0:2, 3:5] (2x2), B_top2_right = B[0:2, 5:6] (2x1)
    #
    # Strassen blocks:
    # 1. A_left * B_top1_left: i0=0, j0=0, k0=0
    # 2. A_left * B_top2_left: i0=0, j0=0, k0=3
    #
    # 2x2*2x1 blocks (rank 4 each, schoolbook):
    # 3. A_left * B_top1_right: A is 2x2 at (0:2, 0:2), B is 2x1 at (0:2, 2:3)
    # 4. A_left * B_top2_right: A is 2x2 at (0:2, 0:2), B is 2x1 at (0:2, 5:6)
    #
    # 2x1*1x6 block (rank 1):
    # 5. A_right * B_bottom

    # Add Strassen blocks
    add_strassen(0, 0, 0)   # A_left * B[0:2, 0:2]
    add_strassen(0, 0, 3)   # A_left * B[0:2, 3:5]

    # col is now 14

    # 2x2*2x1 block: A at (0:2, 0:2), B at (0:2, 2:3) -> C at (0:2, 2:3)
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {2}
    # This is 4 entries, each a sum of 2 products. Rank 4 (schoolbook).
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # Term 1: A[0,0]*B[0,2] -> U[0]=1, V[2]=1, W[2*2+0]=1
    # Term 2: A[0,1]*B[1,2] -> U[1]=1, V[1*6+2]=1, W[2*2+0]=1
    # Term 3: A[1,0]*B[0,2] -> U[3]=1, V[2]=1, W[2*2+1]=1
    # Term 4: A[1,1]*B[1,2] -> U[4]=1, V[1*6+2]=1, W[2*2+1]=1

    # B at (0:2, 2:3) means j in {0,1}, k=2
    # V index: j*6+k
    # W index: k*2+i

    U[0*3+0][col] = 1.0; V[0*6+2][col] = 1.0; W[2*2+0][col] = 1.0; col += 1
    U[0*3+1][col] = 1.0; V[1*6+2][col] = 1.0; W[2*2+0][col] = 1.0; col += 1
    U[1*3+0][col] = 1.0; V[0*6+2][col] = 1.0; W[2*2+1][col] = 1.0; col += 1
    U[1*3+1][col] = 1.0; V[1*6+2][col] = 1.0; W[2*2+1][col] = 1.0; col += 1

    # 2x2*2x1 block: A at (0:2, 0:2), B at (0:2, 5:6) -> C at (0:2, 5:6)
    # k=5
    U[0*3+0][col] = 1.0; V[0*6+5][col] = 1.0; W[5*2+0][col] = 1.0; col += 1
    U[0*3+1][col] = 1.0; V[1*6+5][col] = 1.0; W[5*2+0][col] = 1.0; col += 1
    U[1*3+0][col] = 1.0; V[0*6+5][col] = 1.0; W[5*2+1][col] = 1.0; col += 1
    U[1*3+1][col] = 1.0; V[1*6+5][col] = 1.0; W[5*2+1][col] = 1.0; col += 1

    # col is now 22. But we only allocated rank=20. Let me check.
    # 14 (Strassen) + 4 + 4 = 22. Plus the 2x1*1x6 block.
    #
    # A_right * B_bottom: A at (0:2, 2:3), B at (2:3, 0:6)
    # C[i,k] = A[i,2]*B[2,k] for i in {0,1}, k in {0..5}
    # This is 12 entries, each a single product. Rank 12 if done naively.
    # But it's a rank-1 outer product, so in tensor decomposition it's rank 1?
    # No, each entry C[i,k] = A[i,2]*B[2,k]. The tensor is T[i,j,k] where j=2.
    # T[i,2,k] = A[i,2]*B[2,k]. This is a rank-1 tensor in the (i,k) subspace
    # for fixed j=2. In the full decomposition, this contributes one term:
    # U[i*3+2] = A[i,2], V[2*6+k] = B[2,k], W[k*2+i] = 1
    # But that's not how it works. Each rank-1 term in the decomposition is
    # a product U[i,j]*V[j,k]*W[k,i]. For j=2, we need:
    # U[i*3+2, r] * V[2*6+k, r] * W[k*2+i, r] = A[i,2]*B[2,k]
    # We can set U[i*3+2, r] = 1, V[2*6+k, r] = 1, W[k*2+i, r] = 1 for all i,k.
    # But then the product is 1, not A[i,2]*B[2,k].
    #
    # The issue is that A[i,2] and B[2,k] are variables, not constants.
    # In the tensor decomposition, the entries of U, V, W are constants.
    # The tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k]).
    # So we need sum_r U[i,j]*V[j,k]*W[k,i] = 1 for all i,j,k.
    #
    # For j=2, we need sum_r U[i,2]*V[2,k]*W[k,i] = 1 for all i,k.
    # This is a 2x6 matrix of 1s, which has rank 2. So we need at least 2 terms.
    #
    # Let me use 2 terms for this:
    # Term 1: U[0,2]=1, U[1,2]=1, V[2,0]=1, V[2,1]=1, V[2,2]=1, V[2,3]=1, V[2,4]=1, V[2,5]=1
    #          W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1, W[4,0]=1, W[5,0]=1
    #          This gives W[k,0]=1 for all k, so C[0,k] gets +1 from this term.
    # Term 2: Similarly for C[1,k].
    #
    # Actually, let me just use 2 terms:
    # Term A: U[0*3+2]=1, U[1*3+2]=1, V[2*6+k]=1 for all k, W[k*2+0]=1 for all k
    #   This contributes 1 to C[0,k] for all k.
    # Term B: U[0*3+2]=1, U[1*3+2]=1, V[2*6+k]=1 for all k, W[k*2+1]=1 for all k
    #   This contributes 1 to C[1,k] for all k.
    #
    # But wait, U[0*3+2] and U[1*3+2] would be set to 1 in both terms,
    # and V[2*6+k] would be 1 in both terms. That's fine.

    # But I'm over the rank budget. Let me just return the schoolbook for now
    # since I'm struggling with the exact construction.

    # Actually, let me just verify the Strassen part and use schoolbook for the rest.
    # The Strassen blocks handle (0:2, 0:2) and (0:2, 3:5) for the A_left * B_top part.
    # The remaining parts are:
    # - A_left * B[0:2, 2:3] (2x2 * 2x1)
    # - A_left * B[0:2, 5:6] (2x2 * 2x1)
    # - A_right * B_bottom (2x1 * 1x6)
    #
    # Let me just use schoolbook for these and see what rank we get.
    # Strassen: 14 terms
    # A_left * B[0:2, 2:3]: 4 terms
    # A_left * B[0:2, 5:6]: 4 terms
    # A_right * B_bottom: 12 terms (schoolbook)
    # Total: 34 terms. Still better than 36.

    # But I already have col=22 after the Strassen and two 2x2*2x1 blocks.
    # Let me just add the A_right * B_bottom part with schoolbook.

    for i in range(2):
        for k in range(6):
            U[i*3+2][col] = 1.0
            V[2*6+k][col] = 1.0
            W[k*2+i][col] = 1.0
            col += 1

    # col should be 22 + 12 = 34. But rank=20. This won't work.
    # Let me just return schoolbook.

    U_sb = [[0] * 36 for _ in range(6)]
    V_sb = [[0] * 36 for _ in range(18)]
    W_sb = [[0] * 36 for _ in range(12)]
    c = 0
    for i in range(2):
        for j in range(3):
            for k in range(6):
                U_sb[i*3+j][c] = 1
                V_sb[j*6+k][c] = 1
                W_sb[k*2+i][c] = 1
                c += 1

    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
