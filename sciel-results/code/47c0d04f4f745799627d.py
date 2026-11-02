"""Matrix multiplication 4x4x6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook as fallback
    rank_school = n * m * p  # 96
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Strassen 2x2x2 decomposition terms
    # Each term: (A_coefs, B_coefs, C_coefs)
    # A is 2x2, B is 2x2, C is 2x2
    # Indices: (row, col)
    strassen_terms = [
        # P1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # P3 = A00(B10-B11)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4 = A11(B01-B00)
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    def verify(U, V, W, rank, n, m, p):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    # Strategy 1: Block Strassen - 12 independent 2x2x2 Strassen products
    # A: 4x4 -> 2x2 blocks indexed by (a,c) where a=row_block, c=col_block
    # B: 4x6 -> 2x2 blocks indexed by (c,b) where c=row_block, b=col_block
    # C: 4x6 -> 2x2 blocks indexed by (a,b) where a=row_block, b=col_block
    # C[a,b] = sum_c A[a,c] * B[c,b]
    # Total: 2*2*3 = 12 products, each rank 7 -> rank 84
    
    rank1 = 12 * 7  # 84
    U1 = [[0] * rank1 for _ in range(n * m)]  # 16 x 84
    V1 = [[0] * rank1 for _ in range(m * p)]  # 24 x 84
    W1 = [[0] * rank1 for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for a in range(2):       # row block of A/C
        for c in range(2):   # col block of A / row block of B
            for b in range(3):  # col block of B/C
                for (a_coefs, b_coefs, c_coefs) in strassen_terms:
                    # A-side: A_block(a,c) = A[2a:2a+2, 2c:2c+2]
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar  # global row in A
                        gj = 2*c + ac  # global col in A
                        U1[gi * m + gj][col] += coef
                    
                    # B-side: B_block(c,b) = B[2c:2c+2, 2b:2b+2]
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*c + br  # global row in B
                        gj = 2*b + bc  # global col in B
                        V1[gi * p + gj][col] += coef
                    
                    # C-side: C_block(a,b) = C[2a:2a+2, 2b:2b+2]
                    # W[k*n+i] where k is row of C (=col of B), i is col of C (=row of A)
                    for (cr, cc), coef in c_coefs.items():
                        gi = 2*b + cc  # k (row of C)
                        gj = 2*a + cr  # i (col of C)
                        W1[gi * n + gj][col] += coef
                    
                    col += 1
    
    assert col == rank1, f"col={col}, rank1={rank1}"
    
    if verify(U1, V1, W1, rank1, n, m, p):
        best_rank = rank1
        best_U = U1
        best_V = V1
        best_W = W1
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Strategy 2: Try to reduce rank by sharing factors
    # The key observation: when we have consecutive b values (b=0,1,2), 
    # the B-side factors for b=0 and b=1 share the same row block c.
    # Similarly for b=1 and b=2.
    # 
    # Let's try a different approach: use a 4x4x4 Strassen-like decomposition
    # and extend it to 4x4x6.
    #
    # Actually, let's try to use the fact that we can combine some Strassen terms.
    # For products with the same (a,c) but different b, the A-side is the same.
    # We can potentially share the A-side linear combinations.
    #
    # Alternative: Use a 4x4x4 block decomposition with rank 49 (known optimal for 4x4x4)
    # and extend to 4x4x6.
    #
    # Known: 4x4x4 matrix multiplication has rank 49 (Strassen's result)
    # For 4x4x6, we can think of it as 4x4x4 + 4x4x2
    # But this doesn't directly give us a good bound.
    #
    # Let's try another approach: use the rectangular Strassen algorithm.
    # For 4x4 * 4x6, we can split into:
    # - Left half: 4x4 * 4x4 (rank 49)
    # - Right half: 4x4 * 4x2 (need to find good rank)
    #
    # For 4x4 * 4x2:
    # Split A into 2x2 blocks (4 blocks), B into 2x1 blocks (4 blocks)
    # C is 4x2, split into 2x1 blocks (4 blocks)
    # C[a,b] = sum_c A[a,c] * B[c,b] where a,c in {0,1}, b in {0,1}
    # That's 2*2*2 = 8 products of 2x2 * 2x1
    #
    # For 2x2 * 2x1:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b10]] (2x1)
    # C = [[a00*b00 + a01*b10], [a10*b00 + a11*b10]]
    # This is just 4 multiplications, but we can use Strassen-like tricks.
    #
    # Actually, let's try a completely different approach.
    # Use the known rank-49 decomposition for 4x4x4 and add 3 more columns.
    #
    # For the 4x4x4 part, use Strassen's rank-49 algorithm.
    # For the additional 2 columns (making it 4x4x6), we need to handle
    # the extra B columns and corresponding C columns.
    
    # Let's try a more systematic approach with factor sharing.
    # 
    # Key idea: For the 12 Strassen products, group them by (a,c) pair.
    # For fixed (a,c), we have 3 products with b=0,1,2.
    # The A-side is the same for all 3. We can potentially reduce the
    # number of distinct A-side vectors.
    #
    # Actually, a better approach: use the rectangular matrix multiplication
    # algorithm that exploits the structure more carefully.
    #
    # Let me try: split the problem into two 4x4x3 subproblems.
    # 4x4 * 4x3: 
    #   A: 4x4, B: 4x3, C: 4x3
    #   Split A into 2x2 blocks (4 blocks)
    #   Split B into 2x2 and 2x1 blocks
    #   This gets complicated.
    #
    # Let's stick with the block Strassen and try to optimize.
    #
    # Alternative: Use a 4x4x4 rank-49 decomposition for the first 4 columns
    # of B, and a 4x4x2 decomposition for the last 2 columns.
    #
    # For 4x4x2:
    #   A: 4x4, B: 4x2, C: 4x2
    #   Split A into 2x2 blocks (4 blocks)
    #   B: 4x2 -> 2x1 blocks (4 blocks)
    #   C: 4x2 -> 2x1 blocks (4 blocks)
    #   C[a,b] = sum_c A[a,c] * B[c,b] where a,c in {0,1}, b in {0,1}
    #   8 products of 2x2 * 2x1
    #
    # For 2x2 * 2x1, the minimal rank is 4 (schoolbook) since there's no
    # obvious Strassen-like trick for rectangular 2x2 * 2x1.
    # Actually, let's check: 
    #   A = [[a,b],[c,d]], B = [[e],[f]], C = [[ae+bf],[ce+df]]
    #   T[0,0,0]=ae, T[0,1,0]=af, T[1,0,0]=ce, T[1,1,0]=de,
    #   T[0,0,1]=be, T[0,1,1]=bf, T[1,0,1]=de... wait let me reindex.
    #
    # For 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1
    # C[i] = sum_j A[i,j] * B[j]
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is 4 scalar multiplications, rank 4.
    #
    # So 4x4x2 would have rank 8*4 = 32 with this approach.
    # Total: 49 + 32 = 81. Better than 84!
    #
    # But wait, I need to verify the rank-49 for 4x4x4. Let me use a simpler approach.
    #
    # Actually, let me try a different decomposition for 4x4x4.
    # Use 4 blocks of 2x2x2 Strassen: 4*7 = 28.
    # C[a,b] = sum_c A[a,c] * B[c,b] where a,b,c in {0,1}
    # That's 8 products, each rank 7 -> rank 56.
    #
    # So 4x4x4 with block Strassen: rank 56
    # 4x4x2 with block approach: 8 products of 2x2*2x1, each rank 4 -> rank 32
    # Total: 56 + 32 = 88. Worse than 84.
    #
    # Let me reconsider. The original 84 is from 12 Strassen products.
    # Can I do better?
    #
    # What if I use a 4x4x4 rank-49 and then handle the extra 2 columns?
    # The rank-49 for 4x4x4 is known but complex to implement.
    #
    # Let me try yet another approach: use the 2x2x2 Strassen but with
    # smarter grouping.
    #
    # For 4x4x6:
    # Group the 12 products differently. Instead of (a,c,b) with a,c in {0,1}, b in {0,1,2},
    # what if I use a 2x2x3 structure?
    #
    # Actually, let's try to use the fact that the B matrix is 4x6, which can be
    # viewed as two 4x3 matrices or three 4x2 matrices.
    #
    # Let me try: 4x4 * 4x6 = (4x4 * 4x3) + (4x4 * 4x3)
    # For 4x4 * 4x3:
    #   A: 4x4 -> 2x2 blocks (4 blocks)
    #   B: 4x3 -> 2x2 and 2x1 blocks
    #   C: 4x3 -> 2x2 and 2x1 blocks
    #
    # This is getting complicated. Let me just try to verify if rank 84 is correct
    # and see if I can find a slightly better one through careful optimization.
    #
    # Let's try a different block size: 1x1x1 is schoolbook (96).
    # 2x2x2 gives 84.
    # What about mixing block sizes?
    
    # Let me try to reduce the rank by noticing that some Strassen terms
    # can be combined when they share the same A or B factor.
    #
    # For fixed (a,c), the 3 products with b=0,1,2 all use the same A_block(a,c).
    # The A-side vectors for the 7 Strassen terms are fixed.
    # For 3 different b values, we have 3*7 = 21 rank-1 terms that all share
    # the same 7 A-side vectors (repeated 3 times).
    #
    # This doesn't directly reduce the rank, but it suggests a structure.
    #
    # Let me try a completely different approach: use a 4x4x4 decomposition
    # with rank 49 (Strassen) and add 3 columns.
    #
    # Actually, I recall that the tensor rank of 4x4x4 matrix multiplication
    # is 49. Let me try to construct it.
    #
    # The rank-49 decomposition for 4x4x4 uses a more sophisticated
    # Strassen-like algorithm. It's complex but known.
    #
    # For now, let's try to improve on 84 by using a hybrid approach.
    # Use 2x2x2 Strassen for the first 4 columns (8 products, rank 56)
    # and a different approach for the last 2 columns.
    #
    # For the last 2 columns: 4x4 * 4x2
    # Split A into 2x2 blocks (4 blocks)
    # B: 4x2 -> split into 2x1 blocks (4 blocks)
    # C: 4x2 -> split into 2x1 blocks (4 blocks)
    # C[a,b] = sum_c A[a,c] * B[c,b] where a,c in {0,1}, b in {0,1}
    # 8 products of 2x2 * 2x1, each rank 4 -> rank 32
    # Total: 56 + 32 = 88. Still worse than 84.
    #
    # What if I use 1x1x1 for the last 2 columns?
    # 4x4 * 4x2: 4*4*2 = 32 scalar multiplications.
    # Total: 56 + 32 = 88. Same.
    #
    # Let me try: use 2x2x2 Strassen for all 6 columns, but optimize.
    #
    # Actually, I think 84 might be close to optimal for this approach.
    # Let me try to see if I can get below 84 with a different strategy.
    #
    # New idea: use a 4x2x2 block structure.
    # A: 4x4 -> 4x2 blocks (2 blocks)
    # B: 4x6 -> 2x2 and 2x3 blocks
    # This doesn't work cleanly.
    #
    # Let me try: A: 4x4, split into 2 4x2 blocks (left and right halves)
    # B: 4x6, split into 6 4x1 blocks (columns)
    # C: 4x6, split into 6 4x1 blocks (columns)
    # C[:,j] = A[:,0:2] * B[0:2,j] + A[:,2:4] * B[2:4,j]
    # For each j, this is 4x2 * 2x1 + 4x2 * 2x1
    # 4x2 * 2x1: rank 4 (as computed earlier)
    # So for each j: rank 8, total rank 6*8 = 48?
    # Wait, that can't be right. Let me recompute.
    #
    # 4x2 * 2x1: A is 4x2, B is 2x1, C is 4x1
    # C[i] = sum_j A[i,j] * B[j]
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # C[2] = A[2,0]*B[0] + A[2,1]*B[1]
    # C[3] = A[3,0]*B[0] + A[3,1]*B[1]
    # This is 8 scalar multiplications, rank 8.
    #
    # So for each column j of C: rank 8
    # Total for 6 columns: 6*8 = 48.
    # But wait, this doesn't account for the fact that A is shared across columns.
    #
    # Actually, the tensor rank is for the entire 4x4x6 problem.
    # T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] contributes to C[k,i])
    # The rank is the minimum number of rank-1 tensors that sum to T.
    #
    # With the approach above:
    # C[:,j] = A[:,0:2] * B[0:2,j] + A[:,2:4] * B[2:4,j]
    # For each (j, half) where half in {0,1}:
    #   A_half: 4x2, B_half: 2x1, C_half: 4x1
    #   Rank 8 each
    #   Total: 6*2*8 = 96. Same as schoolbook!
    #
    # The issue is that I'm not sharing factors properly.
    #
    # Let me go back to the block Strassen approach and try to optimize.
    #
    # I believe 84 is a good result. Let me verify it's correct and return it.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
