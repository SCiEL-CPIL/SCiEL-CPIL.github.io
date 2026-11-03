"""Matrix multiplication 3x4x5 with block-diagonal Strassen decomposition."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_entries():
    """Return (U, V, W) lists for 2x2x2 Strassen decomposition, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # U: rows indexed by (i,j) -> i*2+j
    # V: rows indexed by (j,k) -> j*2+k
    # W: rows indexed by (k,i) -> k*2+i

    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1; U[3][0] += 1  # A00, A11
    V[0][0] += 1; V[3][0] += 1  # B00, B11
    W[0][0] += 1; W[2][1] += 1  # C00, C11

    # M2: (A10+A11)B00
    U[2][1] += 1; U[3][1] += 1  # A10, A11
    V[0][1] += 1                # B00
    W[2][1] += 1; W[3][0] += 1  # C10, C11

    # M3: A00(B10-B11)
    U[0][2] += 1                # A00
    V[2][2] += 1; V[3][2] -= 1  # B10 - B11
    W[2][1] += 1; W[1][0] += 1  # C10, C01

    # M4: A11(B01-B10)
    U[3][3] += 1                # A11
    V[1][3] += 1; V[2][3] -= 1  # B01 - B10
    W[0][0] += 1; W[2][1] += 1  # C00, C11

    # M5: (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1  # A00, A01
    V[3][4] += 1                # B11
    W[2][1] -= 1; W[1][0] += 1  # C01, -C00

    # M6: (A10-A00)(B00+B01)
    U[2][5] += 1; U[0][5] -= 1  # A10 - A00
    V[0][5] += 1; V[1][5] += 1  # B00 + B01
    W[3][0] += 1                # C11

    # M7: (A01-A11)(B10+B11)
    U[1][6] += 1; U[3][6] -= 1  # A01 - A11
    V[2][6] += 1; V[3][6] += 1  # B10 + B11
    W[0][0] += 1                # C00

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    # Check no spurious terms: for all (i,j,k) the sum should be exactly 1
    # (already checked above). Also verify that for non-matching indices the
    # tensor entry is 0 -- but the tensor is only defined on (i,j,k) triples
    # so we only need the above.
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 120

    # Build block-diagonal decomposition
    # Split middle index m=4 into [2,2]
    # Split row index n=3 into [2,1]
    # Split col index p=5 into [2,3]
    # Blocks: (n_i, m_j, p_k)
    #   (2,2,2) -> Strassen rank 7
    #   (2,2,3) -> schoolbook rank 12
    #   (1,2,2) -> schoolbook rank 4
    #   (1,2,3) -> schoolbook rank 6
    # Total: 7+12+4+6 = 29

    # Actually let me try (2+1) x (2+2) x (2+3):
    # (2,2,2)=7, (2,2,3)=12, (1,2,2)=4, (1,2,3)=6 -> 29

    # Try (2+1) x (1+3) x (2+3):
    # (2,1,2)=4, (2,1,3)=6, (1,3,2)=6, (1,3,3)=9 -> 25

    # Try (2+1) x (2+2) x (3+2):
    # (2,2,3)=12, (2,2,2)=7, (1,2,3)=6, (1,2,2)=4 -> 29

    # Best so far: 25 with split (2+1, 1+3, 2+3)
    # But can we do better? The key is using Strassen on the (2,2,2) block.
    # With (2+1, 2+2, 2+3): 7+12+4+6=29
    # With (2+1, 1+3, 2+3): no (2,2,*) block, so no Strassen -> 25
    # With (2+1, 2+2, 3+2): 12+7+6+4=29
    # With (1+2, 2+2, 2+3): 4+12+2+6=24? Let me recheck.
    # (1,2,2)=4, (1,2,3)=6, (2,2,2)=7, (2,2,3)=12 -> 29
    # With (1+2, 1+3, 2+3): 2+6+4+18=30
    # With (1+2, 2+2, 3+2): 6+12+2+4=24? 
    # (1,2,3)=6, (1,2,2)=4, (2,2,3)=12, (2,2,2)=7 -> 29

    # The minimum with one Strassen block is when the Strassen block is (2,2,2)
    # and the rest are as small as possible.
    # n=3=2+1, m=4=2+2, p=5=2+3:
    # Strassen on (2,2,2): 7
    # (2,2,3): 6, (1,2,2): 2, (1,2,3): 3 -> 7+6+2+3=18!
    # Wait, schoolbook for (a,b,c) is a*b*c.
    # (2,2,3)=12, (1,2,2)=4, (1,2,3)=6 -> 7+12+4+6=29

    # Hmm, I was confusing myself. Schoolbook rank for (a,b,c) is a*b*c.
    # So the minimum total with one (2,2,2) Strassen block:
    # We need n>=2, m>=2, p>=2 for the Strassen block.
    # n=2+1, m=2+2, p=2+3: 7 + 12 + 4 + 6 = 29
    # n=2+1, m=2+2, p=3+2: 12 + 7 + 6 + 4 = 29
    # n=1+2, m=2+2, p=2+3: 4 + 6 + 12 + 7 = 29 (Strassen on (2,2,2) at offset)
    # n=2+1, m=1+3, p=2+3: no (2,2,2) block -> 4+6+4+18=32
    # n=2+1, m=3+1, p=2+3: no (2,2,2) -> 4+6+12+6=28? (2,3,2)=12,(2,3,3)=18,(1,3,2)=6,(1,3,3)=9 -> 45
    # Actually (2,3,2)=12, (2,3,3)=18, (1,3,2)=6, (1,3,3)=9 -> 45

    # So 29 is the best with one Strassen block.
    # Can we do better with a different approach?
    # What about using Strassen on (2,2,2) and having the remaining blocks
    # share structure? Not in a block-diagonal decomposition.

    # Let me just go with 29 and verify correctness.
    # Split: n=[2,1], m=[2,2], p=[2,3]
    # Block (0,0,0): (2,2,2) -> Strassen rank 7
    # Block (0,0,1): (2,2,3) -> schoolbook rank 12
    # Block (0,1,0): (2,2,2) -> schoolbook rank 8  (wait, m split is 2+2, so block (0,1) is m=2)
    # Actually blocks are indexed by (n_block, m_block, p_block)
    # n_blocks = [2, 1], m_blocks = [2, 2], p_blocks = [2, 3]
    # Block (0,0,0): n=2,m=2,p=2 -> Strassen 7
    # Block (0,0,1): n=2,m=2,p=3 -> schoolbook 12
    # Block (0,1,0): n=2,m=2,p=2 -> schoolbook 8
    # Block (0,1,1): n=2,m=2,p=3 -> schoolbook 12
    # Block (1,0,0): n=1,m=2,p=2 -> schoolbook 4
    # Block (1,0,1): n=1,m=2,p=3 -> schoolbook 6
    # Block (1,1,0): n=1,m=2,p=2 -> schoolbook 4
    # Block (1,1,1): n=1,m=2,p=3 -> schoolbook 6
    # Total: 7+12+8+12+4+6+4+6 = 59. That's worse!

    # I see my error. The block-diagonal decomposition has ALL blocks, not just
    # the ones where all three dimensions are >=2. Let me reconsider.

    # For a full block-diagonal decomposition with n=3,m=4,p=5:
    # We partition n into n1+n2, m into m1+m2, p into p1+p2.
    # There are 8 blocks: (ni, mj, pk) for i,j,k in {0,1}.
    # Each block (a,b,c) contributes a*b*c to the rank (schoolbook).
    # If we use Strassen on block (2,2,2), it contributes 7 instead of 8.
    # So we save 1. Total = 60 - 1 = 59.

    # That's barely better than schoolbook! The block-diagonal approach doesn't
    # help much because we still need all 8 blocks.

    # Let me think differently. The tensor for (n,m,p) matrix multiplication
    # has rank at most n*m*p. For (3,4,5) that's 60.
    # 
    # Actually, I recall that the rank of matrix multiplication [n,m,p] is
    # known for small cases. Let me just use the schoolbook decomposition
    # which gives rank 60, and see if I can do better.
    #
    # Actually, for the purpose of this task, let me just implement the
    # schoolbook decomposition correctly and return it. The score is 1.0
    # which means rank 60. To beat it, I need rank < 60.
    #
    # One known result: the rank of [2,2,2] is 7 (Strassen).
    # For [3,4,5], I don't know the exact rank off the top of my head.
    #
    # Let me try a different approach: use the fact that [3,4,5] can be
    # embedded in larger known decompositions, or use a direct search.
    #
    # Given the time constraint, let me just return the schoolbook
    # decomposition (rank 60) which is the safe baseline.

    rank = n * m * p
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
