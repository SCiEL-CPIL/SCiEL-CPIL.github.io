"""Strassen-based decomposition for 3x3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8

    # We'll build a Strassen-based decomposition.
    # Strategy: For each pair of B-columns (k1, k2), use Strassen 2x2x2 
    # for the (i,j) in {0,1} x {0,1} part, plus schoolbook for remaining.
    #
    # Actually, let me use a cleaner approach:
    # For 3x3x2 blocks (pairs of columns in B), I can use:
    # - Strassen rank-7 for the 2x2x2 sub-tensor (i,j in {0,1}, k in {0,1})
    # - Schoolbook for the remaining 10 terms (involving i=2 or j=2)
    # Total per 3x3x2: 7 + 10 = 17 terms
    # For 3x3x8 = 4 blocks of 3x3x2: 4 * 17 = 68 terms
    #
    # That's only slightly better than 72. Let me think of a better approach.
    #
    # Better: Use the fact that for 3x3x3, we can get rank <= 23.
    # But I need explicit coefficients. Let me use a known construction.
    #
    # For now, let me use a simpler valid improvement:
    # Split 3x3x8 into two 3x3x4 blocks.
    # For each 3x3x4, use:
    # - Strassen 2x2x2 (7 terms) for (i,j,k) in {0,1}x{0,1}x{0,1}
    # - Strassen 2x2x2 (7 terms) for (i,j,k) in {0,1}x{0,1}x{2,3}
    # - Schoolbook for remaining: 12 - 4 - 4 = 4 entries per 3x3x4
    #   Each needs 3 terms: 12 terms
    # Total per 3x3x4: 7 + 7 + 12 = 26 terms
    # Two blocks: 52 terms total.
    #
    # Wait, that's not right either. Let me be more careful.
    #
    # For 3x3x4: C[i][k] = sum_j A[i][j]*B[j][k], i in {0,1,2}, k in {0,1,2,3}
    # The tensor has 3*3*4 = 36 entries, all equal to 1.
    #
    # Using Strassen for (i,j) in {0,1} and k in {0,1}: 7 terms cover 8 of the 36.
    # Using Strassen for (i,j) in {0,1} and k in {2,3}: 7 terms cover 8 more.
    # Remaining: 36 - 16 = 20 entries need schoolbook: 20 terms.
    # Total: 7 + 7 + 20 = 34 per 3x3x4. Two blocks: 68. Same as before.
    #
    # Let me try a different split. What if I use Strassen on (i,k) pairs?
    # No, Strassen is for 2x2x2 where all three dimensions are size 2.
    #
    # OK let me just carefully implement the Strassen 2x2x2 decomposition
    # and combine with schoolbook for the rest.

    # Strassen 2x2x2 decomposition:
    # For (i,j,k) in {0,1}^3, the 7 rank-1 terms are:
    # m1: U covers (0,0),(1,1) with +1; V covers (0,0),(1,1) with +1; W covers (0,0),(1,1) with +1
    #     i.e., (A00+A11)*(B00+B11) contributes to C00 and C11
    # m2: U covers (1,0),(1,1) with +1; V covers (0,0) with +1; W covers (0,1) with +1
    #     i.e., (A10+A11)*B00 contributes to C10
    # m3: U covers (0,0) with +1; V covers (1,0) with +1, (1,1) with -1; W covers (1,0) with +1
    #     i.e., A00*(B10-B11) contributes to C01
    # m4: U covers (1,1) with +1; V covers (0,1) with +1, (0,0) with -1; W covers (0,1) with +1
    #     i.e., A11*(B01-B00) contributes to C01 and C10
    # m5: U covers (0,0),(0,1) with +1; V covers (1,1) with +1; W covers (1,0) with -1, (1,1) with +1
    #     i.e., (A00+A01)*B11 contributes to C01 and C11
    # m6: U covers (1,0) with +1, (0,0) with -1; V covers (0,0),(1,0) with +1; W covers (1,1) with +1
    #     i.e., (A10-A00)*(B00+B10) contributes to C11
    # m7: U covers (0,1) with +1, (1,1) with -1; V covers (0,1),(1,1) with +1; W covers (0,0) with +1, (1,0) with +1
    #     i.e., (A01-A11)*(B01+B11) contributes to C00 and C10

    # Let me define these more carefully using the tensor equation:
    # sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for (i,j,k) in {0,1}^3

    # The 7 Strassen terms (with indices in 2x2x2 space):
    # Term 0: m1 = (A00+A11)*(B00+B11)
    #   U: U[0,0]=1, U[3,0]=1  (i=0,j=0 and i=1,j=1)
    #   V: V[0,0]=1, V[3,0]=1  (j=0,k=0 and j=1,k=1)
    #   W: W[0,0]=1, W[3,0]=1  (k=0,i=0 and k=1,i=1)
    #   Check: (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    #   (i=0,j=0,k=1): U[0,0]*V[1,0]*W[2,0] = 1*0*0 = 0 (should be 1) ✗
    #
    # Hmm, that's not right. Let me reconsider.
    #
    # In Strassen, m1 = (A00+A11)*(B00+B11)
    # This is used in: C00 = m1 + ... and C11 = m1 - ...
    # So m1 contributes to C00 and C11.
    #
    # In tensor form, the term for m1 has:
    # U[i,j] nonzero for (i,j) = (0,0) and (1,1)
    # V[j,k] nonzero for (j,k) = (0,0) and (1,1)
    # W[k,i] nonzero for (k,i) = (0,0) and (1,1)
    #
    # The product U[i,j]*V[j,k]*W[k,i] is nonzero when:
    # (i,j) in {(0,0),(1,1)} AND (j,k) in {(0,0),(1,1)} AND (k,i) in {(0,0),(1,1)}
    #
    # Case (i,j)=(0,0): j=0, so (j,k) requires k=0. Then (k,i)=(0,0) ✓. So (i,j,k)=(0,0,0) gets contribution 1*1*1=1.
    # Case (i,j)=(1,1): j=1, so (j,k) requires k=1. Then (k,i)=(1,1) ✓. So (i,j,k)=(1,1,1) gets contribution 1*1*1=1.
    #
    # So m1 contributes 1 to (0,0,0) and 1 to (1,1,1).
    #
    # But we need ALL 8 entries (i,j,k) in {0,1}^3 to sum to 1.
    # m1 only covers 2 of them. The other 6 terms must cover the rest.
    #
    # Let me list all 7 Strassen terms and their contributions:
    # m1 = (A00+A11)*(B00+B11): contributes to (0,0,0) and (1,1,1)
    # m2 = (A10+A11)*B00: U: (1,0),(1,1); V: (0,0); W: (0,1)
    #   (i,j)=(1,0): j=0, (j,k)=(0,0) => k=0, (k,i)=(0,1) ✓. So (1,0,0) gets 1*1*1=1.
    #   (i,j)=(1,1): j=1, (j,k) requires (1,k)=(0,0) impossible. So only (1,0,0).
    # m3 = A00*(B10-B11): U: (0,0); V: (1,0) with +1, (1,1) with -1; W: (1,0)
    #   (i,j)=(0,0): j=0, (j,k) can be (1,0) or (1,1)? No, j must match. (j,k) has j=0, but V has j=1. Mismatch!
    #
    # I think I'm making an error. Let me restart the Strassen analysis.
    #
    # In matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
    # The tensor T[i,j,k] = 1 means the term A[i,j]*B[j,k] appears in C[i,k].
    #
    # For Strassen 2x2x2:
    # m1 = (A00+A11)*(B00+B11)
    # This product, when expanded, gives A00*B00 + A00*B11 + A11*B00 + A11*B11.
    # In C00, we use m1 to help compute A00*B00 + A01*B10 + ... 
    # Actually C00 = A00*B00 + A01*B10.
    #
    # The Strassen formula:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # So m1 appears in C00 (coeff +1) and C11 (coeff +1).
    # m2 appears in C10 (coeff +1) and C11 (coeff -1).
    # m3 appears in C01 (coeff +1) and C11 (coeff +1).
    # m4 appears in C00 (coeff +1) and C10 (coeff +1).
    # m5 appears in C00 (coeff -1) and C01 (coeff +1).
    # m6 appears in C11 (coeff +1).
    # m7 appears in C00 (coeff +1).
    #
    # Now in tensor decomposition form:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # For m1: It represents (A00+A11)*(B00+B11).
    # U for m1: U[0,0]=1, U[1,1]=1 (the A factors)
    # V for m1: V[0,0]=1, V[1,1]=1 (the B factors)
    # W for m1: W[0,0]=1 (for C00), W[1,1]=1 (for C11)
    #
    # Check (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # Check (i=0,j=0,k=1): U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # Check (i=1,j=1,k=0): U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0
    # Check (i=1,j=1,k=1): U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1
    #
    # So m1 contributes 1 to (0,0,0) and (1,1,1). Good.
    #
    # For m2 = (A10+A11)*B00:
    # U: U[1,0]=1, U[1,1]=1
    # V: V[0,0]=1
    # W: W[0,1]=1 (C10), W[1,1]=-1 (C11)
    #
    # Check (i=1,j=0,k=0): U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1
    # Check (i=1,j=1,k=0): U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0
    # Check (i=1,j=0,k=1): U[1,0]*V[0,1]*W[1,1] = 1*0*(-1) = 0
    # Check (i=1,j=1,k=1): U[1,1]*V[1,1]*W[1,1] = 1*0*(-1) = 0
    #
    # Hmm, m2 only contributes to (1,0,0). But it should also affect (1,1,1) with -1.
    # The issue is that V[1,1] = 0 for m2, so (i=1,j=1,k=1) doesn't get a contribution.
    #
    # I think the issue is that in the tensor decomposition, each rank-1 term
    # U_r ⊗ V_r ⊗ W_r must satisfy the equation for ALL (i,j,k).
    # The Strassen terms don't individually correspond to simple rank-1 tensors
    # in the (i,j,k) space because the "multiplication" combines multiple (i,j,k) entries.
    #
    # Let me think about this differently. The tensor T for 2x2x2 has T[i,j,k]=1 for all 8 entries.
    # A rank-1 tensor is U[i,j]*V[j,k]*W[k,i] for fixed r.
    #
    # For m1 = (A00+A11)*(B00+B11):
    # This is a bilinear form that equals A00*B00 + A00*B11 + A11*B00 + A11*B11.
    # As a trilinear form in (A, B, C^T), it contributes to:
    # - C00: A00*B00 (coeff 1) and A11*B11 (coeff 1) -- but these are different (i,j,k) entries
    #
    # I think the correct interpretation is:
    # m1 as a rank-1 tensor has:
    # U[0,0]=1, U[1,1]=1
    # V[0,0]=1, V[1,1]=1  
    # W[0,0]=1, W[1,1]=1
    #
    # And this contributes 1 to T[0,0,0] and 1 to T[1,1,1], and 0 elsewhere.
    #
    # Similarly for other terms. The 7 terms together must sum to 1 for all 8 (i,j,k).
    #
    # Let me verify all 7 Strassen terms:
    # m1: U=(0,0;1,1), V=(0,0;1,1), W=(0,0;1,1) -> contributes to (0,0,0) and (1,1,1)
    # m2: U=(1,0;1,1), V=(0,0), W=(0,1;-1 for C11) 
    #     U[1,0]=1, U[1,1]=1; V[0,0]=1; W[0,1]=1, W[1,1]=-1
    #     (1,0,0): 1*1*1=1; (1,1,0): 1*0*1=0; (1,0,1): 1*0*(-1)=0; (1,1,1): 1*0*(-1)=0
    #     Only (1,0,0) gets 1.
    #
    # This isn't working. Let me look up the actual Strassen tensor decomposition.
    #
    # Actually, I think the correct approach is:
    # The 7 Strassen multiplications define 7 rank-1 tensors where:
    # Each tensor r has U_r, V_r, W_r as 2x2 matrices.
    # The sum of all 7 outer products equals the all-ones tensor.
    #
    # Let me just directly define the 7 rank-1 tensors for 2x2x2:
    #
    # r=0: U = [[1,0],[0,1]], V = [[1,0],[0,1]], W = [[1,0],[0,1]]
    #   Contributes: (0,0,0):1, (1,1,1):1
    #
    # r=1: U = [[0,0],[1,1]], V = [[1,0],[0,0]], W = [[0,1],[0,-1]]
    #   (1,0,0): 1*1*1=1; others 0
    #
    # r=2: U = [[1,0],[0,0]], V = [[0,0],[1,-1]], W = [[0,0],[1,0]]
    #   (0,1,0): 1*1*1=1; (0,1,1): 1*(-1)*0=0
    #   Only (0,1,0) gets 1.
    #
    # This is getting complicated. Let me just use a verified explicit construction.
    #
    # I'll use the following known rank-7 decomposition for 2x2x2:
    # (Coefficients for U, V, W as 2x2 matrices, indexed [row][col])
    #
    # Term 0: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
    # Term 1: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,1],[0,-1]]
    # Term 2: U=[[1,0],[0,0]], V=[[0,0],[1,-1]], W=[[0,0],[1,0]]
    # Term 3: U=[[0,0],[0,1]], V=[[0,1],[-1,0]], W=[[0,1],[0,0]]
    # Term 4: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[0,-1],[1,0]]
    # Term 5: U=[[1,0],[1,0]], V=[[1,0],[1,0]], W=[[0,0],[0,1]]
    # Term 6: U=[[0,1],[0,1]], V=[[0,1],[0,1]], W=[[1,0],[1,0]]

    # Let me verify term 0:
    # (i,j,k)=(0,0,0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # (i,j,k)=(0,0,1): U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # (i,j,k)=(0,1,0): U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0
    # (i,j,k)=(0,1,1): U[0,1]*V[1,1]*W[1,0] = 0*1*0 = 0
    # (i,j,k)=(1,0,0): U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
    # (i,j,k)=(1,0,1): U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
    # (i,j,k)=(1,1,0): U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0
    # (i,j,k)=(1,1,1): U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1
    # Sum so far: (0,0,0)=1, (1,1,1)=1, rest=0

    # Term 1: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,1],[0,-1]]
    # (1,0,0): U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1
    # (1,1,0): U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0
    # (1,0,1): U[1,0]*V[0,1]*W[1,1] = 1*0*(-1) = 0
    # (1,1,1): U[1,1]*V[1,1]*W[1,1] = 1*0*(-1) = 0
    # Only (1,0,0) gets 1.

    # After terms 0,1: (0,0,0)=1, (1,1,1)=1, (1,0,0)=1, rest=0

    # Term 2: U=[[1,0],[0,0]], V=[[0,0],[1,-1]], W=[[0,0],[1,0]]
    # (0,1,0): U[0,1]*V[1,0]*W[0,0] = 0*1*0 = 0
    # (0,1,1): U[0,1]*V[1,1]*W[1,0] = 0*(-1)*1 = 0
    # Only (0,0,k) possible since U[0,0]=1, but V[0,k]=0 for all k. So no contribution!
    #
    # This is wrong. My Strassen tensors are incorrect.
    #
    # Let me try a completely different approach. I'll just use schoolbook
    # but with a different code structure, since I'm struggling with the
    # Strassen tensor coefficients.

    # Schoolbook decomposition: rank = n*m*p = 72
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + i][idx] = 1.0
                idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
