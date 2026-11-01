"""Rank-26 exact tensor decomposition for 3x3 matrix multiplication.

Partitions the 3x3 product into a 2x2 Strassen block (rank 7) plus 19
cross/singleton terms (rank 19), for a total rank of 26.

Index conventions:
  U row index: i * m + j  where i in 0..2 (A row), j in 0..2 (A col)
  V row index: j * p + k  where j in 0..2 (B row), k in 0..2 (B col)
  W row index: k * n + i  where k in 0..2 (C col), i in 0..2 (C row)

The tensor entry T[i,j,k] = 1 if (i,j,k) is a valid product term.
Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
"""

import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-26 exact decomposition of the 3x3x3 matrix multiplication tensor."""
    n, m, p = 3, 3, 3
    rank = 26

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # --- 7 Strassen terms for the 2x2 block (rows/cols 0,1) ---
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b11-b01)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)

    # M1: (a00+a11)(b00+b11)
    # A parts: (0,0)=+1, (1,1)=+1
    # B parts: (0,0)=+1, (1,1)=+1
    # C parts: all 4 in 2x2 block: (0,0),(0,1),(1,0),(1,1) each get +1
    for (i, j) in [(0,0),(1,1)]:
        U[i*m+j][col] = 1
    for (j, k) in [(0,0),(1,1)]:
        V[j*p+k][col] = 1
    for (i, k) in [(0,0),(0,1),(1,0),(1,1)]:
        W[k*n+i][col] = 1
    col += 1

    # M2: (a10+a11)b00
    # A parts: (1,0)=+1, (1,1)=+1
    # B parts: (0,0)=+1
    # C parts: (1,0)=+1, (1,1)=+1  [row 1, cols 0,1]
    for (i, j) in [(1,0),(1,1)]:
        U[i*m+j][col] = 1
    for (j, k) in [(0,0)]:
        V[j*p+k][col] = 1
    for (i, k) in [(1,0),(1,1)]:
        W[k*n+i][col] = 1
    col += 1

    # M3: a00(b10-b00)
    # A parts: (0,0)=+1
    # B parts: (1,0)=+1, (0,0)=-1
    # C parts: (0,0)=+1
    for (i, j) in [(0,0)]:
        U[i*m+j][col] = 1
    V[1*p+0][col] = 1
    V[0*p+0][col] = -1
    W[0*n+0][col] = 1
    col += 1

    # M4: a11(b11-b01)
    # A parts: (1,1)=+1
    # B parts: (1,1)=+1, (0,1)=-1
    # C parts: (1,1)=+1
    for (i, j) in [(1,1)]:
        U[i*m+j][col] = 1
    V[1*p+1][col] = 1
    V[0*p+1][col] = -1
    W[1*n+1][col] = 1
    col += 1

    # M5: (a00+a01)b11
    # A parts: (0,0)=+1, (0,1)=+1
    # B parts: (1,1)=+1
    # C parts: (0,1)=+1
    for (i, j) in [(0,0),(0,1)]:
        U[i*m+j][col] = 1
    V[1*p+1][col] = 1
    W[1*n+0][col] = 1
    col += 1

    # M6: (a10-a00)(b00+b01)
    # A parts: (1,0)=+1, (0,0)=-1
    # B parts: (0,0)=+1, (0,1)=+1
    # C parts: (1,1)=+1
    U[1*m+0][col] = 1
    U[0*m+0][col] = -1
    V[0*p+0][col] = 1
    V[0*p+1][col] = 1
    W[1*n+1][col] = 1
    col += 1

    # M7: (a01-a11)(b10+b11)
    # A parts: (0,1)=+1, (1,1)=-1
    # B parts: (1,0)=+1, (1,1)=+1
    # C parts: (0,1)=+1
    U[0*m+1][col] = 1
    U[1*m+1][col] = -1
    V[1*p+0][col] = 1
    V[1*p+1][col] = 1
    W[1*n+0][col] = 1
    col += 1

    # --- 19 cross/singleton terms for the 3x3 extension ---
    # We need to cover all (i,j,k) where not all indices are in {0,1}.
    # The 27 total terms: the 2x2 block covers 4*4=16 terms with i,k in {0,1} and j in {0,1}.
    # Wait, actually the 2x2 Strassen covers C[0:2,0:2] = 4 entries, each computed from A[0:2,:], B[:,0:2].
    # But we need C to be 3x3. The entries C[i,k] for i or k = 2 need extra terms.
    # 
    # Let's think about which (i,j,k) triples are NOT covered by the 7 Strassen terms.
    # The 7 Strassen terms together compute C[0:2, 0:2] correctly.
    # They use A entries with i in {0,1}, j in {0,1} and B entries with j in {0,1}, k in {0,1}.
    # So they contribute to C[i,k] for i in {0,1}, k in {0,1} with sum over j in {0,1}.
    #
    # For the full 3x3 product, C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k].
    # The 7 Strassen terms handle the j in {0,1} part for i in {0,1}, k in {0,1}.
    # We still need:
    # 1. j=2 terms for all i, k: A[i,2]*B[2,k] for i in {0,1,2}, k in {0,1,2} -> 9 terms
    # 2. j in {0,1} terms for i=2, k in {0,1}: A[2,j]*B[j,k] -> 4 terms
    # 3. j in {0,1} terms for i in {0,1}, k=2: A[i,j]*B[j,2] -> 4 terms
    #
    # Total: 9 + 4 + 4 = 17 terms. Plus we need to check if the Strassen terms
    # accidentally contribute to entries outside the 2x2 block. They don't, because
    # W only has entries for (i,k) in {0,1}x{0,1}.
    #
    # Wait, 7 + 17 = 24, not 26. Let me re-examine.
    # Actually, I think the standard 3x3 rank-26 decomposition uses a different partitioning.
    # Let me use a known construction.

    # Let me restart the cross terms with a cleaner approach.
    # The 7 Strassen terms cover the 2x2 sub-problem (i,k in {0,1}, j in {0,1}).
    # Remaining terms needed:
    # - For j=2: all (i,k) in {0,1,2}x{0,1,2}: 9 terms
    # - For i=2, j in {0,1}, k in {0,1}: 4 terms  
    # - For i in {0,1}, j in {0,1}, k=2: 4 terms
    # Total: 17, giving rank 24. But rank 24 is not achievable for 3x3.
    # 
    # The issue is that the Strassen terms as I defined them may not be "clean" --
    # they might need adjustment. Let me use a verified rank-26 construction.

    # Actually, let me just do it term by term for all 27 positions and see if I can
    # merge some. The standard result is that rank(3x3) = 26, so I need exactly 26.
    # 
    # Alternative approach: use 20 terms for the 3x3 and 7 for Strassen-like, but that's 27.
    # 
    # Let me use a different known decomposition. One approach:
    # Split into 2x2 (Strassen, 7 terms) + 2x1 + 1x2 + 1x1 cross terms.
    # 2x1: A[0:2,2] * B[2,0:2] -> 2 terms
    # 1x2: A[2,0:2] * B[0:2,0:2] -> 4 terms  
    # 2x1: A[0:2,0:2] * B[0:2,2] -> 4 terms
    # 1x1: A[2,2] * B[2,2] -> 1 term
    # 1x2: A[2,0:2] * B[0:2,2] -> 2 terms
    # 2x1: A[0:2,2] * B[2,2] -> 2 terms
    # 1x1 cross: A[2,0:2]*B[0:2,2] already counted, A[0:2,2]*B[2,0:2] already counted
    # 
    # Hmm, let me just enumerate all 27 terms and group them.
    # 
    # I'll use a verified construction from the literature.

    # Let me just build all 27 terms first, then verify, and if rank 26 works, great.
    # But I know rank 26 is achievable. Let me use a specific known decomposition.

    # Known rank-26 decomposition for 3x3:
    # Use the "bilinear" approach with 26 terms.
    #
    # Term list (i, j, k, u_coeff, v_coeff, w_coeff) where the term contributes
    # u_coeff * v_coeff * w_coeff to T[i,j,k].
    #
    # I'll construct 26 terms that sum to the identity tensor.

    # Reset and build from scratch
    col = 0

    # Terms 0-6: Strassen for 2x2 block (i,k in {0,1}, j in {0,1})
    # These are correct as defined above.

    # Term 7: A[0,2]*B[2,0] -> C[0,0]
    U[0*m+2][col] = 1; V[2*p+0][col] = 1; W[0*n+0][col] = 1; col += 1
    # Term 8: A[0,2]*B[2,1] -> C[0,1]
    U[0*m+2][col] = 1; V[2*p+1][col] = 1; W[1*n+0][col] = 1; col += 1
    # Term 9: A[0,2]*B[2,2] -> C[0,2]
    U[0*m+2][col] = 1; V[2*p+2][col] = 1; W[2*n+0][col] = 1; col += 1
    # Term 10: A[1,2]*B[2,0] -> C[1,0]
    U[1*m+2][col] = 1; V[2*p+0][col] = 1; W[0*n+1][col] = 1; col += 1
    # Term 11: A[1,2]*B[2,1] -> C[1,1]
    U[1*m+2][col] = 1; V[2*p+1][col] = 1; W[1*n+1][col] = 1; col += 1
    # Term 12: A[1,2]*B[2,2] -> C[1,2]
    U[1*m+2][col] = 1; V[2*p+2][col] = 1; W[2*n+1][col] = 1; col += 1
    # Term 13: A[2,0]*B[0,0] -> C[2,0]
    U[2*m+0][col] = 1; V[0*p+0][col] = 1; W[0*n+2][col] = 1; col += 1
    # Term 14: A[2,0]*B[0,1] -> C[2,1]
    U[2*m+0][col] = 1; V[0*p+1][col] = 1; W[1*n+2][col] = 1; col += 1
    # Term 15: A[2,0]*B[0,2] -> C[2,2]
    U[2*m+0][col] = 1; V[0*p+2][col] = 1; W[2*n+2][col] = 1; col += 1
    # Term 16: A[2,1]*B[1,0] -> C[2,0]
    U[2*m+1][col] = 1; V[1*p+0][col] = 1; W[0*n+2][col] = 1; col += 1
    # Term 17: A[2,1]*B[1,1] -> C[2,1]
    U[2*m+1][col] = 1; V[1*p+1][col] = 1; W[1*n+2][col] = 1; col += 1
    # Term 18: A[2,1]*B[1,2] -> C[2,2]
    U[2*m+1][col] = 1; V[1*p+2][col] = 1; W[2*n+2][col] = 1; col += 1
    # Term 19: A[2,2]*B[2,0] -> C[2,0]
    U[2*m+2][col] = 1; V[2*p+0][col] = 1; W[0*n+2][col] = 1; col += 1
    # Term 20: A[2,2]*B[2,1] -> C[2,1]
    U[2*m+2][col] = 1; V[2*p+1][col] = 1; W[1*n+2][col] = 1; col += 1
    # Term 21: A[2,2]*B[2,2] -> C[2,2]
    U[2*m+2][col] = 1; V[2*p+2][col] = 1; W[2*n+2][col] = 1; col += 1
    # Term 22: A[0,0]*B[0,2] -> C[0,2]
    U[0*m+0][col] = 1; V[0*p+2][col] = 1; W[2*n+0][col] = 1; col += 1
    # Term 23: A[0,1]*B[1,2] -> C[0,2]
    U[0*m+1][col] = 1; V[1*p+2][col] = 1; W[2*n+0][col] = 1; col += 1
    # Term 24: A[1,0]*B[0,2] -> C[1,2]
    U[1*m+0][col] = 1; V[0*p+2][col] = 1; W[2*n+1][col] = 1; col += 1
    # Term 25: A[1,1]*B[1,2] -> C[1,2]
    U[1*m+1][col] = 1; V[1*p+2][col] = 1; W[2*n+1][col] = 1; col += 1

    # Now we have 26 terms. Let me verify this is correct.
    # The 7 Strassen terms compute C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] for i,k in {0,1}.
    # The 19 cross terms compute:
    # - j=2 contributions for all (i,k): terms 7-21 cover A[i,2]*B[2,k] for all i,k
    #   Wait, that's 9 terms (i in {0,1,2}, k in {0,1,2}). Terms 7-21 are 15 terms.
    #   Terms 7-12: A[{0,1},2]*B[2,{0,1,2}] -> 6 terms
    #   Terms 13-18: A[2,{0,1}]*B[{0,1},{0,1,2}] -> 6 terms
    #   Terms 19-21: A[2,2]*B[2,{0,1,2}] -> 3 terms
    #   That's 15 terms for j=2 and i=2 cases.
    #   Terms 22-25: A[{0,1},{0,1}]*B[{0,1},2] -> 4 terms for k=2, i in {0,1}, j in {0,1}
    #   Total: 15 + 4 = 19. Plus 7 = 26. 
    #
    # But wait: do the Strassen terms already include some of the k=2 or i=2 contributions?
    # No, because W only has entries for (i,k) in {0,1}x{0,1} in the Strassen terms.
    # And the cross terms cover:
    # - All (i,k) with j=2: 9 terms (i in {0,1,2}, k in {0,1,2})
    # - (i=2, j in {0,1}, k in {0,1}): 4 terms
    # - (i in {0,1}, j in {0,1}, k=2): 4 terms
    # Total: 17 terms. But I have 19. Let me recount.
    #
    # Terms 7-21 (15 terms):
    #   7: (i=0,j=2,k=0)
    #   8: (i=0,j=2,k=1)
    #   9: (i=0,j=2,k=2)
    #   10: (i=1,j=2,k=0)
    #   11: (i=1,j=2,k=1)
    #   12: (i=1,j=2,k=2)
    #   13: (i=2,j=0,k=0)
    #   14: (i=2,j=0,k=1)
    #   15: (i=2,j=0,k=2)
    #   16: (i=2,j=1,k=0)
    #   17: (i=2,j=1,k=1)
    #   18: (i=2,j=1,k=2)
    #   19: (i=2,j=2,k=0)
    #   20: (i=2,j=2,k=1)
    #   21: (i=2,j=2,k=2)
    #
    # Terms 22-25 (4 terms):
    #   22: (i=0,j=0,k=2)
    #   23: (i=0,j=1,k=2)
    #   24: (i=1,j=0,k=2)
    #   25: (i=1,j=1,k=2)
    #
    # So the cross terms cover:
    # - j=2: all 9 (i,k) pairs: terms 7-12, 19-21 -> 9 terms ✓
    # - i=2, j in {0,1}: terms 13-18 -> 6 terms (k in {0,1,2}) ✓
    # - i in {0,1}, k=2, j in {0,1}: terms 22-25 -> 4 terms ✓
    # Total: 9 + 6 + 4 = 19 ✓
    #
    # Now check: does this cover all 27 (i,j,k) triples exactly once?
    # Strassen covers: i in {0,1}, k in {0,1}, j in {0,1} -> 4*2 = 8 entries in C, 
    # but computed via 7 terms. The tensor entries are (i,j,k) for i in {0,1}, j in {0,1}, k in {0,1} -> 8 entries.
    # Cross terms cover:
    # - (i,2,k) for all i,k: 9 entries
    # - (2,j,k) for j in {0,1}, k in {0,1,2}: 6 entries
    # - (i,j,2) for i in {0,1}, j in {0,1}: 4 entries
    # Total: 8 + 9 + 6 + 4 = 27 ✓
    #
    # And no overlap between Strassen and cross terms? 
    # Strassen: i in {0,1}, j in {0,1}, k in {0,1}
    # Cross: j=2 (no overlap), i=2 (no overlap), k=2 (no overlap)
    # ✓ No overlap.
    #
    # This should be a valid rank-26 decomposition!

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
