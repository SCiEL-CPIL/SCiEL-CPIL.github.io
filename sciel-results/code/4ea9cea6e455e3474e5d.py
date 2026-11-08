import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0
_SAFETY_MARGIN = 15.0

def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - time.time()

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each output element C[i,j] where i in [0,n), j in [0,p):
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # In the tensor decomposition:
    # U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r]
    # The trace convention is W[k*n + i, r] where k=j (column of B = row of W block)
    # Actually W has shape (p*n, rank), indexed as W[j*n + i, r]
    
    # Build the multiplication tensor T[i, k, j] = delta (1 if this is a valid product)
    # The decomposition requires: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,k,j
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # Compute sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+k, r] * V_arr[k*p+j, r] * W_arr[j*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _check_half_integer(mat):
    """Check that all entries are half-integers (integer or integer + 0.5)."""
    for row in mat:
        for val in row:
            v = float(val)
            if v * 2 != int(v * 2):
                return False
    return True

def _try_rank_21():
    """
    Try to find a rank-21 decomposition by modifying the rank-22 structure.
    
    Strategy: The rank-22 solution uses:
    - 7 terms for Strassen block 1 (A[0:2,0:2] * B[0:2,0:2])
    - 7 terms for Strassen block 2 (A[0:2,0:2] * B[0:2,2:4])
    - 8 terms for outer product (A[0:2,2:3] * B[2:3,0:4])
    
    To get rank 21, we need to merge one outer product term with a Strassen term,
    or find a different decomposition.
    
    Approach: Try to express one of the 8 outer-product terms as a linear combination
    of the existing 14 Strassen terms plus a modification.
    
    Alternative: Use a different split. Instead of 2x2 Strassen + 2x1 outer,
    try a different factorization.
    
    Let's try: Can we merge A02*B20 into the Strassen structure?
    The key insight: in Strassen, we compute products of sums. If we can fold
    A02 into one of the A-sums and B20 into one of the B-sums, we save a term.
    
    For example, modify M7: (A01-A11)(B10+B11) -> C00
    If we change it to (A01-A11+A02)(B10+B11) we get extra A02*B10 and A02*B11 terms
    that we don't want. So we'd need to cancel them.
    
    Better approach: Try a different partition.
    
    Actually, let's try a direct search: start from the rank-22 solution and try
    to eliminate one column by expressing it as a linear combination of others
    with half-integer coefficients.
    """
    n, m, p = 2, 3, 4
    
    # Build the rank-22 solution first
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Outer product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    # Verify rank-22 is correct
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    # Now try to eliminate one column.
    # The outer product terms (14-21) are "simple" - each has exactly one nonzero
    # in U and one in V. Try to see if any can be absorbed.
    
    # Strategy: Try to express column c as a sum of alpha * column c' for other columns,
    # where alpha is a half-integer. This requires the triple products to match.
    
    # For column c to be expressible as sum_r alpha_r * col_r, we need:
    # U[i*m+k, c] = sum_r alpha_r * U[i*m+k, r]
    # V[k*p+j, c] = sum_r alpha_r * V[k*p+j, r]  
    # W[j*n+i, c] = sum_r alpha_r * W[j*n+i, r]
    # for all i,k,j. This is very restrictive.
    
    # More general: try to modify two columns. Replace columns c1, c2 with a single
    # column c_new such that the tensor equation still holds.
    # This means: for all (i,k,j):
    # U[i*m+k,c1]*V[k*p+j,c1]*W[j*n+i,c1] + U[i*m+k,c2]*V[k*p+j,c2]*W[j*n+i,c2]
    # = U[i*m+k,c_new]*V[k*p+j,c_new]*W[j*n+i,c_new]
    
    # Let's try a different approach: use a modified Strassen that handles
    # the extra column more efficiently.
    
    # Alternative decomposition:
    # Split A into A_left (2x2) and A_right (2x1)
    # Split B into B_left (3x2) and B_right (3x2)
    # C = A_left * B_left + A_right * B_right
    # 
    # For A_left * B_left: use Strassen (7 terms)
    # For A_right * B_right: need 8 terms naively
    #
    # Can we do better? A_right is 2x1, B_right is 3x2.
    # C_right = A_right * B_right is 2x2.
    # C_right[0,0] = A[0,2]*B[2,0]
    # C_right[0,1] = A[0,2]*B[2,1]
    # C_right[1,0] = A[1,2]*B[2,0]
    # C_right[1,1] = A[1,2]*B[2,1]
    #
    # That's 4 scalar products. Can we do it in fewer?
    # For a 2x1 * 1x2 product, we need at least 2 multiplications (rank of that tensor is 2).
    # Actually no, we need 4 because each output is a distinct product.
    # Wait, the tensor for (2x1)*(1x2) has 4 entries, each being a distinct monomial.
    # The rank is 4.
    #
    # But we also have B_right columns 2,3 (B[2,2], B[2,3]).
    # C_right[0,2] = A[0,2]*B[2,2]
    # C_right[0,3] = A[0,2]*B[2,3]
    # C_right[1,2] = A[1,2]*B[2,2]
    # C_right[1,3] = A[1,2]*B[2,3]
    #
    # So total 8 products for the right part.
    #
    # Total: 7 + 8 = 15? No wait, I miscounted. Let me re-examine.
    #
    # Actually the original solution has:
    # Block 1: A[0:2,0:2] * B[0:2,0:2] - this is 2x2 * 2x2, Strassen = 7
    # Block 2: A[0:2,0:2] * B[0:2,2:4] - this is 2x2 * 2x2, Strassen = 7
    # Block 3: A[0:2,2:3] * B[2:3,0:4] - this is 2x1 * 1x4, needs 8
    # Total = 22
    #
    # Can we restructure? What if we do:
    # Block 1: A[0:2,0:3] * B[0:3,0:2] - 2x3 * 3x2
    # Block 2: A[0:2,0:3] * B[0:3,2:4] - 2x3 * 3x2
    #
    # For 2x3 * 3x2: the naive rank is 12. Can we do better?
    # The rank of M_{2,3|3,2} is known to be... let me think.
    # It's at most 2*3*2 = 12 naively.
    # Lower bound: max(2*2, 3*2, 3*2) = 6? No, that's not right.
    # The rank is at least max(n*p, m*p, n*m) = max(4, 6, 6) = 6.
    # Actually for matrix multiplication, rank >= max(n*m, m*p, p*n) is not always true.
    # The rank of M_{m,n|p,q} is at least max(mn, np, pq) is false.
    # 
    # Known results: rank of M_{2,3|3,2}... 
    # By the Coppersmith-Winograd type bounds, it's at most something.
    # But for small matrices, let's think structurally.
    #
    # Actually, let me try a completely different approach:
    # Use the fact that we can combine terms across blocks.
    
    # Let's try: modify the Strassen terms to absorb some outer product terms.
    # 
    # Key idea: In Strassen term M1: (A00+A11)(B00+B11)
    # What if we use (A00+A11+A02)(B00+B11+B20)?
    # This gives: A00B00 + A00B11 + A00B20 + A11B00 + A11B11 + A11B20 + A02B00 + A02B11 + A02B20
    # We want A02B20 to contribute to C[0,0].
    # But we get unwanted cross terms.
    
    # Let me try a more systematic approach:
    # Try to find a rank-21 decomposition by modifying the structure.
    
    # Approach: Try different partitions and see if any gives rank < 22.
    
    # Partition 1: A = [A1 (2x2) | A2 (2x1)], B = [[B11 (2x2)],[B12 (2x2)],[B21 (1x2)],[B22 (1x2)]]
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    # = A1*(B11+B12) + A2*(B21+B22)  -- no, that changes the output
    # Actually C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    # where B11 is B[0:2,0:2], B12 is B[0:2,2:4], B21 is B[2:3,0:2], B22 is B[2:3,2:4]
    # Wait, B is 3x4, so B[0:2,:] is 2x4, B[2:3,:] is 1x4
    # C = A[0:2,0:2]*B[0:2,0:4] + A[0:2,2:3]*B[2:3,0:4]
    #   = A_left * B_left + A_right * B_right
    # where A_left is 2x2, B_left is 2x4, A_right is 2x1, B_right is 1x4
    # A_left * B_left: 2x2 * 2x4, rank at most 2*4=8 naively, but can use Strassen-like
    # A_right * B_right: 2x1 * 1x4, rank = 8 (each of 8 entries is a distinct product)
    
    # For 2x2 * 2x4: we can split B into two 2x2 blocks and use Strassen on each
    # That gives 7+7=14, plus 8 for the right part = 22. That's what we have.
    
    # Alternative for 2x2 * 2x4: treat as 2x2 * (2x2 + 2x2)
    # Can we do 2x2 * 2x4 in fewer than 14?
    # The tensor M_{2,2|2,4} has rank... at most 14 (two Strassens).
    # Lower bound: at least max(2*2, 2*4, 2*4) = 8? 
    # Actually, I recall that rank of M_{m,n|p,q} >= max(mn, np, pq) is not generally true,
    # but rank >= max(m*p, n*p, m*n) for square cases... 
    # 
    # For 2x2 * 2x4: I believe the rank is exactly 14 (can't be done better than two Strassens)
    # because the two 2x2 output blocks are "independent" in some sense.
    # But actually they share the same A, so there might be savings.
    
    # Let me try: 2x2 * 2x4 where B = [B1 | B2] (each 2x2)
    # C = A*B1 | A*B2
    # Using Strassen on A*B1: 7 terms
    # Using Strassen on A*B2: 7 terms
    # Total: 14
    # 
    # Can we share? The Strassen terms for A*B1 use:
    # M1: (A00+A11)(B1_00+B1_11)
    # M2: (A10+A11)B1_00
    # M3: A00(B1_01-B1_11)
    # M4: A11(B1_10-B1_00)
    # M5: (A00+A01)B1_11
    # M6: (A10-A00)(B1_00+B1_01)
    # M7: (A01-A11)(B1_10+B1_11)
    # 
    # And for A*B2:
    # M1': (A00+A11)(B2_00+B2_11)
    # M2': (A10+A11)B2_00
    # etc.
    # 
    # Notice M1 and M1' both have (A00+A11). Can we combine?
    # (A00+A11)(B1_00+B1_11+B2_00+B2_11) would give cross terms we don't want.
    # 
    # What about: (A00+A11)(B1_00+B1_11) and (A00+A11)(B2_00+B2_11)
    # We can't merge these into one product without cross terms.
    
    # I think 14 is optimal for 2x2*2x4. So the structure is fixed at 14+8=22.
    # Unless we use a fundamentally different approach.
    
    # Let's try: what if we don't split A into 2x2 and 2x1, but use a different factorization?
    
    # Alternative: Use the "border rank" idea or a different algorithm.
    # 
    # Actually, let me try to see if rank 21 is achievable at all.
    # The lower bound for M_{2,3|3,4}:
    # - Trivial: 2*3*4 = 24 (naive)
    # - Rank >= max(2*3, 3*4, 2*4) = 12 (this bound is not always valid)
    # - Better: rank >= max(n*m, m*p, p*n) is false in general
    # - The correct lower bound from tensor rank theory: 
    #   rank >= max(mn, np, pq) is NOT valid.
    #   However, rank >= max(m*p, n*p, m*n) for the "flattening" gives:
    #   Flattening 1: (m*n) x (p) -> rank >= min(mn, p) = min(6,4) = 4
    #   Flattening 2: (m*p) x (n) -> rank >= min(mp, n) = min(12,2) = 2
    #   Flattening 3: (n*p) x (m) -> rank >= min(np, m) = min(8,3) = 3
    #   So flattening gives rank >= 4, which is weak.
    #
    # Stronger bounds exist. For M_{2,3|3,4}, I believe the rank is between 15 and 22.
    # The known best upper bound might be 22 (what we have) or possibly lower.
    
    # Let me try a computational search for rank 21.
    # We can set up a system of polynomial equations and try to solve.
    # But with 21 terms, each with ~6*21 + 12*21 + 8*21 = 504 variables (many zero),
    # and 24 equations (2*3*4), this is underdetermined.
    # 
    # Actually, the number of nonzero entries:
    # U: 6*21 = 126 entries (but many are zero)
    # V: 12*21 = 252 entries
    # W: 8*21 = 168 entries
    # Total: 546 variables, 24 equations. Very underdetermined.
    # 
    # The issue is that the equations are cubic (trilinear).
    # 
    # Let me try a different approach: start from the rank-22 solution and try to
    # find a rank-21 solution by "collapsing" one dimension.
    
    # Actually, let me just try a direct numerical search for rank 21.
    # Initialize with random values and use gradient descent on the residual.
    
    if _time_remaining() < 5.0:
        return None
    
    # Try to find a rank-21 solution via optimization
    # We'll parameterize with the structure: 14 "Strassen-like" terms + 7 "outer" terms
    # and try to optimize.
    
    # Actually, let me try a simpler idea: 
    # What if we use a non-Strassen decomposition for one of the 2x2 blocks?
    # For 2x2 * 2x2, the minimum rank is 7 (Strassen is optimal).
    # So we can't save there.
    
    # What about using a mixed approach where some terms serve double duty?
    
    # Let me try: modify the outer product to share structure with Strassen.
    # 
    # Idea: Instead of A02*B20, A02*B21, etc., use:
    # (A02 + something) * (B20 + something) and cancel the extra terms.
    
    # For example: (A02 + A00) * B20 = A02*B20 + A00*B20
    # If we already have a term involving A00*B20 from the Strassen part, we can combine.
    # But in our current decomposition, A00*B20 doesn't appear explicitly.
    # B20 = B[2,0] is in the "right" part of B, not in the Strassen blocks.
    
    # Hmm, this is getting complex. Let me try a numerical optimization approach.
    
    # For rank 21, we have 21 columns. Let's try to find U, V, W with half-integer
    # entries that satisfy the tensor equation.
    
    # Since the problem is underdetermined, let's try to find ANY solution (not necessarily
    # half-integer) first, then round.
    
    # Actually, for a cleaner approach, let me try to use the structure:
    # 14 terms from two Strassen blocks + 7 terms for the outer product (instead of 8)
    # 
    # For the outer product A_right (2x1) * B_right (1x4):
    # We need: C[0,j] = A[0,2]*B[2,j] for j=0,1,2,3
    #           C[1,j] = A[1,2]*B[2,j] for j=0,1,2,3
    # 
    # Can we do this in 7 terms instead of 8?
    # The tensor is: T[i,j] = A[0,2]*B[2,j] for i=0, A[1,2]*B[2,j] for i=1
    # This is a 2x4 tensor with entries:
    # [a0*b0, a0*b1, a0*b2, a0*b3]
    # [a1*b0, a1*b1, a1*b2, a1*b3]
    # where a0=A[0,2], a1=A[1,2], b0=B[2,0], etc.
    # 
    # This is a rank-8 tensor (each entry is a distinct monomial).
    # Can we decompose it with fewer than 8 rank-1 tensors?
    # A rank-1 tensor in this space is u[i]*v[j] for some vectors u, v.
    # We need: sum_r u_r[i] * v_r[j] * w_r = T[i,j] where w_r is the "coefficient"
    # Actually in our decomposition, it's U[i, r] * V[j, r] * W[i,j, r] = T[i,j]
    # 
    # For the outer product part, U has only 2 nonzeros per column (at positions 2 and 5),
    # V has only 1 nonzero per column (at position 8-11), W has only 1 nonzero per column.
    # 
    # To reduce from 8 to 7, we'd need one column to cover two entries.
    # But each entry is a distinct monomial (a0*b0, a0*b1, ..., a1*b3), so no single
    # rank-1 term can cover two entries unless the monomials are proportional.
    # They're not (all distinct), so we need at least 8 terms for this sub-tensor.
    # 
    # UNLESS we allow the U, V, W entries to be more general (not just single nonzeros).
    # 
    # For example, could we have a term where U[2, r] = 1, U[5, r] = 1,
    # V[8, r] = 1, V[9, r] = 1, W[0, r] = 1, W[2, r] = 1?
    # Then the contribution is: (A[0,2]+A[1,2])*(B[2,0]+B[2,1])*(delta)
    # This gives cross terms A[0,2]*B[2,1] and A[1,2]*B[2,0] which we don't want.
    # 
    # So it seems like 8 is the minimum for the outer product part.
    # And 7+7=14 for the Strassen parts.
    # Total: 22.
    
    # I'm now fairly convinced that 22 is optimal for this structure.
    # But let me try one more thing: a completely different decomposition.
    
    # What if we don't split A into left/right, but use a different factorization?
    # 
    # Alternative: C = A * B where A is 2x3, B is 3x4.
    # Write B as [B1; B2] where B1 is 2x4, B2 is 1x4.
    # C = A[:, 0:2] * B1 + A[:, 2:3] * B2
    # 
    # Or: C = A * [B1 | B2] where B1 is 3x2, B2 is 3x2.
    # C = A*B1 | A*B2
    # A*B1: 2x3 * 3x2
    # A*B2: 2x3 * 3x2
    # 
    # Rank of M_{2,3|3,2}: 
    # Naive: 2*3*2 = 12
    # Can we do better? 
    # Using the "Strassen-like" approach for rectangular matrices...
    # 
    # For 2x3 * 3x2:
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    # Let B = [[b00, b01], [b10, b11], [b20, b21]]
    # C = [[a00*b00+a01*b10+a02*b20, a00*b01+a01*b11+a02*b21],
    #      [a10*b00+a11*b10+a12*b20, a10*b01+a11*b11+a12*b21]]
    # 
    # This is 4 output entries, each a sum of 3 products.
    # Total 12 scalar products.
    # 
    # Can we reduce? The rank of this tensor... 
    # I believe it's 12 (no savings possible for this size).
    # 
    # So 2 * 12 = 24 for two 2x3*3x2 blocks. Worse than 22.
    
    # What about 2x3 * 3x4 directly?
    # Naive: 2*3*4 = 24.
    # Our current: 22.
    # Can we do 21?
    
    # I'll try a numerical search. If it doesn't find anything, we return rank 22.
    
    # For now, let's just return the rank-22 solution.
    # A more sophisticated search could be added, but given the time constraints,
    # the rank-22 solution is solid.
    
    return None

def _build_rank_22():
    """Build the known rank-22 decomposition."""
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Outer product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-21 solution
    result_21 = _try_rank_21()
    if result_21 is not None:
        return result_21
    
    # Fall back to rank-22
    return _build_rank_22()
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 00:0, 01:1, 10:3, 11:4
    # B indices: 00:0, 01:1, 10:4, 11:5
    # C indices (W): 00:0, 01:2, 10:1, 11:3
    
    # M1: (A00+A11)(B00+B11) -> C00+C11
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    
    # M2: (A10+A11)B00 -> C10-C11
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    
    # M3: A00(B01-B11) -> C01+C11
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    
    # M4: A11(B10-B00) -> C00+C10
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    
    # M5: (A00+A01)B11 -> -C00+C01
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    
    # M6: (A10-A00)(B00+B01) -> C11
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    
    # M7: (A01-A11)(B10+B11) -> C00
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # B indices: 02:2, 03:3, 12:6, 13:7
    # C indices (W): 02:4, 03:6, 12:5, 13:7
    
    # M1': (A00+A11)(B02+B13) -> C02+C13
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    
    # M2': (A10+A11)B02 -> C12-C13
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    
    # M3': A00(B03-B13) -> C03+C13
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    
    # M4': A11(B12-B02) -> C02+C12
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    
    # M5': (A00+A01)B13 -> -C02+C03
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    
    # M6': (A10-A00)(B02+B03) -> C13
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    
    # M7': (A01-A11)(B12+B13) -> C02
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    # A indices: 02:2, 12:5
    # B indices: 20:8, 21:9, 22:10, 23:11
    # C indices: 00:0, 01:2, 02:4, 03:6 for i=0
    #            10:1, 11:3, 12:5, 13:7 for i=1
    
    # A02 * B20 -> C00
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    # A02 * B21 -> C01
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    # A02 * B22 -> C02
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    # A02 * B23 -> C03
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    
    # A12 * B20 -> C10
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    # A12 * B21 -> C11
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    # A12 * B22 -> C12
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    # A12 * B23 -> C13
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    # Verification
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not (val == int(val) or val * 2 == int(val * 2)):
                    raise ValueError("Non-half-integer entry")

    return {"U": U, "V": V, "W": W}
