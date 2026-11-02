"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W):
        """Check if the decomposition is exact using vectorized numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Build tensors: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
        # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # Compute T[i,j,k] = sum_t U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
        # Use einsum: T[i,j,k] = sum_t U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        # Check all entries equal 1
        return np.allclose(T, 1.0, atol=1e-9)
    
    def build_solution(blocks, schoolbook_terms):
        """Build U, V, W from Strassen blocks and schoolbook terms."""
        total_rank = len(blocks) * 7 + len(schoolbook_terms)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for (i0, j0, k0) in blocks:
            # Strassen M1-M7
            # M1 = (A00+A11)(B00+B11)
            c = col
            U[i0*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M2 = (A10+A11)B00
            c = col
            U[(i0+1)*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1; W[(k0+1)*n + (i0+1)][c] += -1
            col += 1
            # M3 = A00(B01-B11)
            c = col
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1; V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M4 = A11(B10-B00)
            c = col
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1; V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1; W[k0*n + (i0+1)][c] += 1
            col += 1
            # M5 = (A00+A01)B11
            c = col
            U[i0*m + j0][c] += 1; U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1; W[(k0+1)*n + i0][c] += 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            c = col
            U[(i0+1)*m + j0][c] += 1; U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1; V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
            c = col
            U[i0*m + (j0+1)][c] += 1; U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            col += 1
        
        for (i, j, k) in schoolbook_terms:
            c = col
            U[i*m + j][c] += 1
            V[j*p + k][c] += 1
            W[k*n + i][c] += 1
            col += 1
        
        return U, V, W
    
    def get_schoolbook_for_blocks(blocks):
        """Get schoolbook terms not covered by blocks."""
        covered = set()
        for (i0, j0, k0) in blocks:
            for i in (i0, i0+1):
                for j in (j0, j0+1):
                    for k in (k0, k0+1):
                        covered.add((i, j, k))
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
    
    def build_shared_solution(blocks, residual_terms, shared_assignments):
        """
        Build U, V, W where residual terms share rank factors with Strassen blocks.
        shared_assignments: list of (residual_term_idx, strassen_block_idx, strassen_factor_idx, scale)
        where the residual term (i,j,k) gets its contribution from factor (block_idx, factor_idx)
        with the given scale added to the existing Strassen factor.
        
        Actually, a cleaner approach: we build the Strassen factors first, then for each residual
        term, we either assign it a new factor or merge it into an existing factor.
        
        Let's use a different approach: build all factors as a list, where each factor has
        U_vec, V_vec, W_vec (sparse). Then we can try to merge factors.
        """
        # First, build Strassen factors
        factors = []  # Each factor: (U_dict, V_dict, W_dict)
        for (i0, j0, k0) in blocks:
            # M1
            u = {i0*m+j0: 1, (i0+1)*m+(j0+1): 1}
            v = {j0*p+k0: 1, (j0+1)*p+(k0+1): 1}
            w = {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
            factors.append((u, v, w))
            # M2
            u = {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1}
            v = {j0*p+k0: 1}
            w = {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1}
            factors.append((u, v, w))
            # M3
            u = {i0*m+j0: 1}
            v = {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1}
            w = {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
            factors.append((u, v, w))
            # M4
            u = {(i0+1)*m+(j0+1): 1}
            v = {(j0+1)*p+k0: 1, j0*p+k0: -1}
            w = {k0*n+i0: 1, k0*n+(i0+1): 1}
            factors.append((u, v, w))
            # M5
            u = {i0*m+j0: 1, i0*m+(j0+1): 1}
            v = {(j0+1)*p+(k0+1): 1}
            w = {k0*n+i0: -1, (k0+1)*n+i0: 1}
            factors.append((u, v, w))
            # M6
            u = {(i0+1)*m+j0: 1, i0*m+j0: -1}
            v = {j0*p+k0: 1, j0*p+(k0+1): 1}
            w = {(k0+1)*n+(i0+1): 1}
            factors.append((u, v, w))
            # M7
            u = {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1}
            v = {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1}
            w = {k0*n+i0: 1}
            factors.append((u, v, w))
        
        # Now add residual terms
        for (i, j, k) in residual_terms:
            u = {i*m+j: 1}
            v = {j*p+k: 1}
            w = {k*n+i: 1}
            factors.append((u, v, w))
        
        # Convert to matrices
        total_rank = len(factors)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        for t, (u, v, w) in enumerate(factors):
            for idx, val in u.items():
                U[idx][t] += val
            for idx, val in v.items():
                V[idx][t] += val
            for idx, val in w.items():
                W[idx][t] += val
        
        return U, V, W
    
    # Generate all possible disjoint Strassen block configurations
    all_possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_possible_blocks.append((i0, j0, k0))
    
    def are_disjoint(b1, b2):
        """Check if two Strassen blocks are disjoint."""
        return not (b1[0] in (b2[0], b2[0]+1) and b2[0] in (b1[0], b1[0]+1) and
                    b1[1] in (b2[1], b2[1]+1) and b2[1] in (b1[1], b1[1]+1) and
                    b1[2] in (b2[2], b2[2]+1) and b2[2] in (b1[2], b1[2]+1))
    
    def is_set_disjoint(blocks):
        """Check if a set of blocks are pairwise disjoint."""
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                if not are_disjoint(blocks[a], blocks[b]):
                    return False
        return True
    
    # Current best
    best_rank = 75  # Schoolbook
    best_solution = None
    initial_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    sb = get_schoolbook_for_blocks(initial_blocks)
    best_rank = len(initial_blocks) * 7 + len(sb)
    U, V, W = build_solution(initial_blocks, sb)
    if validate(U, V, W):
        best_solution = (U, V, W)
    
    # Try different block combinations
    from itertools import combinations
    for size in range(1, 7):
        if time.time() > deadline - 20:
            break
        for combo in combinations(all_possible_blocks, size):
            if not is_set_disjoint(list(combo)):
                continue
            sb = get_schoolbook_for_blocks(list(combo))
            rank = size * 7 + len(sb)
            if rank < best_rank:
                best_rank = rank
                U, V, W = build_solution(list(combo), sb)
                if validate(U, V, W):
                    best_solution = (U, V, W)
    
    # Now try the "shared factor" approach:
    # Use fewer Strassen blocks, but try to merge residual terms with Strassen factors
    # by adding small corrections to existing factors.
    # 
    # Key insight: A residual term (i,j,k) contributes U[i*m+j]*V[j*p+k]*W[k*n+i] = 1.
    # If we can find a Strassen factor t such that we can modify it to also contribute
    # to the residual term, we save one rank.
    #
    # More practically: try all subsets of blocks (size 2-4) and for the residual,
    # try to see if we can reduce the number of independent residual factors.
    
    # Alternative approach: Use a 2x5x5 Strassen-like decomposition on the first 2 rows
    # and handle the 3rd row separately, but with shared factors.
    
    # Let's try: partition into 2x5x5 (rows 0-1) and 1x5x5 (row 2)
    # For 2x5x5: use Strassen on 2x2x2 blocks within the 5x5
    # The 5x5 can be partitioned into 2x2 blocks + residual
    # 5 = 2+2+1, so we can place up to 4 disjoint 2x2x2 blocks in 2x5x5
    # (at positions (0,0,0), (0,2,0), (0,0,2), (0,2,2) in the 2x5x5 sub-tensor)
    
    # For the 1x5x5 slice (row 2), we need 25 schoolbook terms.
    # But we can try to share factors with the Strassen blocks.
    
    # Let's try a different strategy: use 3 Strassen blocks (not 4) and see if
    # the residual can be reduced.
    
    # Actually, let's try the approach where we use the full 2x5x5 with 4 Strassen blocks
    # (rank 28) and then the 1x5x5 residual has 25 terms, but some of those terms
    # might be mergeable with the Strassen factors.
    
    # A term (2, j, k) in the residual needs U[2*m+j], V[j*p+k], W[k*n+2] all nonzero.
    # The Strassen blocks only use rows 0 and 1 of A, so U[2*m+j] is 0 for all Strassen factors.
    # Therefore, no sharing is possible between the 1x5x5 residual and the Strassen blocks!
    # The 1x5x5 slice MUST use 25 independent factors.
    
    # So the minimum with this partition is 4*7 + 25 = 53.
    # But wait - can we do better with a different partition?
    
    # What if we use 2x4x4 Strassen (4 blocks, rank 28) and then handle the remaining
    # 2x5x5 - 2x4x4 = residual?
    # 2x5x5 has 50 terms, 2x4x4 has 32 terms, so residual is 18 terms.
    # Total: 28 + 18 = 46. But wait, the residual terms might overlap with Strassen factors.
    
    # Actually, let me reconsider. The 2x4x4 uses rows 0-1, cols 0-3 of B, cols 0-3 of C.
    # The residual is:
    # - 2x1x5 (cols 4 of B, all cols of C): 10 terms
    # - 2x5x1 (all cols of B, col 4 of C): 10 terms  
    # - But 2x1x1 is counted twice, so total = 10 + 10 - 2 = 18 terms
    # Total rank: 28 + 18 = 46
    
    # But can we do even better? What about 2x3x3 Strassen (4 blocks, rank 28)?
    # 2x3x3 has 18 terms, residual is 50-18=32. Total: 28+32=60. Worse.
    
    # What about using 2x2x2 blocks differently?
    # In 2x5x5, we can place 2x2x2 blocks at various positions.
    # Max disjoint 2x2x2 blocks in 2x5x5: 
    # i0 can be 0 only (since n=2)
    # j0 can be 0, 2 (since m=5, j0+1<=4, so j0<=3, but for disjointness j0 must differ by >=2)
    # k0 can be 0, 2 (similarly)
    # So max 4 blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2)
    
    # The 4-block solution for 2x5x5 covers:
    # Block (0,0,0): i={0,1}, j={0,1}, k={0,1}
    # Block (0,2,0): i={0,1}, j={2,3}, k={0,1}
    # Block (0,0,2): i={0,1}, j={0,1}, k={2,3}
    # Block (0,2,2): i={0,1}, j={2,3}, k={2,3}
    # Covered: i={0,1}, j={0,1,2,3}, k={0,1,2,3} = 2*4*4=32 terms
    # Residual: 50-32=18 terms (j=4 or k=4)
    # Total: 28+18=46
    
    # Now for the full 3x5x5, we need to handle row 2.
    # Row 2 terms: (2,j,k) for j=0..4, k=0..4 = 25 terms
    # These cannot share with the Strassen blocks (which only use rows 0,1 of A).
    # So total = 46 + 25 = 71. This is what the current program achieves.
    
    # Can we do better? What if we use a different decomposition that shares
    # factors between the 2x5x5 part and the 1x5x5 part?
    
    # Idea: Instead of separate Strassen blocks for rows 0-1, use a decomposition
    # that involves all 3 rows but with fewer total factors.
    
    # For example, a 3x2x2 "Strassen-like" decomposition:
    # We want to compute C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1,2 and k=0,1
    # with j=0,1
    # This is 3x2x2 = 12 terms in schoolbook.
    # Can we do it in fewer? The tensor rank of 3x2x2 matrix mult is known to be...
    # Actually, for M_{3,2,2}, the rank is at most 12 (schoolbook) and at least...
    # Let's just try to find a good decomposition.
    
    # Alternative: use the 3x5x5 as a whole with a smarter decomposition.
    # The key insight is that the 3x5x5 tensor has some structure we can exploit.
    
    # Let's try: decompose 3x5x5 as (3x2x5) + (0x3x5) where the first part
    # uses Strassen-like methods.
    
    # Actually, let's try a completely different approach:
    # Use 2x2x2 Strassen blocks but allow them to overlap in the i-dimension.
    # Wait, they can't overlap in i since n=3 and blocks are 2x2x2.
    
    # Let me try: use blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2) [4 blocks, rank 28]
    # and blocks at (1,0,0), (1,2,0), (1,0,2), (1,2,2) [4 more blocks, rank 28]
    # But these overlap in the i-dimension (both use i=0,1 or i=1,2).
    # They're not disjoint, so we can't just add them.
    
    # What if we use a "Strassen chain" approach?
    
    # Let's try a different partition: 3x5x5 = 3x3x3 + residual
    # 3x3x3 matrix mult has rank at most 23 (Bini et al.) or 21 (Landsberg).
    # But finding an explicit rank-21 decomposition is complex.
    # Let's just use schoolbook for 3x3x3: rank 27.
    # Residual: 3x5x5 - 3x3x3 = terms where j>=3 or k>=3
    # = 3*5*5 - 3*3*3 = 75 - 27 = 48 terms
    # Total: 27 + 48 = 75. No improvement.
    
    # What about 3x4x4 + residual?
    # 3x4x4: use 2x2x2 Strassen blocks. In 3x4x4, we can place blocks at:
    # (0,0,0), (0,2,0), (0,0,2), (0,2,2) - 4 blocks using rows 0-1
    # (1,0,0), (1,2,0), (1,0,2), (1,2,2) - 4 blocks using rows 1-2
    # But (0,*,*) and (1,*,*) overlap in row 1.
    # So we can only use one set: 4 blocks, rank 28.
    # Covered: 2x4x4 = 32 terms (rows 0-1, cols 0-3)
    # Residual in 3x4x4: 48-32=16 terms (row 2)
    # Then 3x5x5 residual: 75 - 48 = 27 terms (j=4 or k=4)
    # Total: 28 + 16 + 27 = 71. Same as before.
    
    # I think 71 might be close to optimal for this approach.
    # Let's verify the current solution is valid and try to find if there's a way
    # to get below 71.
    
    # One more idea: use a 2x5x5 decomposition with rank < 28 for the Strassen part.
    # The 2x5x5 tensor has 50 terms. Can we decompose it in fewer than 28+18=46?
    # The tensor rank of 2x5x5 matrix mult is unknown but bounded.
    # Schoolbook: 50. Strassen + residual: 46.
    # Can we do 45 or less?
    
    # Let's try: 2x5x5 = 2x3x3 + residual
    # 2x3x3: schoolbook rank 18. Or use 1x2x2 Strassen? No, 2x3x3 doesn't fit.
    # Actually, we can use a 2x2x2 Strassen block in 2x3x3: rank 7, covers 8 terms.
    # Residual in 2x3x3: 18-8=10 terms. Total for 2x3x3: 17.
    # Then 2x5x5 residual: 50-18=32 terms. Total: 17+32=49. Worse than 46.
    
    # Let's stick with 46 for 2x5x5 and 25 for 1x5x5, total 71.
    
    # But wait - can the 1x5x5 residual share factors with the 2x5x5 Strassen?
    # The 1x5x5 uses U[2*m+j], which is 0 in all Strassen factors (they only use rows 0,1).
    # So no sharing is possible.
    
    # Unless we modify the Strassen factors to also contribute to row 2!
    # This is the key idea: add small corrections to the Strassen factors to also
    # contribute to the 1x5x5 slice.
    
    # For example, Strassen M1 = (A00+A11)(B00+B11) contributes to C00 and C11.
    # We could modify it to also contribute to C20 or C21 by adding U[2*m+j] terms.
    # But this would break the exactness for the 2x2x2 sub-problem.
    
    # Unless we compensate with other factors. This is getting complex.
    
    # Let's try a simpler approach: use a different set of Strassen blocks that
    # covers more of the 3x5x5 tensor.
    
    # What if we use blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2) AND
    # (1,0,0), (1,2,0), (1,0,2), (1,2,2)?
    # These overlap in the i-dimension, so they're not disjoint.
    # But we can still use them if we carefully account for the overlap.
    
    # Actually, the issue is that if two blocks share an i-index, their U-factors
    # will interfere. We'd need to solve a system of equations.
    
    # For now, let's just return the best solution we've found.
    
    if best_solution is not None:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # Fallback
    U, V, W = build_solution(initial_blocks, get_schoolbook_for_blocks(initial_blocks))
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
