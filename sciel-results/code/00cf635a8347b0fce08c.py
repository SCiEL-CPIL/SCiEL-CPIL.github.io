"""Matrix multiplication 2x3x8 with rank-44 Strassen-based decomposition."""
import time
import random
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build the rank-44 baseline
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), split B1 into 4 blocks of 2 columns
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0:
                        continue
                    u_idx = li * m + la
                    U[u_idx][col] += u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0:
                        continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] += v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0:
                        continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] += w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    baseline = {"U": U, "V": V, "W": W}
    if _validate(baseline):
        best_rank = 44
    else:
        return _schoolbook()

    # Now try to reduce rank via local search:
    # Strategy: try to eliminate one rank-1 term at a time by adjusting
    # the remaining terms. For each candidate term to remove, solve for
    # whether the residual can be absorbed.
    
    # More practical: try pairwise merging. If two rank-1 terms u1*v1*w1 + u2*v2*w2
    # can be replaced by a single term u3*v3*w3, we reduce rank by 1.
    # This is rare, so instead try: can we set one entry of a term to zero
    # and compensate by adjusting another term?
    
    # Alternative structural approach: use the fact that the 2x1 x 1x8 part
    # (16 terms) might be compressible. The tensor for A2*B2 has entries
    # T[i,k] = A2[i] * B2[k] for i in {0,1}, k in {0..7}.
    # This is a rank-2 tensor in the (i,k) space with structure A[i]*B[k].
    # Actually it's already rank 16 in the full tensor sense because each
    # (i,k) pair needs its own rank-1 term in the full 2x3x8 tensor.
    
    # Let's try a different structural approach: use Kronecker product structure.
    # The 2x3x8 tensor = (2x2x8) + (2x1x8) where the first uses Strassen.
    # For the 2x1x8 part, note it's outer product structure.
    # Can we share Strassen terms across the boundary?
    
    # Try: instead of 4 independent 2x2x2 Strassen blocks, use a 2x2x8 
    # decomposition that might have lower rank than 4*7=28.
    # The 2x2x8 matrix multiplication tensor: what's its rank?
    # It's at most 2*2*8 = 32 schoolbook, or 4*7=28 via Strassen blocks.
    # Could it be lower? The 2x2xp tensor has known bounds.
    # For 2x2x2 it's 7. For 2x2x3, it's at most 7+6=13 or less.
    # Let's try to find a better 2x2x8 decomposition via search.
    
    # Actually, let's try a completely different structural approach:
    # Use the fact that we can permute which columns go to Strassen vs schoolbook.
    # What if we use Strassen on a different partition of the 3 columns?
    
    # Partition 1: cols 0,1 -> Strassen (2x2), col 2 -> schoolbook (2x1)
    #   Rank: 4*7 + 16 = 44
    # Partition 2: cols 0,2 -> Strassen-like, col 1 -> schoolbook
    #   Same structure, same rank.
    # Partition 3: Use a 2x3x2 block with rank r, then 2x0x6...
    #   2x3x2 has rank at most 2*3*2=12 schoolbook, or maybe less.
    #   If 2x3x2 has rank 10, then 2x3x2 + 2x3x6... no, that doesn't work.
    
    # Let me try: decompose as 2x3x2 (rank r1) + 2x3x6 (rank r2)
    # 2x3x2: schoolbook = 12. Can we do better?
    # The 2x3x2 tensor: T[i,a,k] = A[i,a]*B[a,k] for i in {0,1}, a in {0,1,2}, k in {0,1}
    # This is a 2x3x2 tensor. Its rank is at most 12.
    # Using Strassen on the 2x2 subpart (a=0,1) gives 7, plus 2*2=4 for a=2: total 11.
    # Or: use a 3x3x2 approach... 
    
    # Actually, let's try a randomized search for a rank-43 or rank-42 decomposition.
    # Start from rank 44, try to perturb and eliminate terms.
    
    # Simpler structural change: try all 3 ways to split 3 columns into (2,1)
    # and use Strassen on the 2-column part. All give rank 44.
    
    # New structural idea: use a 2x3x3 block with a known low-rank decomposition
    # and 2x3x5 with schoolbook.
    # 2x3x3: schoolbook = 18. Strassen-based: 2x2x3 (rank ~10-11) + 2x1x3 (rank 6) = ~16-17
    # 2x3x5: schoolbook = 30. Total: ~46-47. Worse.
    
    # Let's try: 2x3x4 + 2x3x4
    # 2x3x4: schoolbook = 24. Strassen: 2x2x4 (2*7=14) + 2x1x4 (8) = 22. Total: 44.
    # Same as before.
    
    # The structural change: use a randomized local search to try to find
    # rank 43 or 42 by perturbing the rank-44 solution.
    
    # For each pair of terms, try to see if their combined contribution
    # can be represented by modifying one of the remaining terms.
    
    # Practical approach: for each term c that we want to eliminate,
    # check if there exist scalar multipliers on other terms that
    # reproduce term c's contribution. This is a linear system.
    
    # Term c contributes: U[:,c] * V[:,c] * W[:,c] (outer product structure)
    # We need: sum_{c' != c} alpha_{c'} * U[:,c'] * V[:,c'] * W[:,c'] = U[:,c] * V[:,c] * W[:,c]
    # This is generally not solvable with a single multiplier per term.
    
    # Instead, let's try: can two terms be merged into one?
    # Term a + Term b = Term c?
    # U_a*V_a*W_a + U_b*V_b*W_b = U_c*V_c*W_c
    # This requires the sum of two rank-1 tensors to be rank-1, which is
    # only possible if they're linearly dependent (i.e., one is a scalar multiple
    # of the other in the appropriate sense).
    
    # Given the difficulty, let's try a different structural approach:
    # Use the fact that the tensor has symmetries and try to find
    # a decomposition with half-integer entries that might have lower rank.
    
    # Actually, the most promising structural change: use a different
    # algebraic identity. The 2x2x2 Strassen is unique (up to equivalence)
    # for rank 7. But for 2x2x3, there might be rank 10 or 11 decompositions
    # that are better than the naive 7+6=13 or 7+4+2=13.
    
    # Let me try to construct a 2x2x3 decomposition of rank 10 or 11.
    # 2x2x3: T[i,a,k] for i,a in {0,1}, k in {0,1,2}
    # = T[i,a,0] + T[i,a,1] + T[i,a,2] where each is 2x2x1 (rank 4 schoolbook)
    # Using Strassen on k=0: 7 terms for 2x2x1... no, 2x2x1 is just 4 terms.
    # Wait, 2x2x2 has rank 7. 2x2x1 has rank 4 (it's just 4 independent entries).
    # So 2x2x3 = 2x2x2 + 2x2x1 = 7 + 4 = 11.
    # Or 2x2x3 = 2x2x1 + 2x2x1 + 2x2x1 = 12.
    # So rank 11 is achievable for 2x2x3.
    
    # For our problem: 2x3x8 = (2x2x8) + (2x1x8)
    # 2x2x8 = 4 * (2x2x2) = 4*7 = 28, or 2x2x3 + 2x2x3 + 2x2x2 = 11+11+7 = 29. Worse.
    # Or 2x2x8 = 2x2x4 + 2x2x4 = 2*(2x2x2 + 2x2x2) = 2*14 = 28. Same.
    # Or 2x2x8 = 2x2x3 + 2x2x3 + 2x2x2 = 11+11+7 = 29. Worse.
    
    # What about 2x3x8 = 2x3x3 + 2x3x5?
    # 2x3x3: 2x2x3 (rank 11) + 2x1x3 (rank 6) = 17
    # 2x3x5: 2x2x5 + 2x1x5. 2x2x5 = 2x2x2+2x2x2+2x2x1 = 7+7+4=18. 2x1x5=10. Total 28.
    # Grand total: 17+28 = 45. Worse than 44.
    
    # 2x3x8 = 2x3x4 + 2x3x4
    # 2x3x4 = 2x2x4 + 2x1x4 = (7+7) + 8 = 22. Total: 44. Same.
    
    # It seems 44 is hard to beat with block decompositions.
    # Let's try a randomized perturbation approach to see if rank 43 is possible.
    
    # Approach: start with rank 44, try to set one entry to zero in a
    # "flexible" term and see if the rest can be adjusted.
    
    # For a more principled approach, let's try to find if there's a
    # rank-43 decomposition by solving a system of polynomial equations.
    # This is expensive, so let's use a heuristic: 
    # 
    # In the rank-44 solution, the 16 schoolbook terms (Part 2) are all
    # "pure" (each has exactly one nonzero in U, V, W). The 28 Strassen
    # terms have multiple nonzeros. 
    # 
    # Idea: can we merge one schoolbook term with a Strassen term?
    # Schoolbook term for (i=0, a=2, k=0): U[2][c]=1, V[16][c]=1, W[0][c]=1
    # A Strassen term might have U[2][c']=0 (since Strassen only uses a=0,1).
    # So no direct overlap. The Strassen terms don't touch row a=2 of U
    # or row a=2 of V.
    # 
    # This means the 16 schoolbook terms are in a completely independent
    # subspace from the 28 Strassen terms. No merging possible.
    # 
    # Therefore, the rank is at least 28 + 16 = 44 with this partition.
    # 
    # To get below 44, we need a fundamentally different partition where
    # the Strassen-like terms DO overlap with the "leftover" terms.
    # 
    # What if we don't cleanly separate the 3 columns? What if we use
    # a 2x3x2 block where the Strassen identity is applied to a
    # non-rectangular subset?
    
    # Actually, the key insight: the 2x3x8 tensor can be viewed as
    # A (2x3) times B (3x8). The tensor is T[i,a,k] = A[i,a]*B[a,k].
    # 
    # A rank-r decomposition is: T[i,a,k] = sum_{c=1}^r U[i,a,c]*V[a,k,c]*W[i,k,c]
    # 
    # For this to work, for each (i,a,k): sum_c U[i,a,c]*V[a,k,c]*W[i,k,c] = 1
    # For each (i,a,k) that doesn't exist (none here), sum = 0.
    # 
    # The constraint is: for ALL (i,a,k) with i in {0,1}, a in {0,1,2}, k in {0..7}:
    #   sum_c U[i,a,c]*V[a,k,c]*W[i,k,c] = 1
    # 
    # This is 48 equations. With r terms, we have 6*r + 24*r + 16*r = 48*r unknowns.
    # For r=44: 48*44 = 2112 unknowns, 48 equations. Very underdetermined.
    # 
    # The question is whether r=43 is possible. Let's try a randomized search.
    
    # Randomized local search: start from rank 44, randomly perturb entries,
    # then try to zero out one entire column (term) while maintaining validity.
    
    # Simpler: try to find if any single term in the rank-44 solution is
    # "redundant" in the sense that its contribution can be matched by
    # adjusting others. This is unlikely but let's check.
    
    # For a more systematic approach, let's try to construct a rank-43
    # decomposition by hand using a modified Strassen identity.
    
    # Modified idea: use 3 Strassen blocks for 6 of the 8 columns (rank 21)
    # and handle the remaining 2 columns + the a=2 row differently.
    # 2x3x2 (for cols 6,7) with a=0,1: Strassen rank 7.
    # 2x1x8 (for a=2): rank 16.
    # But we can share: the a=2 terms for cols 0-5 are separate.
    # Total: 7 (for cols 6,7, a=0,1) + 7+7 (for cols 0-3,4-5, a=0,1) + 16 (a=2) = 37.
    # Wait, that's the same as before: 4*7 + 16 = 44.
    
    # I think 44 might actually be minimal for this structure. Let's try
    # a randomized search to confirm or find a better solution.
    
    # Randomized search: try to find a rank-43 decomposition by
    # starting from rank 44 and trying to eliminate one term.
    
    # For each term c in the rank-44 solution, check if we can find
    # values for the other 43 terms (with small perturbations) such that
    # all 48 constraints are satisfied without term c.
    
    # This is a nonlinear system, but we can linearize around the current solution.
    
    # Let's try a different approach: use the fact that the problem is
    # underdetermined and try to find a solution with one fewer term by
    # solving a linear system after fixing most variables.
    
    # Practical approach: for rank r from 43 down to 40, try to find
    # a valid decomposition using random initialization and local search.
    
    # Given time constraints, let's try a quick randomized search for rank 43.
    
    # Actually, given the structural analysis above (Strassen terms don't
    # overlap with schoolbook terms), I believe 44 is the minimum for
    # this type of decomposition. A true improvement would require a
    # fundamentally different algebraic identity.
    
    # Let's return the rank-44 solution.
    return baseline
