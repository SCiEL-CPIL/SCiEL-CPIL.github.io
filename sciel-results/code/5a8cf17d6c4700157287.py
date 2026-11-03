"""Rewritten program: structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)
import numpy as np


# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    # We seek a decomposition of the matrix multiplication tensor T_{a,b,c}
    # where a = i*m+j (i in 0..n-1, j in 0..m-1), b = j*p+k (j in 0..m-1, k in 0..p-1),
    # c = k*n+i (k in 0..p-1, i in 0..n-1).
    # T_{a,b,c} = 1 iff there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    #
    # Decomposition: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # i.e., T_{a,b,c} = sum_{r} U[a,r] * V[b,r] * W[c,r]
    #
    # Strategy: Build a dictionary of candidate rank-1 patterns (u, v, w) with small support,
    # then use a greedy set-cover / ILP-like approach to select a minimal subset that satisfies
    # all tensor equations exactly.

    # Generate candidate patterns:
    # For each (i, j, k) target entry, the "natural" pattern is e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i} (schoolbook).
    # But we can also use Strassen-like patterns that cover multiple entries with one rank-1 term.
    #
    # Key insight: A rank-1 term (u, v, w) satisfies T_{a,b,c} = sum_r U[a,r]V[b,r]W[c,r]
    # For a single term r, it contributes U[a,r]*V[b,r]*W[c,r] to entry (a,b,c).
    # If u, v, w have entries in {-1, 0, 1}, then each term contributes at most 1 in magnitude.
    #
    # Let's build a rich dictionary of candidates:
    # 1. Schoolbook terms: 72 terms, each with exactly one non-zero in each vector.
    # 2. Strassen-like terms: combinations that cover 2+ target entries.
    # 3. Random sparse terms with small support.

    # Build target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1

    # All entries that must be zero
    all_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                all_entries.add((a, b, c))

    # Generate candidate rank-1 terms
    candidates = []

    # Type 1: Schoolbook (single entry coverage)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u = [0] * nm
                v = [0] * mp
                w = [0] * pn
                u[a] = 1
                v[b] = 1
                w[c] = 1
                candidates.append((u, v, w))

    # Type 2: Pairwise Strassen-like patterns
    # For a fixed (j, k), we can try to cover multiple (i, j, k) entries with shared structure.
    # E.g., for fixed j, k: entries (i, j, k) for all i.
    # u has non-zeros at positions i*m+j for i in some subset S
    # v has non-zero at j*p+k
    # w has non-zeros at k*n+i for i in S
    # This gives contributions: for each i in S, U[i*m+j]*V[j*p+k]*W[k*n+i] = 1*1*1 = 1
    # But we need to ensure no cross-contamination.
    
    # Actually, let's think more carefully. A rank-1 term (u, v, w) contributes to entry (a, b, c)
    # the value u[a]*v[b]*w[c]. For this to be nonzero, we need u[a] != 0, v[b] != 0, w[c] != 0.
    #
    # For a "Strassen-like" term, we want it to contribute 1 to multiple target entries and 0 to others.
    #
    # Let's generate terms with small support (2-4 non-zeros per vector).
    for support_u in range(1, 5):
        for support_v in range(1, 5):
            for support_w in range(1, 5):
                if support_u * support_v * support_w > 20:
                    continue
                # Skip if this would just be schoolbook
                if support_u == 1 and support_v == 1 and support_w == 1:
                    continue
                
                # Generate random sparse patterns
                for _ in range(100):
                    # Choose support positions
                    u_pos = random.sample(range(nm), support_u)
                    v_pos = random.sample(range(mp), support_v)
                    w_pos = random.sample(range(pn), support_w)
                    
                    u = [0] * nm
                    v = [0] * mp
                    w = [0] * pn
                    for pos in u_pos:
                        u[pos] = random.choice([1, -1])
                    for pos in v_pos:
                        v[pos] = random.choice([1, -1])
                    for pos in w_pos:
                        w[pos] = random.choice([1, -1])
                    
                    # Check if this term is "useful": it should contribute nonzero to at least one target entry
                    # and ideally not contribute to too many non-target entries
                    target_hits = 0
                    non_target_hits = 0
                    for (a, b, c) in all_entries:
                        contrib = u[a] * v[b] * w[c]
                        if contrib != 0:
                            target_hits += 1
                    # Count non-target contributions
                    for a in range(nm):
                        if u[a] == 0:
                            continue
                        for b in range(mp):
                            if v[b] == 0:
                                continue
                            for c in range(pn):
                                if w[c] == 0:
                                    continue
                                if (a, b, c) not in all_entries:
                                    non_target_hits += 1
                    
                    # Keep if it hits some targets and doesn't hit too many non-targets
                    if target_hits >= 1 and non_target_hits <= 5:
                        candidates.append((u, v, w))

    # Also add some structured Strassen-like patterns
    # For 3x3x3, Strassen uses 7 terms. Let's add analogous patterns.
    # For fixed j, k: combine i=0,1,2 with signs
    for j in range(m):
        for k in range(p):
            # Pattern: u has entries at (0*m+j), (1*m+j), (2*m+j)
            # v has entry at (j*p+k)
            # w has entries at (k*n+0), (k*n+1), (k*n+2)
            for sign_pattern in [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                                  [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, 0],
                                  [1, 0, 1], [0, 1, 1], [1, -1, 0], [0, 1, -1]]:
                u = [0] * nm
                v = [0] * mp
                w = [0] * pn
                v[j * p + k] = 1
                for i, s in enumerate(sign_pattern):
                    if s != 0:
                        u[i * m + j] = s
                        w[k * n + i] = s
                # Check validity
                target_hits = 0
                non_target_hits = 0
                for (a, b, c) in all_entries:
                    contrib = u[a] * v[b] * w[c]
                    if contrib != 0:
                        target_hits += 1
                for a in range(nm):
                    if u[a] == 0:
                        continue
                    for b in range(mp):
                        if v[b] == 0:
                            continue
                        for c in range(pn):
                            if w[c] == 0:
                                continue
                            if (a, b, c) not in all_entries:
                                non_target_hits += 1
                if target_hits >= 1 and non_target_hits <= 10:
                    candidates.append((u, v, w))

    print(f"Generated {len(candidates)} candidate terms")

    # Now use a greedy algorithm to select a minimal subset
    # We need to satisfy: for each (a,b,c) in all_entries, sum_r U[a,r]V[b,r]W[c,r] = 1
    # And for each (a,b,c) NOT in all_entries, sum_r U[a,r]V[b,r]W[c,r] = 0
    
    # This is a system of linear equations in the selection variables x_r in {0,1}
    # But the equations are nonlinear in x_r because they involve products of selected terms.
    #
    # Alternative approach: directly construct the decomposition.
    # Let's use a different strategy: start with schoolbook and try to merge terms.
    
    # Actually, let's use a more direct approach:
    # We'll use the fact that the tensor has a lot of structure.
    # For 3x3x8, we can decompose as:
    # A(3x3) * B(3x8) = A(3x3) * [B1(3x3) | B2(3x3) | B3(3x2)]
    # = [A*B1 | A*B2 | A*B3]
    #
    # For A*B1 and A*B2 (each 3x3x3), we can use Strassen-like rank-23 decompositions.
    # For A*B3 (3x3x2), we can use a rank-18 or less decomposition.
    #
    # But combining them requires care. Let's just use a computational approach.
    
    # Given the time, let's use a practical approach:
    # 1. Start with schoolbook rank 72.
    # 2. Try to find linear dependencies among the 72 terms to reduce rank.
    # 3. If that fails, try random local search for rank < 72.
    
    # Actually, let's try a cleaner approach: use the known result that
    # 3x3x3 has rank 23 (Strassen), so 3x3x8 <= 2*23 + rank(3x3x2)
    # rank(3x3x2) <= 18, so total <= 64.
    #
    # Let's construct this explicitly.
    
    # Strassen 3x3x3 decomposition (rank 23):
    # I'll use a known set of 23 rank-1 terms for 3x3x3.
    # This is complex to hardcode, so let's use a simpler approach.
    
    # For now, let's use a randomized approach with verification.
    # Try to find a decomposition of rank R < 72.
    
    def verify_decomp(U, V, W, n, m, p):
        nm, mp, pn = n * m, m * p, p * n
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(R):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
        # Check non-target entries are zero
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Check if (a,b,c) is a target
                    is_target = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if a == i * m + j and b == j * p + k and c == k * n + i:
                                    is_target = True
                                    break
                            if is_target:
                                break
                        if is_target:
                            break
                    if is_target:
                        continue
                    s = 0
                    for r in range(R):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
        return True

    # Try rank 50 with random search
    best_rank = None
    for rank_try in [48, 50, 52, 54, 56, 60, 64]:
        for attempt in range(100):
            U = [[random.choice([0, 1, -1]) for _ in range(rank_try)] for _ in range(nm)]
            V = [[random.choice([0, 1, -1]) for _ in range(rank_try)] for _ in range(mp)]
            W = [[random.choice([0, 1, -1]) for _ in range(rank_try)] for _ in range(pn)]
            
            # Local search
            for it in range(3000):
                if verify_decomp(U, V, W, n, m, p):
                    best_rank = rank_try
                    return {"U": U, "V": V, "W": W}
                # Perturb
                if random.random() < 0.33:
                    a = random.randrange(nm)
                    r = random.randrange(rank_try)
                    U[a][r] = random.choice([0, 1, -1])
                elif random.random() < 0.5:
                    b = random.randrange(mp)
                    r = random.randrange(rank_try)
                    V[b][r] = random.choice([0, 1, -1])
                else:
                    c = random.randrange(pn)
                    r = random.randrange(rank_try)
                    W[c][r] = random.choice([0, 1, -1])
    
    # Fallback: schoolbook rank 72
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
