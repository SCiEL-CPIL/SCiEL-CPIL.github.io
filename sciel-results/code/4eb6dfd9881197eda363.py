"""Structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)
from itertools import product, combinations_with_replacement


# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    # Target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k; else 0
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c

    # Strategy: Generate a dictionary of candidate rank-1 terms (sparse patterns),
    # then use a greedy set-cover / IP approach to select minimal subset.

    # Step 1: Generate candidate terms.
    # Each candidate is a triple (u, v, w) of sparse vectors with entries in {-1, 0, 1}.
    # A good candidate should "cover" at least one target equation (contribute 1 to some T=1 entry)
    # while contributing 0 to all T=0 entries (or at least not conflicting).

    # For efficiency, we focus on candidates where:
    # - u has support of size 1-3 (indices i*m+j)
    # - v has support of size 1-3 (indices j*p+k)  
    # - w has support of size 1-3 (indices k*n+i)
    # And the "matching" condition: if u has entry at (i,j), v at (j',k), w at (k',i'),
    # then for the term to be non-degenerate we want i=i', j=j', k=k' for the diagonal,
    # but Strassen-like terms have off-diagonal structure.

    # Let's generate all "structured" candidates:
    # Type 1: Schoolbook terms (single entry in each vector) - 72 of them
    # Type 2: Strassen-like terms with 2-3 entries per vector

    candidates = []

    # Type 1: Schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                candidates.append((u, v, w))

    # Type 2: Strassen-like patterns for the 3x3x3 sub-blocks, extended to 3x3x8
    # For a 3x3x3 Strassen term, we have pairs of (i,j) in U, (j,k) in V, (k,i) in W
    # with 2 entries each. Let's enumerate all such patterns.

    # Generate all subsets of size 1 or 2 for U indices (i,j) pairs
    u_indices = list(range(nm))  # 0..8
    v_indices = list(range(mp))  # 0..23
    w_indices = list(range(pn))  # 0..23

    # For Strassen-like terms, the key insight is that:
    # (sum of products) = sum of individual products
    # So a term like (u1+u2)⊗(v1+v2)⊗(w1+w2) expands to 8 terms,
    # but we only want the "cross" terms to cancel or add constructively.

    # Let's use a more direct approach: enumerate sparse triples where
    # the bilinear form (u⊗v)·w pattern matches a subset of target entries.

    # For each candidate (u,v,w), compute which target entries it "hits" (contributes nonzero to)
    # and which non-target entries it hits (should be zero for exactness).

    def term_contribution(u, v, w):
        """Return dict: key=(a,b,c) -> value u[a]*v[b]*w[c] for nonzero products"""
        contrib = {}
        for a in range(nm):
            if u[a] == 0: continue
            for b in range(mp):
                if v[b] == 0: continue
                for c in range(pn):
                    if w[c] == 0: continue
                    val = u[a] * v[b] * w[c]
                    if val != 0:
                        contrib[(a,b,c)] = val
        return contrib

    def is_target(a, b, c):
        """Check if (a,b,c) is a target entry (should equal 1)"""
        # a = i*m+j, b = j*p+k, c = k'*n+i' where i=j', j=k', k=i'
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        return j == j2 and k == k2 and i == i2

    # Generate Strassen-like candidates for 3x3x3 blocks, replicated for 8 columns
    # For each pair of "diagonal" indices, create a Strassen term
    
    # Actually, let's use a smarter approach: 
    # Enumerate all triples of sparse vectors with |supp| <= 2 for u and w, |supp| <= 2 for v
    # that are "aligned" in the Strassen sense.

    # For 3x3 matrix multiplication, Strassen uses 7 terms.
    # For 3x3x8, we can think of it as 8 columns of B.
    # Let's generate candidates for each "column group".

    # Simpler: generate all candidates where u, v, w each have exactly 2 nonzero entries
    # (Strassen-like) and check if they're useful.

    # To keep it tractable, generate a limited set:
    # For each pair of i values, pair of j values, pair of k values:
    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2: continue
            for j1 in range(m):
                for j2 in range(m):
                    if j1 == j2: continue
                    for k1 in range(p):
                        for k2 in range(p):
                            if k1 == k2: continue
                            # Try pattern: u has entries at (i1,j1) and (i2,j2)
                            # v has entries at (j1,k1) and (j2,k2)
                            # w has entries at (k1,i1) and (k2,i2)
                            # With signs to make it Strassen-like
                            for s1 in [1, -1]:
                                for s2 in [1, -1]:
                                    u = [0]*nm; v = [0]*mp; w = [0]*pn
                                    u[i1*m+j1] = s1
                                    u[i2*m+j2] = s2
                                    v[j1*p+k1] = s1
                                    v[j2*p+k2] = s2
                                    w[k1*n+i1] = s1
                                    w[k2*n+i2] = s2
                                    candidates.append((u, v, w))

    # Also add mixed-sign Strassen patterns
    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2: continue
            for j1 in range(m):
                for j2 in range(m):
                    if j1 == j2: continue
                    for k1 in range(p):
                        for k2 in range(p):
                            if k1 == k2: continue
                            for s1 in [1, -1]:
                                for s2 in [1, -1]:
                                    u = [0]*nm; v = [0]*mp; w = [0]*pn
                                    u[i1*m+j1] = s1
                                    u[i2*m+j2] = -s2
                                    v[j1*p+k1] = s1
                                    v[j2*p+k2] = s2
                                    w[k1*n+i1] = s1
                                    w[k2*n+i2] = -s2
                                    candidates.append((u, v, w))

    # Deduplicate
    seen = set()
    unique_candidates = []
    for u, v, w in candidates:
        key = (tuple(u), tuple(v), tuple(w))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((u, v, w))
    candidates = unique_candidates

    # Step 2: For each candidate, compute its contribution vector
    # We need: sum_r contrib_r = target_tensor
    # Target tensor has 72 ones, rest zeros.

    # Build target as a sparse dict
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i*m+j, j*p+k, k*n+i)] = 1

    # For each candidate, precompute contribution
    cand_contribs = []
    for u, v, w in candidates:
        contrib = term_contribution(u, v, w)
        # Check: does this candidate only contribute to target entries?
        # If it contributes to any non-target entry, it's "dirty" and harder to use
        dirty = any(key not in target for key in contrib)
        # Count how many target entries it helps
        hits = sum(1 for key in contrib if key in target and contrib[key] == 1)
        # Count conflicts (contributes -1 to a target entry, or nonzero to non-target)
        conflicts = 0
        for key, val in contrib.items():
            if key in target:
                if val != 1:
                    conflicts += 1
            else:
                conflicts += 1
        cand_contribs.append((u, v, w, contrib, hits, conflicts, dirty))

    # Step 3: Greedy selection
    # Start with empty set, repeatedly pick the candidate that maximally reduces
    # the residual (unmet target entries) while minimizing conflicts.

    # Represent residual as a dict: target entries still need to be satisfied
    residual = dict(target)  # all target entries need value 1
    # Also track non-target entries that must remain 0
    # We'll track "bad" contributions to non-target entries

    selected = []
    current_sum = {}  # (a,b,c) -> current sum from selected candidates

    # Greedy: repeatedly pick best candidate
    # Score = number of target entries it correctly satisfies - penalty for conflicts
    
    max_iters = 200
    for _ in range(max_iters):
        if not residual:
            break
        best_score = -1
        best_idx = -1
        for idx, (u, v, w, contrib, hits, conflicts, dirty) in enumerate(cand_contribs):
            # Compute how this candidate would affect the residual
            score = 0
            # For each target entry, if currently 0 and candidate gives 1, that's good
            for key, val in contrib.items():
                if key in target:
                    current_val = current_sum.get(key, 0)
                    if current_val == 0 and val == 1:
                        score += 2  # satisfies an unmet target
                    elif current_val == 1 and val == 1:
                        score -= 3  # over-satisfies
                    elif current_val == 0 and val == -1:
                        score -= 1  # moves away from target
                    elif current_val == 1 and val == -1:
                        score -= 2  # unsatisfies a met target
                else:
                    # Non-target entry
                    current_val = current_sum.get(key, 0)
                    if current_val == 0 and val != 0:
                        score -= 1  # introduces a conflict
                    elif current_val != 0 and val != 0:
                        score -= 1  # adds to existing conflict
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx == -1 or best_score <= 0:
            break
        
        u, v, w, contrib, hits, conflicts, dirty = cand_contribs[best_idx]
        selected.append((u, v, w))
        for key, val in contrib.items():
            current_sum[key] = current_sum.get(key, 0) + val
        
        # Update residual
        for key in list(residual.keys()):
            if current_sum.get(key, 0) == 1:
                del residual[key]

    # Check if we got an exact solution
    exact = True
    for key, val in current_sum.items():
        if key in target:
            if val != 1:
                exact = False
                break
        else:
            if val != 0:
                exact = False
                break
    # Also check all target entries are satisfied
    for key in target:
        if current_sum.get(key, 0) != 1:
            exact = False
            break

    if exact and len(selected) < n*m*p:
        R = len(selected)
        U = [[0]*R for _ in range(nm)]
        V = [[0]*R for _ in range(mp)]
        W = [[0]*R for _ in range(pn)]
        for r, (u, v, w) in enumerate(selected):
            for a in range(nm):
                if u[a] != 0:
                    U[a][r] = u[a]
            for b in range(mp):
                if v[b] != 0:
                    V[b][r] = v[b]
            for c in range(pn):
                if w[c] != 0:
                    W[c][r] = w[c]
        return {"U": U, "V": V, "W": W}

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
