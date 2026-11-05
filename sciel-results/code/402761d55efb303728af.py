"""Matrix multiplication tensor decomposition via minimum-cardinality exact cover.

Approach:
1. Enumerate all "admissible" rank-1 tensor supports: a rank-1 term with supports
   (Su, Sv, Sw) is admissible if for every (a, b, c) in Su x Sv x Sw, the tensor
   T[a, b, c] is non-zero (i.e., (a, b, c) is in the support of T).
2. For each admissible support, we can assign coefficients (1/2 or 1) to the vectors
   u, v, w such that the sum of the selected terms equals T exactly.
3. Since T only has entries 1 (and 0 elsewhere), and we're looking for exact decomposition,
   we need to find a set of rank-1 terms whose outer products sum to T.
4. We formulate this as an integer linear program: minimize the number of terms,
   subject to the constraint that the sum of the tensor entries from each term equals T.

For the 3x5x7 case, the support of T has exactly 105 non-zero entries.
Each rank-1 term contributes to a "combinatorial rectangle" of the support.

We use a greedy set-cover approach with local search to find a near-minimal decomposition.
"""

import numpy as np
from itertools import product
from collections import defaultdict
import time

DIMENSIONS = [3, 5, 7]


def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 15
    M = m * p  # 35
    P = p * n  # 21

    # Build the support of T as a set of (a, b, c) indices
    # T[i*m+j, j*p+k, k*n+i] = 1 for i in range(n), j in range(m), k in range(p)
    support_T = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support_T.add((a, b, c))

    total_entries = len(support_T)  # 105

    # Precompute: for each a in [0, N), which (b, c) pairs are in support_T
    # For a given a, support_T_a = {(b, c) : (a, b, c) in support_T}
    support_by_a = defaultdict(set)
    for (a, b, c) in support_T:
        support_by_a[a].add((b, c))

    # For a rank-1 term to be valid, we need:
    # Su x Sv x Sw ⊆ support_T
    # This means for all a in Su, b in Sv, c in Sw: (a, b, c) in support_T
    
    # Let's enumerate all possible rank-1 terms.
    # A rank-1 term is defined by (Su, Sv, Sw) where Su ⊆ [0,N), Sv ⊆ [0,M), Sw ⊆ [0,P)
    # and Su x Sv x Sw ⊆ support_T.

    # For efficiency, we'll enumerate small supports first.
    # Note: The schoolbook decomposition uses 105 terms, each with |Su|=1, |Sv|=1, |Sw|=1.
    # We want to find terms with larger supports to reduce the count.

    # Let's find all maximal "rectangles" in the support.
    # A rectangle is (A, B, C) where A x B x C ⊆ support_T and A, B, C are non-empty.

    # For a given A ⊆ [0,N), the possible (B, C) must satisfy:
    # B x C ⊆ ∩_{a in A} support_by_a[a]
    
    # This is getting complex. Let's use a simpler approach:
    # Enumerate all subsets A of [0,N) up to size 3, and for each, find the largest B x C.

    # Actually, let's just try to find a good decomposition using a greedy approach
    # combined with a small search.

    # First, let's find all "maximal" rank-1 terms.
    # A rank-1 term (Su, Sv, Sw) is maximal if we can't add any more elements to Su, Sv, or Sw.

    # For efficiency, let's enumerate all possible rank-1 terms with |Su| <= 2, |Sv| <= 2, |Sw| <= 2
    # and check if they are valid.

    valid_terms = []

    # For each subset Su of [0,N) with |Su| <= 2:
    for su_size in range(1, 3):
        for Su in product(range(N), repeat=su_size):
            Su = set(Su)
            if len(Su) != su_size:
                continue

            # Find the intersection of support_by_a for all a in Su
            # B x C must be a subset of this intersection
            if len(Su) == 1:
                a = list(Su)[0]
                bc_set = support_by_a[a]
            else:
                a1, a2 = list(Su)[:2]
                bc_set = support_by_a[a1] & support_by_a[a2]

            if not bc_set:
                continue

            # Now find all maximal B x C ⊆ bc_set
            # For simplicity, let's just use singletons and pairs
            for sv_size in range(1, 3):
                for Sv in product(range(M), repeat=sv_size):
                    Sv = set(Sv)
                    if len(Sv) != sv_size:
                        continue

                    # Check if all b in Sv have c values in bc_set that are consistent
                    # Actually, we need B x C ⊆ bc_set, which means for all b in B, c in C: (b, c) in bc_set
                    # Let's find the largest C for this B
                    if len(Sv) == 1:
                        b = list(Sv)[0]
                        c_candidates = {c for (b2, c) in bc_set if b2 == b}
                    else:
                        b1, b2 = list(Sv)[:2]
                        c1 = {c for (b2_, c) in bc_set if b2_ == b1}
                        c2 = {c for (b2_, c) in bc_set if b2_ == b2}
                        c_candidates = c1 & c2

                    if not c_candidates:
                        continue

                    for sw_size in range(1, 3):
                        for Sw in product(c_candidates, repeat=sw_size):
                            Sw = set(Sw)
                            if len(Sw) != sw_size:
                                continue

                            # Verify: Su x Sv x Sw ⊆ support_T
                            valid = True
                            for a in Su:
                                for b in Sv:
                                    for c in Sw:
                                        if (a, b, c) not in support_T:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if valid:
                                valid_terms.append((frozenset(Su), frozenset(Sv), frozenset(Sw)))

    # Deduplicate
    valid_terms = list(set(valid_terms))

    print(f"Found {len(valid_terms)} valid rank-1 terms")

    # Now we need to select a subset of these terms such that:
    # For each (a, b, c) in support_T, the sum of coefficients over terms containing (a,b,c) equals 1.
    # For simplicity, let's assume all coefficients are 1 (or 1/2, but let's start with 1).
    # This is a set cover problem: cover each (a,b,c) exactly once.

    # Build incidence: term t covers a set of (a,b,c) pairs
    term_covers = []
    for (Su, Sv, Sw) in valid_terms:
        covers = set()
        for a in Su:
            for b in Sv:
                for c in Sw:
                    covers.add((a, b, c))
        term_covers.append(covers)

    # Greedy set cover: repeatedly pick the term that covers the most uncovered elements
    uncovered = set(support_T)
    selected_terms = []
    term_index = list(range(len(valid_terms)))

    while uncovered:
        best_idx = -1
        best_count = 0
        for idx in term_index:
            count = len(term_covers[idx] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1 or best_count == 0:
            break
        selected_terms.append(best_idx)
        uncovered -= term_covers[best_idx]
        term_index.remove(best_idx)

    print(f"Greedy selected {len(selected_terms)} terms")

    # Check if we covered everything
    if uncovered:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Build the decomposition
    rank = len(selected_terms)
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]

    for col, idx in enumerate(selected_terms):
        Su, Sv, Sw = valid_terms[idx]
        for a in Su:
            U[a][col] = 1
        for b in Sv:
            V[b][col] = 1
        for c in Sw:
            W[c][col] = 1

    # Verify the decomposition
    # Check that the sum of outer products equals T
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0
                for col in range(rank):
                    val += U[a][col] * V[b][col] * W[c][col]
                if val != 1:
                    error += 1

    # Check off-support
    for a in range(N):
        for b in range(M):
            for c in range(P):
                if (a, b, c) not in support_T:
                    val = 0
                    for col in range(rank):
                        val += U[a][col] * V[b][col] * W[c][col]
                    if val != 0:
                        error += 1

    if error > 0:
        print(f"Verification failed with {error} errors, falling back to schoolbook")
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    print(f"Successful decomposition with rank {rank}")
    return {"U": U, "V": V, "W": W}
