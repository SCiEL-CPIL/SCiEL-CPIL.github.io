"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

DEADLINE = 115.0  # safety margin before 120s kill
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def construct():
    # We build a set of integer vectors in Z^11.
    # Constraint: for all distinct u, v in the set, ||u - v|| >= max_{w} ||w||.
    # Score = k (number of vectors).
    #
    # Strategy:
    # 1. Start with a small seed set that is known to satisfy the constraint.
    # 2. Use local search / greedy addition to expand the set.
    # 3. The key insight: if we scale all vectors by a factor, the constraint scales too.
    #    But we need integer vectors. So we work with integer vectors directly.
    #
    # Known good construction: vectors with entries in {-1, 0, 1} or small integers.
    # The seed program uses vectors with two non-zero entries (±1), giving norm sqrt(2).
    # There are C(11,2)*4 = 55*4 = 220 such vectors.
    #
    # Let's verify: for two such vectors u, v with u[i]=a, u[j]=b and v[k]=c, v[l]=d,
    # ||u-v||^2 = sum (u_m - v_m)^2.
    # If they share no indices: ||u-v||^2 = 2+2 = 4, so ||u-v|| = 2 >= sqrt(2). OK.
    # If they share one index i: u[i]=a, v[i]=c, and u[j]=b, v[k]=d (j!=k).
    #   ||u-v||^2 = (a-c)^2 + b^2 + d^2.
    #   a,c in {1,-1}, so (a-c)^2 is 0 or 4.
    #   If a=c: ||u-v||^2 = 0 + 1 + 1 = 2, ||u-v|| = sqrt(2) >= sqrt(2). OK.
    #   If a=-c: ||u-v||^2 = 4 + 1 + 1 = 6, ||u-v|| = sqrt(6) >= sqrt(2). OK.
    # If they share both indices: u[i]=a, u[j]=b, v[i]=c, v[j]=d.
    #   ||u-v||^2 = (a-c)^2 + (b-d)^2.
    #   If (a,b)=(c,d): same vector, not considered.
    #   If (a,b)=(-c,-d): ||u-v||^2 = (2a)^2 + (2b)^2 = 4+4=8, ||u-v||=2sqrt(2)>=sqrt(2). OK.
    #   If (a,b)=(c,-d): ||u-v||^2 = 0 + (2b)^2 = 4, ||u-v||=2>=sqrt(2). OK.
    #   If (a,b)=(-c,d): ||u-v||^2 = (2a)^2 + 0 = 4, ||u-v||=2>=sqrt(2). OK.
    #
    # So the 220 vectors from the seed are valid! Score = 220.
    #
    # Can we do better? Let's try to add vectors with three non-zero entries.
    # A vector with three ±1 entries has norm sqrt(3) > sqrt(2).
    # If we add such a vector w, we need ||w - u|| >= sqrt(3) for all existing u.
    # For u with two non-zero entries:
    #   If u and w share no indices: ||w-u||^2 = 3+2=5, ||w-u||=sqrt(5)>=sqrt(3). OK.
    #   If they share one index i: w[i]=a, u[i]=c, w has two other non-zero entries.
    #     ||w-u||^2 = (a-c)^2 + 1 + 1 + 1 (the remaining entries of w and u).
    #     Wait, let's be more careful. w has non-zero at i, p, q. u has non-zero at i, j.
    #     If j is not p or q: ||w-u||^2 = (a-c)^2 + w[p]^2 + w[q]^2 + u[j]^2 = (a-c)^2 + 1+1+1.
    #     If a=c: ||w-u||^2 = 0+3=3, ||w-u||=sqrt(3)>=sqrt(3). OK.
    #     If a=-c: ||w-u||^2 = 4+3=7, OK.
    #     If j is p (say): u has non-zero at i and p. w has non-zero at i, p, q.
    #     ||w-u||^2 = (a-c)^2 + (w[p]-u[p])^2 + w[q]^2.
    #     If a=c and w[p]=u[p]: ||w-u||^2 = 0+0+1=1, ||w-u||=1 < sqrt(3). FAIL!
    #
    # So we cannot freely add 3-entry vectors. The constraint is tight.
    #
    # Let's try a different approach: use the Leech lattice or other known constructions.
    # But for dimension 11, the kissing number is known to be 432 (for equal radius spheres).
    # However, we need integer vectors, and the constraint is ||u-v|| >= max ||w||.
    #
    # Let's try to build a larger set by considering vectors with entries in {-1, 0, 1}
    # and using a greedy algorithm to add vectors that satisfy the constraint.

    def _norm_sq(v):
        return int(np.dot(v, v))

    def _dist_sq(u, v):
        d = u - v
        return int(np.dot(d, d))

    # Start with the 220 two-entry vectors
    vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vectors.append(v)

    vectors = np.array(vectors, dtype=np.int64)
    k = len(vectors)
    max_norm_sq = max(_norm_sq(v) for v in vectors)  # = 2

    # Try to add vectors with three non-zero entries
    # We'll use a greedy approach: try each candidate, check if it satisfies the constraint
    candidates = []
    for i, j, k_idx in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k_idx] = sk
                    candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)

    # Greedy addition
    for c in candidates:
        if _time_left() < 5.0:
            break
        c_norm_sq = _norm_sq(c)
        if c_norm_sq > max_norm_sq:
            # Adding this would increase max_norm_sq, need to recheck all pairs
            # This is expensive, skip for now
            continue
        # Check if c can be added: ||c - v||^2 >= max_norm_sq for all v in vectors
        # Also need to check that adding c doesn't violate constraints among existing vectors
        # (it doesn't, since existing vectors already satisfy the constraint)
        diffs = vectors - c
        min_dist_sq = min(int(np.dot(d, d)) for d in diffs)
        if min_dist_sq >= max_norm_sq:
            vectors = np.vstack([vectors, c])
            k += 1

    # Try to add vectors with four non-zero entries (norm^2 = 4)
    # This would increase max_norm_sq to 4, so we need to recheck all pairs
    # This is expensive, so we'll skip unless we have lots of time
    if _time_left() > 30.0:
        # Try adding 4-entry vectors
        four_entry_candidates = []
        for idxs in combinations(range(D), 4):
            for signs in np.ndindex(2, 2, 2, 2):
                v = np.zeros(D, dtype=np.int64)
                for idx, sign in zip(idxs, signs):
                    v[idx] = 1 if sign == 0 else -1
                four_entry_candidates.append(v)
        four_entry_candidates = np.array(four_entry_candidates, dtype=np.int64)

        for c in four_entry_candidates:
            if _time_left() < 10.0:
                break
            c_norm_sq = _norm_sq(c)  # = 4
            # Check if c can be added with max_norm_sq = 4
            # Need ||c - v||^2 >= 4 for all v in vectors
            diffs = vectors - c
            min_dist_sq = min(int(np.dot(d, d)) for d in diffs)
            if min_dist_sq >= c_norm_sq:
                # Also need to check that all existing pairs satisfy ||u-v||^2 >= 4
                # This is expensive, so we'll skip this check for now
                # and just add c, then verify at the end
                vectors = np.vstack([vectors, c])
                k += 1

    # Verify the final set
    if len(vectors) > 0:
        max_norm_sq_final = max(_norm_sq(v) for v in vectors)
        valid = True
        for i in range(len(vectors)):
            if _time_left() < 2.0:
                break
            for j in range(i + 1, len(vectors)):
                if _dist_sq(vectors[i], vectors[j]) < max_norm_sq_final:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            # Fallback to the known valid 220 vectors
            vectors = []
            for i, j in combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        vectors.append(v)
            vectors = np.array(vectors, dtype=np.int64)

    return vectors
# EVOLVE-BLOCK-END
