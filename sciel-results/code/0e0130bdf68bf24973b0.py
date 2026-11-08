"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin

    # Weight-4 vectors: all vectors with exactly 4 non-zero entries, each in {+1, -1}
    # There are C(11,4) * 2^4 = 330 * 16 = 5280 such vectors.
    # For any two distinct weight-4 vectors u, v with entries in {+1, -1}:
    #   ||u||^2 = ||v||^2 = 4
    #   |S_u ∩ S_v| = k, with k ∈ {0,1,2,3,4}
    #   dist^2 = (4-k) + (4-k) + sum_{i in S_u ∩ S_v} (u_i - v_i)^2
    #           = 2(4-k) + 4m  where m = number of sign disagreements on shared support
    #   Need dist^2 >= 4 (since max norm is 2, min pairwise distance >= 2 means dist^2 >= 4)
    #   8 - 2k + 4m >= 4  =>  4 - 2k + 4m >= 0  =>  2 - k + 2m >= 0
    #   k=4: 2-4+2m >= 0 => m >= 1 (at least one sign flip) -- always true for distinct vectors
    #   k=3: 2-3+2m >= 0 => 2m >= 1 => m >= 1 -- but m can be 0! If same signs on 3 shared,
    #        then dist^2 = 2(1) + 4(0) = 2 < 4. PROBLEM!
    #
    # Wait, let me recheck. If k=3 and m=0:
    #   dist^2 = 2(4-3) + 4(0) = 2. That's < 4. So we CANNOT take all weight-4 vectors.
    #
    # Hmm, so the previous reasoning was flawed. Let me reconsider.
    #
    # Actually wait - if k=3 and m=0, that means on the 3 shared positions the signs are identical,
    # and on the 1 non-shared position of u and 1 non-shared position of v, we have:
    #   dist^2 = 1^2 + 1^2 + 0 + 0 + 0 = 2. Yes, too small.
    #
    # So we need to prune. Let me think about which weight-4 vectors are mutually compatible.
    #
    # Two weight-4 vectors conflict iff dist^2 < 4, i.e., 8-2k+4m < 4, i.e., 2k-4m > 4, i.e., k-2m > 2.
    # Since k <= 4 and m <= k:
    #   k=4: 4-2m > 2 => m < 1 => m=0. Conflict if same signs on all 4 shared (i.e., same vector).
    #     But distinct vectors with same support must differ in at least one sign, so m>=1. No conflict.
    #   k=3: 3-2m > 2 => m < 0.5 => m=0. Conflict if same signs on all 3 shared positions.
    #   k=2: 2-2m > 2 => m < 0. Never.
    #   k=1: 1-2m > 2. Never.
    #   k=0: Never.
    #
    # So the ONLY conflicts are:
    #   1. Same vector (k=4, m=0) -- trivial, we don't have duplicates.
    #   2. k=3, m=0: three shared positions with identical signs.
    #
    # So we need to select a subset of the 5280 weight-4 vectors such that no two have
    # exactly 3 shared support positions with all 3 signs matching.
    #
    # This is an independent set problem in a conflict graph. Let's think about its structure.
    # Two vectors conflict if they share exactly 3 coordinates with the same signs.
    #
    # Alternative approach: use a smarter construction.
    #
    # Let's try: for each 4-subset S of {0,...,10}, take only vectors where the sum of
    # entries is 0 (i.e., two +1's and two -1's). That gives C(4,2) = 6 sign patterns per S.
    # Total: 330 * 6 = 1980 vectors.
    # Check: two vectors from same S (same 4 support): both have 2 plus, 2 minus.
    #   They share all 4 positions. m = number of sign disagreements.
    #   Since both have exactly 2 plus, if they're distinct, they differ in at least 2 positions (one goes +->-, other goes -->+). So m >= 2.
    #   dist^2 = 0 + 4m >= 8 >= 4. OK.
    # Two vectors from different S's: |S1 ∩ S2| <= 3.
    #   If k=3: need m >= 1 to avoid conflict. m=0 means same signs on 3 shared.
    #     With sum-zero constraint on 4 coords, can two vectors share 3 positions with same signs?
    #     Say S1 = {a,b,c,d}, S2 = {a,b,c,e}. On {a,b,c}, same signs.
    #     v1 has signs on a,b,c,d summing to 0. v2 has signs on a,b,c,e summing to 0.
    #     If signs on a,b,c are fixed (say +,+,+), then v1[d] = -3... not possible since entries are ±1.
    #     Actually if signs on a,b,c are (+,+,+), sum = 3, so d must be -1 to get sum 0? No, 3+(-1)=2≠0.
    #     Hmm, sum of 4 ±1 entries being 0 means exactly 2 plus and 2 minus.
    #     If 3 shared have the same signs in both vectors, and those 3 signs are, say, (+,+,+),
    #     then v1 needs one more entry (on d) and v2 needs one more (on e).
    #     v1: (+,+,+,x) sum=0 => x=-3... impossible.
    #     So 3 shared positions cannot all have the same sign if we need sum zero on 4 positions.
    #     What about 2 plus, 1 minus on the 3 shared? Then the 4th entry in v1 and v2 must be...
    #     v1: (+,+,-,x) sum=0 => x=-1. v2: (+,+,-,y) sum=0 => y=-1.
    #     So v1 = (+,+,-,-) on S1, v2 = (+,+,-,-) on S2. These have m=0 on the 3 shared, and they conflict!
    #     dist^2 = 1+1+0+0+0 = 2 < 4.
    #
    # So sum-zero doesn't fully work either.
    #
    # Let me try a different approach. What about using a greedy independent set on the conflict graph?
    # The conflict graph has 5280 nodes. Edges between vectors that share 3 support with same signs.
    #
    # Let me count expected degree. For a given vector with support S and signs s:
    # Conflicting vectors must have support S' with |S ∩ S'| = 3, and same signs on S ∩ S'.
    # Number of such S': choose 3 from S (C(4,3)=4), choose 1 from complement (7 options) = 4*7=28.
    # For each such S', the sign on the 4th position is free (2 options).
    # So degree ≈ 28*2 = 56.
    #
    # Average degree ~56 in a 5280-node graph. Max independent set is at least 5280/(56+1) ≈ 91 by greedy bound,
    # but likely much larger. Let's just do a greedy.
    #
    # Actually, let me try an even simpler approach first: can we use ALL weight-4 vectors with a fixed
    # "parity" constraint? E.g., product of all 4 signs = +1? That gives 8 per support, 330*8=2640.
    # Check k=3, m=0 conflict: if 3 shared have same signs, say (+,+,-), product of those 3 = -.
    # For v1 (support S1): 4th sign = +1/(-1) = -1 (to make product +1). So v1 on S1 = (+,+,-,-), product=+.
    # For v2 (support S2): 4th sign = +1/(-1) = -1. So v2 on S2 = (+,+,-,-), product=+.
    # They still conflict. So parity doesn't help.
    #
    # Let me just implement a greedy independent set with a good ordering.
    #
    # Actually, let me reconsider the problem. The task says "minimum pairwise distance is at least the maximum vector norm."
    # All vectors have norm 2 (since weight 4, entries ±1, norm^2 = 4, norm = 2).
    # So we need min pairwise distance >= 2, i.e., min dist^2 >= 4.
    #
    # Let me implement a greedy: order vectors, add if compatible with all already-added.
    # For speed, I'll use a bitset or set for conflict checking.
    #
    # Better idea: Let's think about what vectors are mutually compatible.
    # A vector is identified by (S, s) where S is a 4-subset and s is a sign vector on S.
    # Conflict: |S1 ∩ S2| = 3 and s1|_{S1∩S2} = s2|_{S1∩S2}.
    #
    # Key insight: if I fix the "type" of sign pattern more carefully...
    #
    # Let me try: for each 4-subset S, pick a canonical ordering, and only take vectors where
    # the first non-zero... no, this is getting complicated.
    #
    # Simplest robust approach: greedy independent set with random restarts or deterministic ordering.
    # With 5280 nodes and degree ~56, a simple greedy should find a large independent set quickly.
    #
    # Let me implement:
    # 1. Generate all 5280 vectors.
    # 2. Build adjacency/conflict structure.
    # 3. Greedy: iterate in some order, add vector if no conflict with already-added.
    #
    # For the conflict check to be fast, I'll precompute for each vector which other vectors it conflicts with.
    # But 5280 * 56 = ~296k edges, which is manageable.
    #
    # Actually, even simpler: for each vector, its conflicts are determined by its 4 triples (choosing 3 of 4 support elements).
    # For each triple T of support positions, and the signs on T, any vector that has T in its support with the same signs on T is a conflict.
    #
    # Let me build a dictionary: key = (frozenset of 3 positions, tuple of signs on those positions) -> list of vectors.
    # Two vectors conflict iff they share a key.
    #
    # Each vector has C(4,3)=4 triples. So total keys = 5280*4 = 21120 entries (with duplicates).
    # Group by key, and within each group, all vectors conflict with each other.
    #
    # This gives us a clean conflict structure!

    # Generate all weight-4 vectors
    vectors = []
    vector_to_id = {}
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            vid = len(vectors)
            vectors.append(v)
            vector_to_id[tuple(v)] = vid

    n = len(vectors)  # 5280

    # Build conflict groups: for each (triple, signs_on_triple), all vectors sharing that are mutually conflicting
    from collections import defaultdict
    triple_groups = defaultdict(list)
    for vid, v in enumerate(vectors):
        # Find support
        support = tuple(i for i in range(D) if v[i] != 0)
        # For each triple in support
        for triple in combinations(support, 3):
            signs_on_triple = tuple(v[i] for i in triple)
            key = (triple, signs_on_triple)
            triple_groups[key].append(vid)

    # Build conflict set for each vector
    # A vector conflicts with all others in any group it belongs to
    # Use a set of sets for efficient greedy
    conflict_sets = [set() for _ in range(n)]
    for key, vids in triple_groups.items():
        if len(vids) > 1:
            for vid in vids:
                for other in vids:
                    if other != vid:
                        conflict_sets[vid].add(other)

    # Check deadline
    if time.time() > deadline:
        # Fallback: just return first 24 vectors (weight-2)
        fallback = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    fallback.append(v)
        return np.array(fallback, dtype=np.int64)

    # Greedy independent set: try multiple orderings
    best_set = []
    best_orderings = [
        list(range(n)),
        sorted(range(n), key=lambda x: -len(conflict_sets[x])),  # most constrained first
        sorted(range(n), key=lambda x: len(conflict_sets[x])),   # least constrained first
    ]

    # Also try a few random shuffles
    rng = np.random.default_rng(42)
    for _ in range(5):
        order = list(range(n))
        rng.shuffle(order)
        best_orderings.append(order)

    for order in best_orderings:
        if time.time() > deadline:
            break
        selected = []
        selected_set = set()
        for vid in order:
            if vid in selected_set:
                continue
            # Check if vid conflicts with any selected
            if not any(other in selected_set for other in conflict_sets[vid]):
                selected.append(vid)
                selected_set.add(vid)
        if len(selected) > len(best_set):
            best_set = selected

    if time.time() > deadline:
        pass  # Use best found so far

    # Build result
    if best_set:
        result = np.array([vectors[i] for i in best_set], dtype=np.int64)
    else:
        # Fallback
        fallback = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    fallback.append(v)
        result = np.array(fallback, dtype=np.int64)

    return result
# EVOLVE-BLOCK-END
