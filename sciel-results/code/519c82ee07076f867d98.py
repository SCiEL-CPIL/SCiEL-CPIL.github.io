"""Rewritten program for the suite task; maximizes comparable pairs in a 40-subset family on 12 vertices."""
import itertools
import time

def _count_comparable_pairs(family_masks):
    """Count unordered pairs (i, j) where mask_i is a strict subset of mask_j or vice versa."""
    n = len(family_masks)
    count = 0
    for i in range(n):
        mi = family_masks[i]
        for j in range(i + 1, n):
            mj = family_masks[j]
            if (mi & mj) == mi:
                # mi is subset of mj; check strict
                if mi != mj:
                    count += 1
            elif (mi & mj) == mj:
                # mj is subset of mi; check strict
                if mi != mj:
                    count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    n_vertices = 12
    m = 40

    # Strategy: Use a chain-like structure to maximize comparable pairs.
    # The maximum number of comparable pairs in a family of m sets is achieved
    # when the family is a chain (totally ordered by inclusion), giving m*(m-1)/2 pairs.
    # But we can't have a full chain of length 40 with 12 vertices because the max
    # chain length is 13 (from empty set to full set, adding one element at a time).
    
    # However, we can create multiple chains or a structure that is close to a chain.
    # A better approach: use a "near-chain" where we have sets of different sizes,
    # and within each size, we try to make them comparable with sets of other sizes.
    
    # Actually, the optimal structure for maximizing comparable pairs in a family of
    # m subsets of an n-element set is to take the m sets that are "closest" to being
    # a chain. One effective method is to use a greedy approach or dynamic programming.
    
    # For n=12, m=40, let's try a structured approach:
    # 1. Include the empty set (1 set)
    # 2. Include all singletons (12 sets) - each is comparable with empty set
    # 3. Include sets of size 2, 3, ..., up to some size, chosen to maximize comparability
    
    # A known result: the maximum number of comparable pairs is achieved by taking
    # sets from a "compressed" family. Let's use a greedy algorithm.
    
    # Greedy approach: start with empty set, then repeatedly add the set that
    # maximizes the number of new comparable pairs.
    
    # But for speed, let's use a heuristic:
    # Take sets in order of increasing size, and within each size, take sets that
    # share the most elements with previously chosen sets.
    
    # Actually, let's try a simpler approach: use all sets of size 0, 1, 2, ..., k
    # until we have 40 sets, choosing within each size to maximize comparability.
    
    # For n=12, the number of sets of size k is C(12, k):
    # C(12,0)=1, C(12,1)=12, C(12,2)=66, C(12,3)=220, ...
    # So we can take all sets of size 0 (1) and all sets of size 1 (12) = 13 sets.
    # We need 27 more. We can take 27 sets of size 2.
    
    # But to maximize comparability, we should choose the size-2 sets that are
    # supersets of the singletons we already have. Since all singletons are included,
    # any size-2 set is a superset of exactly 2 singletons.
    
    # Let's try: 1 empty + 12 singletons + 27 size-2 sets.
    # Each size-2 set is comparable with: empty (1) + 2 singletons = 3 sets.
    # Total comparable pairs: 
    # - Empty with 12 singletons: 12
    # - Empty with 27 size-2 sets: 27
    # - Each singleton with 27 size-2 sets that contain it: each singleton is in C(11,1)=11 size-2 sets, but we only have 27.
    
    # This is getting complex. Let me just use a greedy search with a time limit.
    
    # For now, let's use a simple but effective structure:
    # A "star" structure: pick one element (say 0), and include:
    # - Empty set
    # - All sets that contain 0, of various sizes
    # - Some sets that don't contain 0
    
    # Actually, the best approach for small n and m is to use dynamic programming or
    # a smart greedy. Given the time constraint, let's use a heuristic that works well.
    
    # Heuristic: Use a "colexicographic" ordering of sets and take the first 40.
    # This tends to create a structure with many comparable pairs.
    
    # Generate all 2^12 = 4096 subsets, order them by colexicographic order,
    # and take the first 40.
    
    # Colexicographic order: compare by the largest differing element.
    # For sets A and B, let k be the largest element where they differ.
    # A comes before B if k is in B but not in A.
    
    # Actually, let's use a simpler approach: order by (size, then lexicographic),
    # but that might not be optimal.
    
    # Let me try: order by size, and within each size, use a "nested" structure.
    
    # For size 0: {}
    # For size 1: {0}, {1}, {2}, ..., {11}
    # For size 2: {0,1}, {0,2}, {0,3}, ..., {0,11}, {1,2}, {1,3}, ..., {1,11}, ...
    # This creates a structure where sets of the same size are not comparable,
    # but sets of different sizes are often comparable.
    
    # Let's just generate all subsets, sort by (popcount, then integer value of mask),
    # and take the first 40.
    
    all_masks = []
    for mask in range(1 << n_vertices):
        all_masks.append(mask)
    
    # Sort by popcount, then by mask value
    all_masks.sort(key=lambda x: (bin(x).count('1'), x))
    
    selected_masks = all_masks[:m]
    
    # Convert to lists
    result = []
    for mask in selected_masks:
        subset = [i for i in range(n_vertices) if (mask >> i) & 1]
        result.append(subset)
    
    # Verify: exactly 40 distinct subsets
    assert len(result) == 40
    assert len(set(tuple(s) for s in result)) == 40
    
    # Check types
    for s in result:
        assert isinstance(s, list)
        for x in s:
            assert isinstance(x, int)
            assert not isinstance(x, bool)
            assert 0 <= x <= 11
    
    # Check distinct integers within each subset
    for s in result:
        assert len(s) == len(set(s))
    
    return result

# EVOLVE-BLOCK-END
