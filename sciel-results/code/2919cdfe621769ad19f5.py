"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def _generate_seed_vectors():
    """Generate a small valid set as a fallback."""
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit

    m = 4
    max_ip_allowed = 2

    candidates = _generate_shell(m)
    N = len(candidates)

    # Precompute Gram matrix
    G = candidates @ candidates.T

    # Build conflict graph: edge between i and j if G[i,j] > max_ip_allowed (and i != j)
    # For MIS, we want the INDEPENDENT SET in the conflict graph,
    # which equals the CLIQUE in the compatibility graph.
    # Alternatively: find MIS in conflict graph directly.
    # Conflict graph adjacency: i ~ j if G[i,j] > max_ip_allowed, i != j

    # Build adjacency as bitsets for speed
    # N = 5280, so each adjacency row is 5280 bits = 660 bytes. Total ~3.5MB.
    # Use Python ints as bitsets.

    # conflict[i] is a set of j where G[i,j] > max_ip_allowed and j != i
    # For Bron-Kerbosch with pivot, we need efficient neighbor operations.

    # Let's use a different approach: build the COMPATIBILITY graph
    # compat[i] = set of j where G[i,j] <= max_ip_allowed and j != i
    # We want maximum clique in compat graph = maximum independent set in conflict graph.

    # For N=5280, building full adjacency sets in Python might be slow.
    # Let's use NumPy-based bitsets.

    # Actually, let's use a scipy-free approach with numpy boolean arrays.
    # conflict_mask[i, j] = True if G[i,j] > max_ip_allowed and i != j
    conflict_mask = (G > max_ip_allowed).astype(np.uint8)
    np.fill_diagonal(conflict_mask, 0)  # No self-conflict

    # For Bron-Kerbosch with pivot, we need:
    # - neighbors(P): set of nodes in P adjacent to v
    # - non_neighbors(P): set of nodes in P not adjacent to v
    # With bitsets, this is fast.

    # Let's represent sets as Python integers (bitsets).
    # N = 5280, so we need 5280-bit integers. Python handles big ints natively.

    # Build bitset adjacency for the COMPATIBILITY graph (we want max clique in this).
    # compat_bits[i] = bitmask where bit j is set if i and j are compatible (G[i,j] <= max_ip, i != j)

    # Convert conflict_mask rows to Python int bitsets.
    # This is the expensive part: 5280 rows of 5280 bits.

    # Actually, for Bron-Kerbosch with pivot on a 5280-node graph, even with bitsets,
    # the recursion depth and branching could be enormous. The graph is dense (most pairs are compatible).
    # Let's check: for m=4 shell in D=11, what's the average IP?
    # Two random m=4 vectors in D=11: they share on average 16/11 ≈ 1.45 positions.
    # IP = sum over shared positions of signs. Expected |IP| is small.
    # P(IP > 2) is not that high. Let's estimate.
    # Actually, let's just compute the density.

    # For now, let's try a different exact approach:
    # Use a greedy coloring + branch and bound for maximum independent set.
    # This is the standard approach for exact MIS.

    # But first, let's try a much faster heuristic that's still different from ILS:
    # **Greedy with multiple random orderings + local search with 2-swap moves**

    # Actually, let's try the exact MIS with a time limit, falling back to best heuristic.

    # For the exact approach, let's use a recursive branch-and-bound:
    # - Color the graph (compatibility graph) to get an upper bound on clique size
    # - Branch on vertices

    # Given N=5280, this is likely too slow for exact. Let's use a hybrid:
    # 1. Fast greedy to get a good lower bound
    # 2. Local search with 1-swap and 2-swap moves to improve
    # 3. Use the time budget to do many random restarts of (1+2)

    # This is still in the same family as the current ILS. Let me think differently.

    # **Different strategy: Use the structure of the shell-4 vectors.**
    # Shell-4 vectors in D=11 can be identified by (support set of size 4, sign pattern).
    # Two vectors conflict iff their inner product > 2.
    # IP = sum over intersection of supports of (sign_i * sign_j).
    # If supports are disjoint: IP = 0 (compatible)
    # If supports share 1 element: IP = ±1 (compatible)
    # If supports share 2 elements: IP = s1*s2 + s3*s4 where s1,s2,s3,s4 are ±1
    #   IP can be -2, 0, or 2. Compatible iff IP <= 2, which is always true.
    #   Wait: IP = ±1 ± 1, so IP ∈ {-2, 0, 2}. All ≤ 2. So always compatible!
    # If supports share 3 elements: IP = s1*s2 + s3*s4 + s5*s6, each term ±1.
    #   IP ∈ {-3, -1, 1, 3}. Compatible iff IP ≤ 2, so IP ≠ 3.
    #   IP = 3 iff all three sign products are +1.
    # If supports are identical (share 4 elements): IP = s1*s2 + s3*s4 + s5*s6 + s7*s8
    #   IP ∈ {-4, -2, 0, 2, 4}. Compatible iff IP ≤ 2, so IP ≠ 4.
    #   IP = 4 iff all four sign products are +1, i.e., the sign patterns are identical.

    # So the conflict condition is:
    # - Supports share 3 elements AND the 3 sign products are all +1 (IP=3)
    # - Supports are identical AND sign patterns are identical (IP=4)

    # This structure means the conflict graph has a special form!
    # Let's exploit this.

    # Two vectors conflict iff:
    # (a) They have the same support and same signs (IP=4), OR
    # (b) They share exactly 3 support elements and the signs on those 3 elements are the same (IP=3)

    # Wait, let me recheck (b): if they share 3 elements, the IP is the sum of 3 sign products.
    # For IP=3, all 3 sign products must be +1, meaning the signs match on all 3 shared positions.
    # The 4th element of each is in a different position, so it doesn't contribute.

    # This is a very structured conflict graph. Let's build it efficiently.

    # Index shell-4 vectors by (support_tuple, sign_tuple).
    # support_tuple: sorted tuple of 4 indices
    # sign_tuple: tuple of 4 signs (±1) corresponding to the support positions

    # Total: C(11,4) * 2^4 = 330 * 16 = 5280. ✓

    # Let's create a dictionary mapping (support, signs) -> index
    from collections import defaultdict

    support_to_vectors = defaultdict(list)
    vec_info = []  # (support_tuple, sign_tuple, index)

    idx = 0
    for comb in combinations(range(D), m):
        comb = tuple(comb)
        for mask in range(1 << m):
            signs = tuple(1 if (mask & (1 << j)) else -1 for j in range(m))
            support_to_vectors[comb].append((signs, idx))
            vec_info.append((comb, signs, idx))
            idx += 1

    # Now, conflicts:
    # Type (a): same support, same signs -> these are the SAME vector, so no conflict (they're the same index).
    #   Actually, each (support, signs) pair is unique, so there's only one vector per (support, signs).
    #   So type (a) conflicts don't exist between DISTINCT vectors.
    #   Wait, that means the only conflicts are type (b): share 3 support elements, signs match on those 3.

    # Let me verify: can two distinct vectors have the same support and same signs? No, because
    # the mapping is bijective. So type (a) doesn't create edges between distinct nodes.

    # So conflicts are ONLY: two vectors with supports sharing exactly 3 elements, and the signs
    # on those 3 shared positions are identical.

    # Let's count expected conflicts. For a given vector with support S (|S|=4),
    # how many other vectors share exactly 3 elements of S?
    # Choose 3 from S: C(4,3) = 4 ways.
    # For each such 3-element subset T ⊂ S, the other vector's support must contain T and one element
    # from (R^11 \ S), so 11-4 = 7 choices for the 4th element.
    # For each such support T ∪ {x}, the signs on T must match (1 way for the 3 shared signs),
    # and the sign on x can be ±1 (2 ways).
    # So for each vector: 4 * 7 * 2 = 56 conflicting vectors.
    # Total edges: 5280 * 56 / 2 = 147,840.

    # That's a sparse graph! 5280 nodes, ~148k edges. This is very tractable for exact MIS.

    # Let's build the conflict adjacency lists.
    # For each vector, find all vectors that share 3 support elements with matching signs.

    # For efficiency, let's index vectors by (3-element subset, signs on those 3).
    # For each vector with support (a,b,c,d) and signs (sa,sb,sc,sd):
    #   It conflicts with vectors that have support containing {a,b,c} with signs (sa,sb,sc) and a 4th element not in {a,b,c,d}.
    #   Similarly for {a,b,d}, {a,c,d}, {b,c,d}.

    # Build a dictionary: key = (frozenset of 3 indices, sign tuple on those indices) -> list of vector indices
    # But we need to be careful about ordering.

    # Let's use: key = (sorted_tuple_of_3, sign_tuple_in_that_order)
    triple_to_vecs = defaultdict(list)

    for (support, signs, vidx) in vec_info:
        # support is a 4-tuple, signs is a 4-tuple aligned with support
        # For each 3-subset of the 4 support elements:
        for omit in range(4):
            # Get the 3 indices and their signs
            triple_idx = tuple(support[i] for i in range(4) if i != omit)
            triple_signs = tuple(signs[i] for i in range(4) if i != omit)
            # triple_idx is already sorted since support is sorted and we remove one
            key = (triple_idx, triple_signs)
            triple_to_vecs[key].append(vidx)

    # Now, for each vector, its conflicts are all vectors in triple_to_vecs[key] for its 4 triple keys,
    # EXCLUDING itself.

    # But wait: if two vectors share the same triple (indices and signs), they conflict.
    # How many vectors share a given triple key?
    # A triple key is (3 specific indices, 3 specific signs).
    # Vectors with this triple as a 3-subset of their support: the 4th support element can be any of the
    # 11-3=8 other indices, and the 4th sign can be ±1. So 8*2=16 vectors per triple key.
    # But we also need the 4th element to NOT be in the triple (which is automatic since triple has 3 elements
    # and support has 4 distinct elements).
    # So each triple key has exactly 16 vectors.

    # For a given vector, it appears in 4 triple keys (one for each omitted element).
    # In each triple key, there are 16 vectors including itself, so 15 others.
    # But some of these 15 might be the same vector appearing in multiple triple keys?
    # No, because two distinct vectors can share at most one triple key (if they share 3 support elements).
    # If they shared 2 triple keys, they'd share 3+3=6 support elements, but supports have only 4 elements,
    # so they'd share all 4, meaning same support. But we established same support + same signs = same vector.
    # So the 4 triple keys give 4 * 15 = 60 distinct conflicting vectors? But we counted 56 earlier.
    # Let me recheck.

    # Earlier count: for each vector, 4 choices of which 3 elements to keep, 7 choices for the 4th element
    # (from the 7 elements not in the support), 2 choices for the sign of the 4th element. 4*7*2=56.
    # But in the triple_to_vecs approach, for each triple, the 4th element can be any of the 8 elements
    # not in the triple (11-3=8), including the 1 element that was omitted from the original support.
    # If the 4th element is the omitted one, then the support is the same as the original, and the signs
    # on the triple match, so the sign of the 4th element determines if it's the same vector or a different one.
    # If the 4th sign matches, it's the same vector (exclude). If different, it's a different vector with same support.
    # But same support with different 4th sign: do they conflict?
    # Support S, signs (sa,sb,sc,sd) vs (sa,sb,sc,sd'). IP = sa*sa+sb*sb+sc*sc+sd*sd' = 1+1+1+sd*sd' = 3+sd*sd'.
    # If sd*sd'=1 (same sign), IP=4 (conflict, but same vector). If sd*sd'=-1, IP=2 (compatible).
    # So same support, different 4th sign: IP=2, compatible. No conflict.
    # So in the triple key, the vector with same support and different 4th sign is NOT a conflict.
    # So for each triple key, the 16 vectors include:
    #   - The original vector (same support, same 4th sign) - exclude
    #   - 1 vector with same support, different 4th sign - compatible, not a conflict
    #   - 14 vectors with different 4th element (7 elements * 2 signs) - all conflicts
    # So 14 conflicts per triple key, 4 triple keys, total 56. ✓

    # So the conflict degree is 56 for each node. The graph is 56-regular (approximately).

    # Now, for exact MIS on a 5280-node 56-regular graph, we can use a branch-and-bound algorithm.
    # With a good coloring upper bound, this should be feasible.

    # Let's implement a standard MIS branch-and-bound:
    # 1. Order vertices by degree (or use a static ordering)
    # 2. Use a greedy coloring to get an upper bound on the remaining independent set size
    # 3. Branch: include or exclude the next vertex

    # Given the time limit, let's implement a practical version.

    # Build adjacency sets as Python sets for the conflict graph.
    adj = [set() for _ in range(N)]

    for (support, signs, vidx) in vec_info:
        for omit in range(4):
            triple_idx = tuple(support[i] for i in range(4) if i != omit)
            triple_signs = tuple(signs[i] for i in range(4) if i != omit)
            key = (triple_idx, triple_signs)
            for other_vidx in triple_to_vecs[key]:
                if other_vidx != vidx:
                    adj[vidx].add(other_vidx)
                    adj[other_vidx].add(vidx)  # Symmetric, but we'll add both to be safe

    # Actually, the above adds each edge twice (once from each endpoint). Let's just build it once.
    # Reset adj.
    adj = [set() for _ in range(N)]
    for key, vec_list in triple_to_vecs.items():
        # All vectors in vec_list conflict with each other EXCEPT those with same support.
        # Wait, no. Vectors in the same triple key all share the same 3 indices and 3 signs.
        # Two vectors in the same triple key: do they always conflict?
        # They share 3 support elements with matching signs. The 4th elements may differ.
        # If they have the same 4th element (same support), they differ only in the 4th sign.
        # As shown, same support with different 4th sign gives IP=2, compatible.
        # So within a triple key, vectors with the same support do NOT conflict.
        # Vectors with different supports DO conflict.

        # So for each triple key, group vectors by their full support.
        support_groups = defaultdict(list)
        for vidx in vec_list:
            support = vec_info[vidx][0]
            support_groups[support].append(vidx)

        # Within each support group, no conflicts (same support, different 4th signs -> compatible)
        # Between different support groups, all conflict.

        # So the conflict subgraph for this triple key is a complete multipartite graph
        # with parts = support groups.

        # For building adj, for each pair of vectors from different support groups, add an edge.
        # This is O(|vec_list|^2) per key, but |vec_list|=16, so 16*15/2=120 pairs per key.
        # Number of keys: 5280*4/16 = 1320 keys (each vector in 4 keys, each key has 16 vectors).
        # Total edge additions: 1320 * 120 = 158,400. But each edge is added once per shared triple key.
        # Each conflicting pair shares exactly one triple key (as argued), so each edge is added once.
        # Total edges: 158,400 / 2 = 79,200? No, we're adding directed edges here.
        # Let's just add undirected edges.

        all_vecs = vec_list
        for i in range(len(all_vecs)):
            for j in range(i+1, len(all_vecs)):
                vi, vj = all_vecs[i], all_vecs[j]
                if vec_info[vi][0] != vec_info[vj][0]:  # Different supports -> conflict
                    adj[vi].add(vj)
                    adj[vj].add(vi)

    # Now adj[i] is the set of vectors conflicting with i.
    # We want the maximum independent set in this conflict graph.

    # Use a branch-and-bound algorithm for MIS.
    # Order vertices by degree (highest first) for better pruning.
    # Actually, for MIS, we want to branch on vertices with high degree in the remaining graph.

    # Let's implement a standard recursive MIS with:
    # - P: set of available vertices
    # - X: set of vertices already excluded
    # - current_size: size of current independent set
    # - best_size: global best

    # Upper bound: |P| (trivial) or a coloring bound.

    # For a 5280-node graph, even with good pruning, this might be too slow.
    # Let's try a greedy + local search approach instead, but with a different local search:
    # **2-exchange local search**: remove 2 vertices, add back the best set of vertices that fit.

    # Actually, let's try a **genetic algorithm** or **simulated annealing with 2-swap moves**.

    # Given the time, let's implement a fast greedy + 1-swap local search with many restarts,
    # but optimized using the structure of the conflict graph.

    # Key optimization: for a given vector v, its non-neighbors (compatible vectors) are N-1-56 = 5223.
    # So the compatibility graph is very dense. Greedy should find a large set quickly.

    # Let's do:
    # 1. Greedy pack (random order)
    # 2. Local search: repeatedly try to add any compatible unselected vector
    # 3. If stuck, try 1-swap: remove a selected vector, add the best unselected vector that now fits
    # 4. Repeat with random restarts

    # But this is similar to the current ILS. Let me try the exact MIS with a time limit.

    # Implement MIS branch-and-bound:
    # Use a list-based representation for speed.

    # Convert adj to lists for faster iteration.
    adj_lists = [list(s) for s in adj]

    # Order vertices by degree (descending) for better branching.
    order = sorted(range(N), key=lambda i: -len(adj_lists[i]))

    # For the branch-and-bound, we'll work with the ordered list.
    # pos[i] = position of vertex i in the order.

    # This is getting complex. Let's just do a well-optimized greedy + local search.

    # **Strategy: Greedy with random order + 2-opt local search**
    # 2-opt: remove 2 selected vectors, then greedily add back as many as possible from the
    # unselected vectors that are now compatible.

    def _greedy_pack_fast(selected, available_mask):
        """Greedy add: repeatedly add any compatible unselected vector."""
        changed = True
        while changed:
            changed = False
            # Find all unselected vectors that are compatible with all selected
            # For each unselected i, check if any selected j is in adj[i]
            # This is O(N * K) where K is the number of selected.
            # Use the adj lists: for each unselected i, check if any neighbor is selected.
            for i in range(N):
                if selected[i]:
                    continue
                # Check if i is compatible with all selected
                compatible = True
                for j in adj_lists[i]:
                    if selected[j]:
                        compatible = False
                        break
                if compatible:
                    selected[i] = True
                    changed = True
        return selected

    def _local_search_2swap(selected):
        """Try 2-swap: remove 2 random selected, re-greedy."""
        selected_indices = [i for i in range(N) if selected[i]]
        if len(selected_indices) < 2:
            return selected
        # Remove 2 random
        i1, i2 = np.random.choice(selected_indices, size=2, replace=False)
        selected[i1] = False
        selected[i2] = False
        # Re-greedy
        _greedy_pack_fast(selected, None)
        return selected

    # Main loop: multiple restarts of greedy + local search
    best_score = 0
    best_selected = None

    # First, do one greedy to get a baseline
    selected = np.zeros(N, dtype=bool)
    order = np.random.permutation(N)
    for i in order:
        # Check if i is compatible with all currently selected
        compatible = True
        for j in adj_lists[i]:
            if selected[j]:
                compatible = False
                break
        if compatible:
            selected[i] = True

    # Local search: add any that fit
    _greedy_pack_fast(selected, None)
    score = np.sum(selected)
    best_score = score
    best_selected = selected.copy()

    # ILS with 2-swap
    while time.time() < deadline:
        current = best_selected.copy()
        # Perturb: 2-swap
        _local_search_2swap(current)
        # Local search: add any that fit
        _greedy_pack_fast(current, None)
        new_score = np.sum(current)
        if new_score > best_score:
            best_score = new_score
            best_selected = current.copy()

    if best_selected is not None and best_score > 0:
        return candidates[best_selected]
    else:
        return _generate_seed_vectors()

# EVOLVE-BLOCK-END
