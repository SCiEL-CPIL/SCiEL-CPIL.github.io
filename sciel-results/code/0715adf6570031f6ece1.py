import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    D = 11

    # Strategy:
    # 1. Find a large set of 4-element supports (subsets of {0..10}) with pairwise intersection <= 2.
    #    This is the "core" packing problem. We solve it via greedy + local search on the support graph.
    # 2. For each selected support, we include ALL 16 sign patterns (2^4).
    #    Why is this valid?
    #    - All vectors have norm 2 (4 non-zero entries of magnitude 1).
    #    - Max norm = 2.
    #    - Condition: ||u - v|| >= 2  =>  ||u-v||^2 >= 4  =>  4 + 4 - 2(u.v) >= 4  =>  u.v <= 2.
    #    - For two vectors u, v with supports A, B:
    #        If A != B and |A ∩ B| <= 2, then |u.v| <= |A ∩ B| <= 2. So u.v <= 2 is always satisfied.
    #        If A == B (same support), u and v are different sign patterns.
    #        u.v = sum_{i in A} u_i v_i. Since u != v, they differ in at least one position.
    #        Actually, for distinct sign patterns on the same support:
    #        Let k be the number of positions where signs agree, and (4-k) where they differ.
    #        u.v = k - (4-k) = 2k - 4.
    #        Since u != v, k < 4, so k <= 3.
    #        Max u.v = 2(3) - 4 = 2.
    #        So u.v <= 2 is always satisfied even for same support.
    #    - Thus, if we pick a set of supports with pairwise intersection <= 2, we can include
    #      ALL 16 sign patterns for each support, and the configuration is valid.
    #    - Total vectors = 16 * (number of selected supports).
    #    - To beat the current score of 1.8909 (which is ~416 vectors if seed is 220, or just k),
    #      we need k > 220 * 1.8909 ≈ 416.
    #      Actually, the score is k. The seed is likely 220 (the simple 2-nonzero construction).
    #      Current program score 1.8909 implies k ≈ 416? No, the prompt says "score is k".
    #      "score is value/seed, so 1.0 equals the seed program".
    #      Seed program returns 220 vectors. So seed = 220.
    #      Current score 1.8909 => k = 1.8909 * 220 ≈ 416.
    #      We need k > 416.
    #      Since k = 16 * |S|, we need |S| > 416 / 16 = 26.
    #      We need to find a set of at least 27 supports with pairwise intersection <= 2.
    #      Total supports = C(11,4) = 330.
    #      Max size of such a set (packing number) is likely much larger than 27.
    #      Let's estimate: each support "blocks" supports sharing 3 elements.
    #      Number of supports sharing 3 elements with a fixed support:
    #      Fix 3 elements, choose 1 from remaining 7: 7 choices.
    #      So each support conflicts with 7 others (plus itself).
    #      Naive bound: 330 / 8 ≈ 41.
    #      So we can likely find a set of size ~30-40 easily.
    #      40 supports * 16 = 640 vectors. Score = 640/220 ≈ 2.9.
    #      This should beat the inspirations (2.5455) and the current program (1.8909).

    # Step 1: Generate all supports
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    # Convert to bitmasks
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)

    # Step 2: Build conflict graph
    # Conflict if intersection size is 3 (or 4, but supports are distinct so 4 is impossible for i!=j)
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)

    # Precompute adjacency as boolean matrix for fast vectorized operations
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in adj[i]:
            adj_bool[i, j] = True
    adj_bool = adj_bool | adj_bool.T  # Ensure symmetry

    # Step 3: Find Max Independent Set on the support graph
    # We use a greedy approach with local search (1-swap, 2-swap) to maximize the number of supports.

    def greedy_random(rng):
        """Randomized greedy independent set."""
        order = rng.permutation(N_S)
        selected = np.zeros(N_S, dtype=bool)
        for v in order:
            if not selected[v]:
                # Check if any neighbor is selected
                if not np.any(selected & adj_bool[v]):
                    selected[v] = True
        return selected

    def local_search(selected, rng, max_iter=500):
        """Local search with 1-swap and 2-swap improvements."""
        selected = selected.copy()
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < deadline - 5.0:
            improved = False
            it += 1
            # Compute blocked set
            in_set = selected
            if np.any(in_set):
                blocked = np.any(adj_bool[in_set], axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            candidates = np.where(~(in_set | blocked))[0]
            
            # Try 1-swap: remove one selected, add one candidate
            # We want to increase size, so we only do swaps if we can add more than we remove.
            # Actually, standard local search for MaxIS tries to add if possible, or swap if it leads to larger set.
            # Since we can only add if not blocked, if candidates is empty, we are stuck.
            # If candidates is not empty, we just add one.
            if len(candidates) > 0:
                # Add a random candidate
                v_add = int(candidates[rng.randint(len(candidates))])
                selected[v_add] = True
                improved = True
                continue
            
            # If no candidates, try 1-swap: remove one, see if new candidates appear
            selected_indices = np.where(selected)[0]
            if len(selected_indices) == 0:
                break
                
            # Try random swaps
            for _ in range(10):  # Try a few random swaps per iteration
                if time.time() > deadline - 5.0:
                    break
                v_remove = int(selected_indices[rng.randint(len(selected_indices))])
                # Temporarily remove
                selected[v_remove] = False
                
                # Recompute blocked
                if np.any(selected):
                    blocked2 = np.any(adj_bool[selected], axis=0)
                else:
                    blocked2 = np.zeros(N_S, dtype=bool)
                
                candidates2 = np.where(~(selected | blocked2))[0]
                
                if len(candidates2) > 0:
                    # Can we add one?
                    v_add = int(candidates2[rng.randint(len(candidates2))])
                    selected[v_add] = True
                    improved = True
                    break
                else:
                    # Restore
                    selected[v_remove] = True

        return selected

    # Run multiple restarts
    rng = np.random.RandomState(42)
    best_selected = np.zeros(N_S, dtype=bool)
    best_size = 0

    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Greedy initialization
        init_selected = greedy_random(rng)
        
        # Local search
        final_selected = local_search(init_selected, rng, max_iter=200)
        
        size = np.sum(final_selected)
        if size > best_size:
            best_size = size
            best_selected = final_selected.copy()

    # Step 4: Construct vectors
    vectors = []
    selected_support_indices = np.where(best_selected)[0]
    
    for s_idx in selected_support_indices:
        support = supports[s_idx]
        # Include all 16 sign patterns
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    if not vectors:
        # Fallback: simple 2-nonzero construction
        vectors = []
        for support in itertools.combinations(range(D), 2):
            i, j = support
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)

    final_vectors = np.array(vectors, dtype=np.int8)
    
    # Optional: Verify validity (for debugging, remove if too slow)
    # if len(final_vectors) > 0:
    #     V = final_vectors.astype(float)
    #     norms = np.linalg.norm(V, axis=1)
    #     max_norm = np.max(norms)
    #     # Check pairwise distances
    #     # This is O(N^2), might be slow for large N.
    #     # For now, trust the construction logic.
    
    return final_vectors
