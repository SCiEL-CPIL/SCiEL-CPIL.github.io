import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)  # 5280

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor sets as numpy boolean arrays
    # Also compute degree
    degrees = conflicts.sum(axis=1).astype(np.int64)

    # Build neighbor lists as sorted arrays for fast operations
    neighbor_lists = []
    for i in range(N4):
        neighbor_lists.append(np.where(conflicts[i])[0])

    # Tomita-style branch and bound for Maximum Independent Set
    # Upper bound: greedy clique cover on the subgraph induced by candidates
    # For independent set in G, the upper bound is the size of a minimum clique cover
    # of the subgraph, which equals the maximum independent set of the complement subgraph.
    # Simpler upper bound: |S| + max_degree_bound or use a greedy coloring of the complement.

    def greedy_clique_cover_upper_bound(candidate_set, time_limit):
        """
        Compute an upper bound on the independent set size in the subgraph induced by candidate_set.
        We use a greedy coloring of the complement graph: the chromatic number of the complement
        is an upper bound on the independence number of the original.
        
        Actually, for MIS in G: upper bound = clique cover number of G = chromatic number of complement of G.
        A greedy coloring gives an upper bound on chromatic number, hence an upper bound on clique cover.
        
        Simpler: use the fact that if we greedily color the complement, the number of colors is an upper bound.
        But that's expensive. Instead, use a simple bound: |S| itself, or a bound based on degree.
        
        Let's use a different approach: the upper bound for MIS in a graph with max degree Δ is at most n/(Δ+1) * something.
        Actually, a simple bound: for each vertex v, the independent set can contain v plus at most (n-1-|N(v)|) others.
        So upper bound = max over v in S of (1 + |S \\ (N(v) ∪ {v})|).
        """
        S = np.array(candidate_set, dtype=np.int64)
        if len(S) == 0:
            return 0
        if len(S) == 1:
            return 1

        # Simple upper bound: for each vertex, 1 + number of non-neighbors in S
        # This is O(n^2) per call which is too slow.
        # Use a simpler bound: the size of S itself (trivial), or a bound based on average degree.
        
        # Better: use the bound that in a graph where each vertex has at least d neighbors within S,
        # the independence number is at most |S| / (d+1) roughly. But this is loose.
        
        # For performance, let's use a greedy coloring of the complement as upper bound,
        # but only on a subset if S is large.
        
        # Actually, the standard Tomita bound uses a greedy coloring of the subgraph to get
        # an upper bound on the clique number, which equals the independence number of the complement.
        # For MIS in G, we want the independence number of G, which equals the clique number of complement(G).
        # So we can greedily color G to get an upper bound on the clique number of complement(G),
        # which is the independence number of G.
        
        # Greedy coloring of G: color vertices so that adjacent vertices have different colors.
        # The number of colors used is an upper bound on the clique number of the complement,
        # which equals the independence number of G.
        
        # Let's implement a simple greedy coloring.
        S_list = list(S)
        n_s = len(S_list)
        
        # Map global indices to local indices for the subgraph
        # We need to check conflicts within S
        # For speed, use the conflicts matrix directly
        
        color = [-1] * n_s
        # Order by degree within S (descending) for better coloring
        deg_in_s = np.zeros(n_s, dtype=np.int64)
        for i in range(n_s):
            vi = S_list[i]
            # Count neighbors in S
            nbrs = neighbor_lists[vi]
            # How many of nbrs are in S?
            # This is expensive. Let's just use global degree as a proxy.
            deg_in_s[i] = degrees[vi]
        
        order = np.argsort(-deg_in_s)
        
        num_colors = 0
        for idx in order:
            vi = S_list[idx]
            # Find the smallest color not used by neighbors of vi in S
            used_colors = set()
            for j in range(n_s):
                if j == idx or color[j] == -1:
                    continue
                vj = S_list[j]
                if conflicts[vi, vj]:
                    used_colors.add(color[j])
            c = 0
            while c in used_colors:
                c += 1
            color[idx] = c
            num_colors = max(num_colors, c + 1)
        
        return num_colors

    def tomita_mis(time_limit):
        """
        Tomita-style branch and bound for Maximum Independent Set.
        Returns the maximum independent set found.
        """
        best_set = []
        best_size = 0

        # We'll use a recursive approach with the following strategy:
        # - Order vertices by degree (descending)
        # - At each step, pick a vertex, either include it (and remove its neighbors) or exclude it
        # - Use upper bounds to prune
        
        # For efficiency, we'll use an iterative approach with a stack.
        
        # Initial candidate set: all vertices
        # We'll use a greedy initial solution first
        alive = np.ones(N4, dtype=bool)
        greedy_set = []
        deg_copy = degrees.copy()
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            # Pick vertex with minimum degree (to leave more options)
            min_deg_idx = alive_indices[np.argmin(deg_copy[alive_indices])]
            greedy_set.append(min_deg_idx)
            # Remove min_deg_idx and its neighbors
            to_remove = np.append([min_deg_idx], neighbor_lists[min_deg_idx][alive[neighbor_lists[min_deg_idx]]])
            alive[to_remove] = False
            # Update degrees
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                deg_copy[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        
        best_set = greedy_set
        best_size = len(greedy_set)

        # Now do branch and bound
        # We'll use a simple recursive B&B with a time limit
        # For efficiency, we'll limit the depth and use the greedy solution as initial bound
        
        # Actually, let's use a different approach: local search to improve the greedy solution.
        # Start with the greedy solution and try to improve it via swaps.
        
        # Convert to a set for fast lookup
        in_set = np.zeros(N4, dtype=bool)
        for v in greedy_set:
            in_set[v] = True

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to find a vertex not in the set that can be added by removing few vertices
            # For each vertex not in set, count how many of its neighbors are in the set
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            # For each candidate, compute the number of conflicts with the current set
            # and the potential gain
            for v in not_in_set:
                if time.time() > time_limit:
                    break
                # Count neighbors of v that are in the set
                nbrs_in_set = neighbor_lists[v][in_set[neighbor_lists[v]]]
                # If we add v, we must remove all nbrs_in_set
                # Gain = 1 - len(nbrs_in_set)
                # Only beneficial if len(nbrs_in_set) == 0 (can add directly)
                if len(nbrs_in_set) == 0:
                    # Add v
                    in_set[v] = True
                    greedy_set.append(v)
                    best_size += 1
                    improved = True
                    continue
                
                # Try a 1-for-2 swap: remove 2 vertices from set, add v
                # This is more complex. Let's try a simpler approach:
                # Try to replace one vertex in the set with v if v has fewer conflicts
                # Actually, let's try a different local search: for each pair of vertices in the set,
                # check if removing both and adding v (and possibly others) helps.
                
                # For now, let's just do the direct addition.
        
        # Try a more sophisticated local search: 2-opt style
        # For each vertex not in set, and each vertex in set, try swapping
        for v_not in np.where(~in_set)[0]:
            if time.time() > time_limit:
                break
            nbrs_in_set = neighbor_lists[v_not][in_set[neighbor_lists[v_not]]]
            if len(nbrs_in_set) == 0:
                in_set[v_not] = True
                greedy_set.append(v_not)
                best_size += 1
                continue
            # Try removing one neighbor and adding v_not
            for v_in in nbrs_in_set:
                # Check if v_in has any other neighbors in the set besides v_not
                # If we remove v_in, can we add v_not?
                # v_not's neighbors in set after removing v_in:
                remaining_nbrs = neighbor_lists[v_not][in_set[neighbor_lists[v_not]]]
                remaining_nbrs = remaining_nbrs[remaining_nbrs != v_in]
                if len(remaining_nbrs) == 0:
                    # Swap: remove v_in, add v_not
                    in_set[v_in] = False
                    in_set[v_not] = True
                    greedy_set[greedy_set.index(v_in)] = v_not
                    improved = True
                    break

        return greedy_set

    # Run the Tomita/local search
    best_selected = tomita_mis(deadline - 20)
    best_size = len(best_selected)

    # Try multiple random restarts of greedy MIS to find a better starting point
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        
        # Randomized greedy
        alive = np.ones(N4, dtype=bool)
        selected = []
        deg_copy = degrees.astype(np.float64).copy()
        
        while np.any(alive) and time.time() < deadline - 20:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            # Randomized: pick from the 3 lowest degree vertices
            alive_degrees = deg_copy[alive_indices]
            k = min(3, len(alive_indices))
            candidates = alive_indices[np.argsort(alive_degrees)[:k]]
            chosen = candidates[rng.integers(0, k)]
            selected.append(chosen)
            to_remove = np.append([chosen], neighbor_lists[chosen][alive[neighbor_lists[chosen]]])
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                deg_copy[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected

    # Polish: try to add any vertex that has no conflicts with the current set
    in_set = np.zeros(N4, dtype=bool)
    for v in best_selected:
        in_set[v] = True
    
    for v in range(N4):
        if time.time() > deadline - 10:
            break
        if in_set[v]:
            continue
        nbrs_in_set = neighbor_lists[v][in_set[neighbor_lists[v]]]
        if len(nbrs_in_set) == 0:
            in_set[v] = True
            best_selected.append(v)
            best_size += 1

    result_v4 = vectors_v4[best_selected]

    # V1 vectors: all 22 are mutually compatible
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
