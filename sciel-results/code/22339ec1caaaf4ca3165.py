import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges_list = P['edges']
    edge_set = set()
    for e in edges_list:
        a, b = e
        if a > b:
            a, b = b, a
        edge_set.add((a, b))

    # Build adjacency as sets
    adj = [set() for _ in range(n)]
    for a, b in edge_set:
        adj[a].add(b)
        adj[b].add(a)

    start_time = time.time()
    deadline = start_time + 80.0

    # Step 1: Enumerate all maximal bicliques
    # A biclique is (L, R) where every l in L connects to every r in R
    # Maximal means no vertex can be added to L or R while maintaining the property
    
    all_bicliques = []
    
    # For each pair of vertices (u,v) with an edge, try to expand to a maximal biclique
    # Key: for a biclique (L,R), L must be a clique in the "neighborhood" sense
    # Actually, in a complete bipartite subgraph, L and R are independent sets in the original graph? No.
    # L can have edges between its members, R can have edges between its members.
    # The constraint is: every l in L is adjacent to every r in R.
    
    # Strategy: for each edge (u,v), generate the maximal biclique containing it
    # by iteratively adding vertices that connect to all of the opposite side
    
    def expand_biclique(L, R):
        """Expand (L, R) to a maximal biclique."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Try adding to L
            for x in range(n):
                if x in L or x in R:
                    continue
                # x must be adjacent to all in R
                if all(x in adj[y] for y in R):
                    L.add(x)
                    changed = True
            # Try adding to R
            for y in range(n):
                if y in L or y in R:
                    continue
                # y must be adjacent to all in L
                if all(y in adj[x] for x in L):
                    R.add(y)
                    changed = True
        return frozenset(L), frozenset(R)
    
    seen_bicliques = set()
    for (u, v) in edge_set:
        L, R = expand_biclique({u}, {v})
        key = (L, R)
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            all_bicliques.append((L, R))
    
    # Also try starting from single vertices to catch bicliques where one side is size 1
    # Actually the edge-based approach should catch all since every biclique has at least one edge
    # But let's also try: for each vertex, the biclique where L={v} and R=N(v)
    for v in range(n):
        if adj[v]:
            L, R = expand_biclique({v}, set(adj[v]))
            key = (L, R)
            if key not in seen_bicliques:
                seen_bicliques.add(key)
                all_bicliques.append((L, R))
    
    # Precompute edge sets for each biclique
    biclique_edges = []
    for L, R in all_bicliques:
        edges = set()
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                edges.add(e)
        biclique_edges.append(edges)
    
    # Step 2: Solve minimum set cover via branch and bound
    # We need to cover all edges in edge_set using bicliques from all_bicliques
    # such that the edge sets are disjoint (partition)
    
    # Precompute: for each biclique, which edges it covers
    # And check that biclique edges are subset of edge_set (they should be by construction)
    
    # Filter: only keep bicliques whose edges are all in edge_set
    valid_biclique_indices = []
    valid_biclique_edges = []
    for i, edges in enumerate(biclique_edges):
        if edges.issubset(edge_set):
            valid_biclique_indices.append(i)
            valid_biclique_edges.append(edges)
    
    all_bicliques = [all_bicliques[i] for i in valid_biclique_indices]
    biclique_edges = valid_biclique_edges
    
    # For efficiency, represent each biclique's edge set as a frozenset
    # and precompute which edges each covers
    
    # Branch and bound for minimum set cover with disjointness constraint
    # This is actually a partition problem, which is harder than set cover
    
    # Let's use a different approach: greedy + local search with better initialization
    # from the enumerated bicliques
    
    # Greedy: repeatedly pick the biclique that covers the most uncovered edges
    # But we need disjointness, so we can only pick bicliques that don't overlap with already chosen ones
    
    # Actually, let me think about this differently.
    # The problem is: partition edge_set into biclique edge sets.
    # Each biclique edge set must be one of our enumerated bicliques (or a subset).
    
    # Since we want minimum number of bicliques, we want to use as few as possible.
    # Let's try: sort bicliques by size (largest first), then greedily pick non-overlapping ones.
    
    # Better: use the enumerated bicliques as candidates, and do branch-and-bound
    # where we try to cover all edges with minimum bicliques, ensuring disjointness.
    
    # For the branch-and-bound:
    # State: set of uncovered edges, set of bicliques already used
    # Branch: pick a biclique that covers at least one uncovered edge and doesn't overlap with used bicliques
    # Bound: lower bound on additional bicliques needed = ceil(uncovered_edges / max_biclique_size)
    
    # But this might be too slow. Let's try a hybrid:
    # 1. Greedy to get an initial solution
    # 2. Local search to improve
    
    # Greedy initialization using enumerated bicliques
    def greedy_init():
        remaining = set(edge_set)
        chosen = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_bic = None
            best_count = 0
            
            for i, edges in enumerate(biclique_edges):
                # Count how many of these edges are in remaining
                overlap = len(edges & remaining)
                if overlap > best_count:
                    # Check if this biclique can be added (disjoint from already chosen)
                    # Actually in greedy, we just pick the one covering most remaining edges
                    # The disjointness is automatically maintained since we remove covered edges
                    best_count = overlap
                    best_bic = i
            
            if best_bic is None or best_count == 0:
                # Fallback: single edge
                e = next(iter(remaining))
                chosen.append(((e[0],), (e[1],)))
                remaining.discard(e)
            else:
                L, R = all_bicliques[best_bic]
                chosen.append((L, R))
                for e in biclique_edges[best_bic]:
                    remaining.discard(e)
        
        return chosen
    
    # Local search: try to merge pairs of bicliques
    def local_search(bicliques):
        # Convert to mutable lists
        bicliques = [list(b) for b in bicliques]  # Each is [L, R] where L, R are tuples
        
        improved = True
        while improved:
            improved = False
            n_b = len(bicliques)
            for i in range(n_b):
                for j in range(i + 1, n_b):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    L1s = set(L1)
                    R1s = set(R1)
                    L2s = set(L2)
                    R2s = set(R2)
                    
                    # Try all 4 combinations of merging
                    # (L1∪L2, R1), (L1, R1∪R2), (L1∪L2, R1∪R2), etc.
                    # Actually, we can merge if the union forms a valid biclique
                    # and covers exactly the union of edges
                    
                    candidates = [
                        (L1s | L2s, R1s),
                        (L1s, R1s | R2s),
                        (L1s | L2s, R1s | R2s),
                        (L1s, R2s),
                        (L2s, R1s),
                    ]
                    
                    for L_new, R_new in candidates:
                        if not L_new or not R_new:
                            continue
                        if L_new & R_new:
                            continue
                        
                        # Check if L_new × R_new is a valid biclique in the graph
                        valid = True
                        for l in L_new:
                            for r in R_new:
                                if r not in adj[l]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if not valid:
                            continue
                        
                        # Check that the edge set of (L_new, R_new) equals E1 ∪ E2
                        E1 = set()
                        for l in L1s:
                            for r in R1s:
                                E1.add((min(l, r), max(l, r)))
                        
                        E2 = set()
                        for l in L2s:
                            for r in R2s:
                                E2.add((min(l, r), max(l, r)))
                        
                        E_target = E1 | E2
                        
                        E_new = set()
                        for l in L_new:
                            for r in R_new:
                                E_new.add((min(l, r), max(l, r)))
                        
                        if E_new == E_target:
                            # Merge successful
                            bicliques[i] = (tuple(L_new), tuple(R_new))
                            bicliques.pop(j)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return bicliques
    
    # Run greedy initialization multiple times with different orderings
    best_result = None
    best_count = float('inf')
    
    # Sort bicliques by size for greedy
    biclique_sizes = [len(e) for e in biclique_edges]
    sorted_indices = sorted(range(len(biclique_edges)), key=lambda i: biclique_sizes[i], reverse=True)
    
    # Try greedy with different random perturbations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Shuffle the order slightly to get different greedy choices
        random.seed(trial)
        shuffled = sorted_indices[:]
        # Perturb: swap some elements
        for _ in range(5):
            i, j = random.randint(0, len(shuffled)-1), random.randint(0, len(shuffled)-1)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        
        remaining = set(edge_set)
        chosen = []
        
        for idx in shuffled:
            if not remaining:
                break
            edges = biclique_edges[idx]
            overlap = edges & remaining
            if overlap:
                # Check disjointness with already chosen
                can_add = True
                for prev_L, prev_R in chosen:
                    prev_edges = set()
                    for l in prev_L:
                        for r in prev_R:
                            prev_edges.add((min(l, r), max(l, r)))
                    if overlap & prev_edges:
                        can_add = False
                        break
                if can_add:
                    L, R = all_bicliques[idx]
                    chosen.append((L, R))
                    remaining -= overlap
        
        # Fill in remaining edges as single-edge bicliques
        for e in remaining:
            chosen.append(((e[0],), (e[1],)))
        
        # Local search
        result = local_search(chosen)
        if len(result) < best_count:
            best_count = len(result)
            best_result = result
    
    # If we still have time, try more trials
    if best_result is None:
        # Fallback: star decomposition
        result = []
        for u in range(n):
            neighbors = [v for v in adj[u] if v > u]
            if neighbors:
                result.append(([int(u)], [int(v) for v in neighbors]))
        best_result = result
        best_count = len(result)
    
    # Convert to required format
    final_result = []
    for L, R in best_result:
        left = [int(x) for x in L]
        right = [int(x) for x in R]
        final_result.append([left, right])
    
    return final_result
