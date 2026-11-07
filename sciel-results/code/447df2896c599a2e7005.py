import time
import sys
from collections import defaultdict
from itertools import combinations
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # 1. Preprocessing
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
    
    m = len(edge_set)
    edge_indices = {e: i for i, e in enumerate(edge_set)}
    edges_list = list(edge_set)

    # 2. Find all maximal bicliques using Bron-Kerbosch on the complement graph
    # A biclique (L, R) in G is a clique in the "bipartite complement" where
    # vertices in L are independent in G, vertices in R are independent in G,
    # and every u in L is connected to every v in R in G.
    
    # We use a standard Bron-Kerbosch variant to find all maximal bicliques.
    # To avoid duplicates (L,R) and (R,L), we enforce min(L) < min(R) or similar,
    # but simpler: just collect all and deduplicate later, or use a specific ordering.
    # Given n=30, the number of maximal bicliques is likely manageable.

    # We will run Bron-Kerbosch on the graph H where an edge (u, v) exists in H
    # if u and v are NOT connected in G (for the same side) OR if they ARE connected in G (for cross side).
    # This is tricky to map directly to a single graph.
    
    # Alternative: Enumerate all subsets of vertices? No, 2^30 is too big.
    # Alternative: Use the property that a maximal biclique is determined by its "core" edges.
    
    # Let's use a heuristic enumeration: 
    # For each edge (u, v), try to expand it to a maximal biclique.
    # This might miss some, but combined with multiple seeds or a smarter expansion, it's good.
    # However, to be "materially different" and robust, let's try to find *all* maximal bicliques
    # using a recursive search on the edge set.
    
    # Actually, a simpler and very effective approach for small graphs:
    # Generate a set of candidate bicliques by:
    # 1. Taking each edge (u, v).
    # 2. Expanding L and R greedily to be maximal with respect to the current state.
    # 3. Deduplicating.
    # Then solve Set Cover on these candidates.
    
    # To ensure we don't miss the optimal solution, we need a sufficiently rich set of candidates.
    # The current program's weakness was the *partition* constraint during greedy.
    # Here, we generate candidates independently, then select a partition.
    
    candidates = set()
    
    def add_candidate(L, R):
        # Normalize: sort L and R, and ensure a canonical representation
        # e.g., min(L) < min(R) if possible, or just sort the tuple of frozensets
        # But we need to preserve which is Left and which is Right for the final output?
        # The problem says "each [left_vertices, right_vertices]". It doesn't specify an order between L and R,
        # but the edge set is undirected. So (L, R) and (R, L) represent the same biclique.
        # We will store them as (frozenset(L), frozenset(R)) and handle symmetry in set cover.
        
        # To avoid duplicates, let's impose an order: 
        # If min(L) < min(R), keep (L, R). Else if min(L) > min(R), swap to (R, L).
        # If min(L) == min(R) (impossible since disjoint), handle.
        # Actually, just store both (L, R) and (R, L) as the same object in the candidate pool?
        # No, let's just normalize: 
        # Sort L, Sort R.
        # If min(L) < min(R): key = (frozenset(L), frozenset(R))
        # Else: key = (frozenset(R), frozenset(L))
        
        L_list = sorted(list(L))
        R_list = sorted(list(R))
        if not L_list or not R_list:
            return
        if L_list[0] < R_list[0]:
            key = (frozenset(L_list), frozenset(R_list))
        else:
            key = (frozenset(R_list), frozenset(L_list))
        candidates.add(key)

    # Generate candidates by expanding each edge
    for u, v in edges:
        L = {u}
        R = {v}
        # Greedy expansion to maximal
        changed = True
        while changed:
            changed = False
            # Try adding to L
            for x in range(n):
                if x in L:
                    continue
                # x must be connected to all in R
                if all(adj[x][y] for y in R):
                    L.add(x)
                    changed = True
            # Try adding to R
            for y in range(n):
                if y in R:
                    continue
                # y must be connected to all in L
                if all(adj[x][y] for x in L):
                    R.add(y)
                    changed = True
        add_candidate(L, R)
        
        # Also try swapping initial roles
        L = {v}
        R = {u}
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x in L:
                    continue
                if all(adj[x][y] for y in R):
                    L.add(x)
                    changed = True
            for y in range(n):
                if y in R:
                    continue
                if all(adj[x][y] for x in L):
                    R.add(y)
                    changed = True
        add_candidate(L, R)

    # To increase diversity, we can also try expanding from pairs of edges that share a vertex, etc.
    # But let's first see if the single-edge expansion gives enough candidates.
    # If not, we can add a randomization step.
    
    # Let's add a few random restarts of the expansion process with different tie-breaking
    import random
    random.seed(42)
    
    for _ in range(10):
        for u, v in edges:
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                # Random order for adding to L
                candidates_L = [x for x in range(n) if x not in L and all(adj[x][y] for y in R)]
                if candidates_L:
                    x = random.choice(candidates_L)
                    L.add(x)
                    changed = True
                candidates_R = [y for y in range(n) if y not in R and all(adj[x][y] for x in L)]
                if candidates_R:
                    y = random.choice(candidates_R)
                    R.add(y)
                    changed = True
            add_candidate(L, R)

    # Convert candidates to list for Set Cover
    # Each candidate is a set of edges
    cand_edge_sets = []
    cand_vertices = [] # (L, R)
    for L, R in candidates:
        edge_mask = 0
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in edge_indices:
                    idx = edge_indices[e]
                    edge_mask |= (1 << idx)
        if edge_mask != 0:
            cand_edge_sets.append(edge_mask)
            cand_vertices.append((list(L), list(R)))

    if not cand_edge_sets:
        # Fallback
        return [[[u], [v]] for u, v in edges]

    # 3. Solve Set Cover (Partition) using Branch and Bound
    # We need to select a subset of candidates such that their edge sets are disjoint and cover all edges.
    
    all_edges_mask = (1 << m) - 1
    best_count = float('inf')
    best_selection = []
    
    # Sort candidates by size (larger first) to help pruning
    # We also need to handle the "disjoint" constraint.
    
    # Precompute compatibility? Or just check during search.
    # Given m ~ 150, bitmasks are efficient.
    
    # Algorithm X with dancing links is great for exact cover, but we have a "partition" constraint
    # which is exactly exact cover if we treat each edge as a column and each biclique as a row
    # that covers specific columns. However, standard Exact Cover requires *every* column to be covered
    # *exactly once*. This is exactly our problem!
    
    # So we can use Algorithm X / Dancing Links or a simple recursive backtracking with bitmasks.
    # Since m=150, we can't use a single integer for the mask if we use Python ints (they are arbitrary precision, so we CAN).
    
    def solve_exact_cover():
        nonlocal best_count, best_selection
        
        # We want to find the minimum number of sets from cand_edge_sets that partition all_edges_mask.
        
        # Heuristic: pick the edge that appears in the fewest candidates (most constrained)
        # Count occurrences of each edge in candidates
        edge_counts = [0] * m
        for mask in cand_edge_sets:
            for i in range(m):
                if mask & (1 << i):
                    edge_counts[i] += 1
        
        # If any edge count is 0, no solution (should not happen if candidates cover all edges)
        
        def backtrack(covered_mask, selected_count, current_selection):
            nonlocal best_count, best_selection
            
            if time.time() > deadline:
                return
                
            if selected_count >= best_count:
                return
                
            if covered_mask == all_edges_mask:
                if selected_count < best_count:
                    best_count = selected_count
                    best_selection = current_selection[:]
                return
                
            # Find the uncovered edge with the fewest candidate options
            # This is the "minimum remaining values" heuristic
            best_edge = -1
            min_options = float('inf')
            remaining_mask = all_edges_mask ^ covered_mask
            
            # Iterate through uncovered edges
            # To speed up, we can just pick the first one or the most constrained one.
            # Let's pick the most constrained one among the uncovered.
            for i in range(m):
                if remaining_mask & (1 << i):
                    # Count how many candidates cover this edge AND are compatible with current selection
                    # Compatible means: candidate does not overlap with covered_mask
                    # Since we are building a partition, if a candidate overlaps with covered_mask, it's invalid.
                    # So we only consider candidates where (cand_mask & covered_mask) == 0.
                    # And it must cover edge i.
                    
                    options = 0
                    for mask in cand_edge_sets:
                        if mask & (1 << i):
                            if (mask & covered_mask) == 0:
                                options += 1
                    if options < min_options:
                        min_options = options
                        best_edge = i
                        if options == 0:
                            return # Dead end
                    # Early stop if we find an edge with 1 option? No, we need the min.
                    # But if min_options becomes 1, we can't get lower.
                    if min_options == 1:
                        break
                        
            if best_edge == -1:
                return
                
            # Try all candidates that cover best_edge and are compatible
            for idx, mask in enumerate(cand_edge_sets):
                if mask & (1 << best_edge):
                    if (mask & covered_mask) == 0:
                        # Select this candidate
                        backtrack(covered_mask | mask, selected_count + 1, current_selection + [idx])
                        
            # Note: We don't explicitly "skip" best_edge because if we don't cover it now, 
            # we must cover it later. But in Exact Cover, every element must be covered by exactly one row.
            # So we MUST pick a row that covers best_edge.
            
        backtrack(0, 0, [])
        
    solve_exact_cover()
    
    if best_selection:
        result = []
        for idx in best_selection:
            L, R = cand_vertices[idx]
            result.append([list(L), list(R)])
        return result
    else:
        # Fallback to greedy partition if exact cover fails or times out
        # Re-use the greedy logic from the original but just to have a valid output
        rem = set(edges)
        bics = []
        for u, v in edges:
            if (u, v) not in rem and (v, u) not in rem:
                continue
            # Simple greedy
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in L: continue
                    if all(adj[x][y] for y in R):
                        # Check if edges are in rem
                        if all((min(x,y), max(x,y)) in rem or (max(x,y), min(x,y)) in rem for y in R):
                            L.add(x)
                            changed = True
                for y in range(n):
                    if y in R: continue
                    if all(adj[x][y] for x in L):
                         if all((min(x,y), max(x,y)) in rem or (max(x,y), min(x,y)) in rem for x in L):
                            R.add(y)
                            changed = True
            b_edges = set((min(u, v), max(u, v)) for u in L for v in R)
            bics.append([list(L), list(R)])
            rem -= b_edges
        return bics

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")

