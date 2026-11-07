<<<<<<< SEARCH
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import sys
sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    # Total edges
    m = len(edge_set)
    
    if m == 0:
        return []
    
    # Find all maximal bicliques using a heuristic approach
    # For each pair of vertices (u, v) that form an edge, try to extend to a maximal biclique
    # A biclique is defined by (L, R) where L and R are disjoint, and every u in L is connected to every v in R
    
    # Strategy: enumerate candidate bicliques by growing from seed edges
    # To limit the search, we use a greedy + local improvement approach combined with exact cover
    
    # First, let's find a good set of candidate bicliques
    # We'll use the following heuristic:
    # For each vertex u, consider its neighbors. Partition neighbors into groups that form bicliques.
    
    # Better approach: find all "maximal" bicliques via a limited enumeration
    # Given the graph is small (30 vertices), we can try to find good bicliques
    
    # Let's use a different strategy: 
    # 1. Find a greedy biclique cover
    # 2. Try to improve it by merging/optimizing
    
    # Greedy approach: repeatedly find the largest biclique, add it, remove its edges
    def find_largest_biclique(remaining_edges, adj_remaining):
        """Find a large biclique in the remaining graph using a heuristic."""
        if not remaining_edges:
            return None, None, set()
        
        # Try each edge as a seed
        best_size = 0
        best_biclique = None
        
        # Limit the number of seeds we try
        edge_list = list(remaining_edges)
        # Sort by degree product to try promising ones first
        for u, v in edge_list:
            # Start with biclique ({u}, {v})
            L = {u}
            R = {v}
            
            # Try to grow the biclique
            changed = True
            while changed:
                changed = False
                # Try adding a vertex to L
                for x in range(n):
                    if x not in L and x not in R:
                        # x can join L if connected to all in R
                        if all(x in adj_remaining[y] for y in R):
                            L.add(x)
                            changed = True
                # Try adding a vertex to R
                for x in range(n):
                    if x not in L and x not in R:
                        # x can join R if connected to all in L
                        if all(x in adj_remaining[y] for y in L):
                            R.add(x)
                            changed = True
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_biclique = (L, R)
        
        if best_biclique is None:
            return None, None, set()
        
        L, R = best_biclique
        # Get the edges in this biclique
        biclique_edges = set()
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e in remaining_edges:
                    biclique_edges.add(e)
        
        return L, R, biclique_edges
    
    # Greedy cover
    remaining = set(edge_set)
    adj_remaining = [set(adj[i]) for i in range(n)]
    cover = []
    
    while remaining:
        L, R, biclique_edges = find_largest_biclique(remaining, adj_remaining)
        if L is None or not biclique_edges:
            # Fallback: pick any edge
            u, v = next(iter(remaining))
            L = {u}
            R = {v}
            biclique_edges = {frozenset((u, v))}
        
        cover.append((L, R))
        for e in biclique_edges:
            remaining.discard(e)
            u, v = tuple(e)
            if u < v:
                adj_remaining[u].discard(v)
                adj_remaining[v].discard(u)
            else:
                adj_remaining[v].discard(u)
                adj_remaining[u].discard(v)
    
    num_bicliques = len(cover)
    
    # Now try to improve: can we merge some bicliques or find a better cover?
    # Let's try a local search: for each biclique, try to see if it can be replaced
    # by a larger one that covers more edges
    
    # Actually, let's try a different approach: use the greedy result as a starting point
    # and try to reduce the number of bicliques
    
    # For now, let's also try a simpler heuristic: 
    # Group edges by one endpoint and see if we can form bicliques that way
    
    # Let's try to find a better cover using a different heuristic
    # For each vertex, look at its neighbors and try to partition them
    
    # Alternative: use the fact that within a biclique, all left vertices share the same neighbors on the right
    
    # Let's try a smarter greedy: instead of largest biclique, try to maximize edge coverage per biclique
    # while keeping bicliques large
    
    # For now, return the greedy cover
    # Convert to the required format: list of [left_vertices, right_vertices]
    result = []
    for L, R in cover:
        left = sorted(list(L))
        right = sorted(list(R))
        # Ensure non-empty and disjoint
        if left and right and not set(left) & set(right):
            result.append([left, right])
        else:
            # This shouldn't happen, but handle it
            result.append([[min(L)], [max(R)]])
    
    return result
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import sys
sys.setrecursionlimit(10000)
import time

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    m = len(edge_set)
    if m == 0:
        return []
    
    # Step 1: Find all maximal bicliques using a Bron-Kerbosch variant
    # A biclique (L, R) is maximal if:
    # - No vertex can be added to L (i.e., no vertex outside L∪R is connected to all of R)
    # - No vertex can be added to R (i.e., no vertex outside L∪R is connected to all of L)
    # - L and R are non-empty and disjoint
    # - Every vertex in L is connected to every vertex in R
    
    # We'll enumerate maximal bicliques by considering each edge as a seed
    # and growing to maximality, then deduplicating
    
    all_bicliques = []  # list of (L_frozen, R_frozen, edge_mask)
    seen_bicliques = set()  # set of (frozenset(L), frozenset(R)) to avoid duplicates
    
    def is_valid_biclique(L, R):
        """Check if (L, R) forms a valid biclique."""
        if not L or not R:
            return False
        if L & R:
            return False
        for u in L:
            for v in R:
                if v not in adj[u]:
                    return False
        return True
    
    def grow_to_maximal(L, R):
        """Grow a biclique (L, R) to a maximal one."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Try to add vertices to L
            for x in range(n):
                if x not in L and x not in R:
                    if all(x in adj[y] for y in R):
                        L.add(x)
                        changed = True
            # Try to add vertices to R
            for x in range(n):
                if x not in L and x not in R:
                    if all(x in adj[y] for y in L):
                        R.add(x)
                        changed = True
        return L, R
    
    # Enumerate bicliques from each edge seed
    for u, v in edge_set:
        L, R = grow_to_maximal({u}, {v})
        # Also try the other orientation
        L2, R2 = grow_to_maximal({v}, {u})
        
        for Lc, Rc in [(L, R), (L2, R2)]:
            if Lc and Rc and not (Lc & Rc):
                key = (frozenset(Lc), frozenset(Rc))
                if key not in seen_bicliques:
                    seen_bicliques.add(key)
                    # Compute edge mask
                    edge_mask = 0
                    for a in Lc:
                        for b in Rc:
                            e = frozenset((a, b))
                            if e in edge_set:
                                edge_mask |= (1 << edge_idx[e])
                    all_bicliques.append((Lc, Rc, edge_mask))
    
    # Also try single-vertex seeds to catch bicliques where one side has size 1
    for u in range(n):
        for v in adj[u]:
            if u < v:
                continue
        # Try growing from ({u}, {v}) for each neighbor v
        for v in sorted(adj[u]):
            L, R = grow_to_maximal({u}, {v})
            key = (frozenset(L), frozenset(R))
            if key not in seen_bicliques:
                seen_bicliques.add(key)
                edge_mask = 0
                for a in L:
                    for b in R:
                        e = frozenset((a, b))
                        if e in edge_set:
                            edge_mask |= (1 << edge_idx[e])
                all_bicliques.append((L, R, edge_mask))
    
    # We need edge_idx mapping
    edge_list = list(edge_set)
    edge_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    
    # Recompute edge masks now that we have edge_idx
    biclique_masks = []
    for L, R, _ in all_bicliques:
        mask = 0
        for a in L:
            for b in R:
                e = frozenset((a, b))
                if e in edge_idx:
                    mask |= (1 << edge_idx[e])
        biclique_masks.append(mask)
    
    # Deduplicate bicliques with same edge mask (keep the one with larger |L|*|R|)
    best_by_mask = {}
    for i, (L, R, _) in enumerate(all_bicliques):
        mask = biclique_masks[i]
        size = len(L) * len(R)
        if mask not in best_by_mask or size > best_by_mask[mask][0]:
            best_by_mask[mask] = (size, L, R)
    
    # Final list of candidate bicliques
    candidates = []
    for mask, (size, L, R) in best_by_mask.items():
        candidates.append((L, R, mask))
    
    # Sort candidates by size descending for better pruning
    candidates.sort(key=lambda x: -(len(x[0]) * len(x[1])))
    
    # Step 2: Solve minimum set cover using recursive backtracking
    # Branch on the uncovered edge with the fewest candidate bicliques
    
    # Precompute for each edge, which bicliques cover it
    edge_to_bicliques = [[] for _ in range(m)]
    for i, (L, R, mask) in enumerate(candidates):
        for j in range(m):
            if mask & (1 << j):
                edge_to_bicliques[j].append(i)
    
    # Sort candidates by number of edges they cover (descending) for heuristic
    # But for backtracking, we'll use the edge with fewest candidates
    
    # Memoization: state is the set of uncovered edges (as a frozenset of edge indices)
    memo = {}
    
    def solve(uncovered_mask, current_bicliques):
        """Recursively find minimum biclique cover."""
        if uncovered_mask == 0:
            return current_bicliques
        
        # Memoization check
        key = uncovered_mask
        if key in memo:
            if memo[key] is None:
                return None
            return memo[key]
        
        # Find the uncovered edge with the fewest candidate bicliques
        min_count = float('inf')
        best_edge = -1
        for j in range(m):
            if uncovered_mask & (1 << j):
                # Count how many candidates cover this edge and don't conflict
                count = 0
                for i in edge_to_bicliques[j]:
                    L, R, mask = candidates[i]
                    # Check if this biclique only covers uncovered edges (or at least covers edge j)
                    if mask & (1 << j):
                        count += 1
                if count < min_count:
                    min_count = count
                    best_edge = j
                if min_count <= 1:
                    break
        
        if best_edge == -1:
            memo[key] = None
            return None
        
        if min_count == 0:
            memo[key] = None
            return None
        
        # Try each biclique that covers best_edge
        best_result = None
        for i in edge_to_bicliques[best_edge]:
            L, R, mask = candidates[i]
            # This biclique covers some set of edges
            # New uncovered = uncovered - mask
            new_uncovered = uncovered_mask & ~mask
            if new_uncovered == uncovered_mask:
                continue  # No progress
            
            # Recurse
            result = solve(new_uncovered, current_bicliques + [i])
            if result is not None:
                if best_result is None or len(result) < len(best_result):
                    best_result = result
                if len(best_result) == len(current_bicliques) + 1:
                    break  # Can't do better than 1 more biclique
        
        memo[key] = best_result
        return best_result
    
    # Initial uncovered mask: all edges
    initial_mask = (1 << m) - 1
    result_indices = solve(initial_mask, [])
    
    if result_indices is None:
        # Fallback: greedy cover
        remaining = set(edge_set)
        cover = []
        while remaining:
            # Find largest biclique in remaining
            best_size = 0
            best_biclique = None
            for u, v in remaining:
                L = {u}
                R = {v}
                changed = True
                while changed:
                    changed = False
                    for x in range(n):
                        if x not in L and x not in R:
                            if all(x in adj[y] for y in R):
                                L.add(x)
                                changed = True
                    for x in range(n):
                        if x not in L and x not in R:
                            if all(x in adj[y] for y in L):
                                R.add(x)
                                changed = True
                # Only keep edges that are in remaining
                valid_edges = set()
                for a in L:
                    for b in R:
                        e = frozenset((a, b))
                        if e in remaining:
                            valid_edges.add(e)
                if len(valid_edges) > best_size:
                    best_size = len(valid_edges)
                    best_biclique = (L, R, valid_edges)
            if best_biclique is None:
                u, v = next(iter(remaining))
                best_biclique = ({u}, {v}, {frozenset((u, v))})
            L, R, biclique_edges = best_biclique
            cover.append((L, R))
            for e in biclique_edges:
                remaining.discard(e)
        result_indices = list(range(len(cover)))
        # Build result from cover
        result = []
        for L, R in cover:
            left = sorted(list(L))
            right = sorted(list(R))
            if left and right and not set(left) & set(right):
                result.append([left, right])
            else:
                result.append([[min(L)], [max(R)]])
        return result
    
    # Build result from selected bicliques
    result = []
    for i in result_indices:
        L, R, mask = candidates[i]
        left = sorted(list(L))
        right = sorted(list(R))
        if left and right and not set(left) & set(right):
            result.append([left, right])
        else:
            result.append([[min(L)], [max(R)]])
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I have a bug - I'm using `edge_idx` before it's defined. Let me fix that.

