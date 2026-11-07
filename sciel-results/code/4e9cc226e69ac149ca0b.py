"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import time
import sys
sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build edge bitmask representation
    # We need to cover all edges in P['edges']
    m = len(edges)
    if m == 0:
        return []
        
    # Map edge index to bitmask
    # Since n=30, we can use Python integers as bitmasks for vertices
    # But we need to track which *edges* are covered.
    # Let's represent the set of uncovered edges as a frozenset of edge indices or a bitmask of edge indices.
    # m is approx 150-200 edges. Python int bitmask works for edges too.
    
    # Precompute adjacency bitmasks for vertices
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        
    # Map each edge to an index for bitmasking the edge set
    edge_index = {}
    edge_list = []
    for i, (a, b) in enumerate(edges):
        if a > b:
            a, b = b, a
        edge_list.append((a, b))
        edge_index[(a, b)] = i
        
    # Full mask of all edges
    full_mask = (1 << m) - 1
    
    # Precompute for each vertex pair (u, v) the bitmask of edges incident to u that connect to v?
    # No, we need to find bicliques.
    # A biclique is defined by L and R.
    # The edges covered are all (u, v) where u in L, v in R.
    # We need to efficiently check if a candidate biclique is valid (all its edges are in the current uncovered set)
    # and what edges it covers.
    
    # Let's use a recursive search for the minimum number of bicliques.
    # State: current uncovered edge mask.
    # We try to cover it with k bicliques.
    
    # Optimization: Precompute for every possible pair (u, v) the bitmask of edges in the biclique {u} x {v}? That's just 1 edge.
    # Better: For a set of vertices L, the edges covered by L x R are the intersection of adj_mask[u] for u in L, restricted to R.
    
    # Let's implement a helper to find maximal bicliques in a given edge mask.
    # Actually, finding *any* valid biclique that covers a subset of edges is key.
    # We can iterate over all pairs (u, v) in the remaining edges, and try to expand them into a maximal biclique.
    
    def get_maximal_bicliques(uncovered_mask):
        """
        Yields maximal bicliques (L, R) such that all edges in LxR are in uncovered_mask.
        We only need to consider bicliques that cover at least one edge.
        """
        # Get list of edges in uncovered_mask
        # This is slow if done naively.
        # Instead, iterate over vertices u, v.
        # If edge (u, v) is in uncovered_mask, we can try to form a biclique containing u in L and v in R.
        
        # To avoid duplicates, we enforce an ordering or just collect all and deduplicate later?
        # For k=2,3, the number of candidates might be high.
        # Let's just find *all* maximal bicliques.
        
        bicliques = []
        seen = set()
        
        # Iterate over all pairs (u, v)
        for u in range(n):
            for v in range(u + 1, n):
                if (1 << edge_index.get((min(u,v), max(u,v)), -1)) & uncovered_mask:
                    # Edge exists and is uncovered
                    # Expand to maximal biclique
                    # L must contain u, R must contain v.
                    # w can join L if connected to all in R.
                    # w can join R if connected to all in L.
                    
                    L = {u}
                    R = {v}
                    changed = True
                    while changed:
                        changed = False
                        # Try adding to L
                        for w in range(n):
                            if w not in L and w not in R:
                                # Check if w is connected to all in R via uncovered edges
                                valid = True
                                for x in R:
                                    idx = edge_index.get((min(w,x), max(w,x)), -1)
                                    if idx == -1 or not (uncovered_mask & (1 << idx)):
                                        valid = False
                                        break
                                if valid:
                                    L.add(w)
                                    changed = True
                        # Try adding to R
                        for w in range(n):
                            if w not in L and w not in R:
                                valid = True
                                for x in L:
                                    idx = edge_index.get((min(w,x), max(w,x)), -1)
                                    if idx == -1 or not (uncovered_mask & (1 << idx)):
                                        valid = False
                                        break
                                if valid:
                                    R.add(w)
                                    changed = True
                                    
                    # Canonicalize L and R to avoid duplicates
                    # Note: The biclique L x R is the same as R x L.
                    # Also, different expansion paths might lead to the same set.
                    key = (frozenset(L), frozenset(R))
                    key2 = (frozenset(R), frozenset(L))
                    if key not in seen and key2 not in seen:
                        seen.add(key)
                        seen.add(key2)
                        bicliques.append((frozenset(L), frozenset(R)))
                        
        return bicliques

    # Recursive search
    # To speed up, we can use a heuristic: pick the edge with the most constraints?
    # Or just pick the first available edge and try to extend it?
    # The "get_maximal_bicliques" above is O(n^3 * 2^n) worst case? No, the expansion is polynomial.
    # But the number of maximal bicliques can be exponential.
    # However, for k=1,2,3, we only need to find *one* valid sequence.
    
    # Strategy:
    # 1. Try k=1. Find any biclique that covers all edges. (Just check if graph is bipartite? No, biclique partition allows L,R to be any sets. k=1 means the graph itself is a biclique).
    # 2. Try k=2. Pick a maximal biclique B1. Check if remaining edges form a biclique B2.
    # 3. Try k=3. Pick B1. Pick B2 from remaining. Check if B3 covers rest.
    
    # To make this efficient, we should order the bicliques.
    # We can iterate through the list of maximal bicliques of the *initial* graph?
    # No, the maximal bicliques of the *remaining* graph change.
    
    # Let's refine the search:
    # def search(uncovered_mask, depth, max_depth):
    #   if uncovered_mask == 0: return True
    #   if depth == max_depth: return False
    #   
    #   # Find a candidate edge to cover
    #   # Pick an edge e in uncovered_mask.
    #   # Find all maximal bicliques in uncovered_mask that contain e.
    #   # For each such biclique B:
    #      new_mask = uncovered_mask & ~mask(B)
    #      if search(new_mask, depth+1, max_depth): return True
    #   return False
    
    # To avoid recomputing maximal bicliques from scratch every time (which is expensive),
    # we can cache them or use a smarter generation.
    # Given n=30 and time limit 60s, and k likely small, this should work.
    
    # Optimization: Sort vertices by degree to find maximal bicliques faster?
    
    def find_bicliques_containing_edge(uncovered_mask, u, v):
        """Find maximal bicliques in uncovered_mask that include edge (u,v)."""
        # We know u in L, v in R.
        # We expand this specific seed.
        # There might be multiple maximal bicliques containing (u,v) if the expansion choices branch?
        # No, the "maximal" biclique containing a specific seed (u,v) with respect to the edge set is unique?
        # Actually, no. Consider a diamond graph. 
        # But for a fixed edge set, the set of vertices that *can* be in L is determined by common neighbors of R.
        # The *maximal* biclique containing (u,v) is unique?
        # Let L_max be the set of all vertices that can be in L.
        # Let R_max be the set of all vertices that can be in R.
        # The intersection L_max x R_max is the unique maximal biclique containing (u,v).
        
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w not in L and w not in R:
                    valid = True
                    for x in R:
                        idx = edge_index.get((min(w,x), max(w,x)), -1)
                        if idx == -1 or not (uncovered_mask & (1 << idx)):
                            valid = False
                            break
                    if valid:
                        L.add(w)
                        changed = True
            for w in range(n):
                if w not in L and w not in R:
                    valid = True
                    for x in L:
                        idx = edge_index.get((min(w,x), max(w,x)), -1)
                        if idx == -1 or not (uncovered_mask & (1 << idx)):
                            valid = False
                            break
                    if valid:
                        R.add(w)
                        changed = True
                        
        return frozenset(L), frozenset(R)
        
    # Precompute edge masks for bicliques? No, bicliques are dynamic.
    
    # We need to convert a biclique (L, R) to an edge mask.
    def biclique_to_mask(L, R):
        mask = 0
        for u in L:
            for v in R:
                idx = edge_index.get((min(u,v), max(u,v)), -1)
                if idx != -1:
                    mask |= (1 << idx)
        return mask

    # Search function
    # We need to return the actual bicliques, not just True/False.
    
    def solve(uncovered_mask, max_bicliques):
        # Returns list of bicliques or None
        if uncovered_mask == 0:
            return []
        if max_bicliques == 0:
            return None
            
        # Pick an edge to cover.
        # Heuristic: Pick the edge with the fewest candidate bicliques?
        # Or just the first one.
        # Let's pick the edge (u, v) that is "most constrained" or just the first available.
        # To ensure determinism, we pick the first available edge index.
        
        # Find first set bit
        # e_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        # This is efficient.
        
        e_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        u, v = edge_list[e_idx]
        
        # Find the maximal biclique containing (u, v) in the current uncovered graph.
        # As discussed, the maximal biclique containing a specific edge is unique.
        # Is it always optimal to take the *maximal* one?
        # Not necessarily. A smaller biclique might allow a better fit for the remaining edges.
        # However, if we take a subset, we leave more edges for later, which is harder.
        # Usually, taking the maximal is a good heuristic.
        # But for exactness, we might need to try subsets.
        # Given the time limit, let's try the maximal one first. If it fails, we might need backtracking on subsets.
        # For now, let's assume the maximal biclique is the best candidate.
        # If this fails, we can iterate over all maximal bicliques containing (u,v) if we generate them properly.
        # But the maximal one is unique for a given seed?
        # Yes, the closure of (u,v) is unique.
        
        L, R = find_bicliques_containing_edge(uncovered_mask, u, v)
        b_mask = biclique_to_mask(L, R)
        
        # Ensure we actually cover the seed edge
        if not (b_mask & (1 << e_idx)):
             return None
             
        new_mask = uncovered_mask & ~b_mask
        
        # Recurse
        res = solve(new_mask, max_bicliques - 1)
        if res is not None:
            return [(L, R)] + res
        else:
            return None

    # Try k = 1, 2, 3...
    # Upper bound: number of edges (each edge is a biclique)
    
    # To improve success rate, we can shuffle the edge selection?
    # No, determinism is preferred.
    # But if the maximal biclique choice is a dead end, we fail.
    # To handle this, we should try *all* maximal bicliques that contain the chosen edge?
    # Wait, is the maximal biclique unique?
    # Let's verify.
    # Graph: 4-cycle 0-1-2-3-0.
    # Edge (0,1).
    # L={0}, R={1}.
    # 2 is connected to 1. Can 2 join L? 2-1 is edge. 2-? R only has 1. Yes. L={0,2}.
    # 3 is connected to 0. Can 3 join R? 3-0 is edge. 3-? L has 0,2. 3-2 is not edge. So 3 cannot join R if 2 is in L.
    # If 2 didn't join L, could 3 join R?
    # If L={0}, R={1,3}.
    # Then 2 can join L? 2-1 (yes), 2-3 (no). So 2 cannot join L.
    # So we have two maximal bicliques containing (0,1):
    # 1. L={0,2}, R={1}
    # 2. L={0}, R={1,3}
    # My `find_bicliques_containing_edge` function adds *all* valid vertices.
    # In the loop, it adds 2 to L. Then it checks 3 for R. 3 is not connected to 2 (in L), so 3 is not added.
    # So it returns L={0,2}, R={1}.
    # It misses the other maximal biclique L={0}, R={1,3}.
    
    # So the maximal biclique is NOT unique.
    # We need to generate *all* maximal bicliques containing the seed edge.
    # This is the "Enumeration of Maximal Bicliques" problem.
    # For small n, we can do a DFS on the expansion.
    
    def generate_maximal_bicliques(uncovered_mask, u, v):
        """Generates all maximal bicliques containing (u,v)."""
        # State: (L, R)
        # We start with L={u}, R={v}.
        # We need to explore all ways to expand.
        # This can be exponential.
        # To prune, we can use the fact that we only care about *maximal* ones.
        
        # A simpler approach:
        # The set of all maximal bicliques containing (u,v) can be found by
        # iterating over all subsets of the "common neighbors" that can be added?
        
        # Let's use a recursive generator.
        # To avoid duplicates, we enforce an order on vertices added to L and R.
        
        def dfs(L, R, next_L_idx, next_R_idx):
            # Try to add vertices to L from next_L_idx onwards
            # Try to add vertices to R from next_R_idx onwards
            
            # 1. Check if we can add any more vertices
            added = False
            for i in range(next_L_idx, n):
                w = i
                if w not in L and w not in R:
                    valid = True
                    for x in R:
                        idx = edge_index.get((min(w,x), max(w,x)), -1)
                        if idx == -1 or not (uncovered_mask & (1 << idx)):
                            valid = False
                            break
                    if valid:
                        # Branch: Add w to L
                        yield from dfs(L | {w}, R, i + 1, next_R_idx)
                        added = True
                        # Branch: Do not add w to L (skip it for now, but it might be added later? No, if we skip it, we can't add it later in this path if we maintain order?
                        # Actually, standard enumeration:
                        # We must decide for each vertex w if it is in L, in R, or in neither.
                        # If it's in neither, it's not in the biclique.
                        # If we skip it, we can't add it later.
                        # So:
                        # 1. Add w to L
                        # 2. Add w to R (if valid)
                        # 3. Skip w (not in biclique)
                        
                        # We need to check validity for R too.
                        
            # This simple DFS might miss combinations.
            # Let's use a more robust enumeration:
            # The vertices not in L or R yet are candidates.
            # We partition the candidate set into L_add, R_add, and Ignore.
            
            # Given the complexity, and n=30, the number of maximal bicliques is usually small for random graphs.
            # Let's try a heuristic: Just try the "greedy maximal" (add all possible) and if that fails, we might miss solutions.
            # However, for k=2,3, the greedy maximal often works.
            # If we want to be safe, we can iterate over a few variations.
            
            # For now, let's stick to the single "greedy maximal" path but try different starting edges?
            # No, the seed edge is fixed by the first uncovered edge.
            
            # Let's refine: The "greedy maximal" adds all possible vertices.
            # If this leads to a dead end, it's possible a smaller biclique would work.
            # To handle this, we can backtrack on the expansion.
            
            pass # Placeholder

        # For the sake of the 60s limit, let's implement a limited backtracking search on the expansion of the seed biclique.
        # We'll generate up to K maximal bicliques containing the seed edge.
        
        bicliques = []
        
        def expand(L, R, candidates):
            # candidates is a list of vertices not in L or R
            if not candidates:
                # Check if maximal
                # If maximal, add to list
                bicliques.append((frozenset(L), frozenset(R)))
                return
                
            # Pick the first candidate
            w = candidates[0]
            rest = candidates[1:]
            
            # Option 1: Add to L
            valid_L = True
            for x in R:
                idx = edge_index.get((min(w,x), max(w,x)), -1)
                if idx == -1 or not (uncovered_mask & (1 << idx)):
                    valid_L = False
                    break
            if valid_L:
                expand(L | {w}, R, rest)
                
            # Option 2: Add to R
            valid_R = True
            for x in L:
                idx = edge_index.get((min(w,x), max(w,x)), -1)
                if idx == -1 or not (uncovered_mask & (1 << idx)):
                    valid_R = False
                    break
            if valid_R:
                expand(L, R | {w}, rest)
                
            # Option 3: Skip w
            expand(L, R, rest)

        candidates = [w for w in range(n) if w != u and w != v]
        expand({u}, {v}, candidates)
        
        # Deduplicate
        unique_bicliques = []
        seen = set()
        for L, R in bicliques:
            key = (L, R)
            key2 = (R, L)
            if key not in seen and key2 not in seen:
                seen.add(key)
                seen.add(key2)
                unique_bicliques.append((L, R))
                
        return unique_bicliques

    # Re-implement solve with multiple biclique options
    def solve(uncovered_mask, max_bicliques):
        if uncovered_mask == 0:
            return []
        if max_bicliques == 0:
            return None
            
        e_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        u, v = edge_list[e_idx]
        
        candidates = generate_maximal_bicliques(uncovered_mask, u, v)
        
        # Sort candidates by size (larger first) to find solutions faster
        candidates.sort(key=lambda x: len(x[0]) * len(x[1]), reverse=True)
        
        for L, R in candidates:
            b_mask = biclique_to_mask(L, R)
            if not (b_mask & (1 << e_idx)):
                continue
            new_mask = uncovered_mask & ~b_mask
            res = solve(new_mask, max_bicliques - 1)
            if res is not None:
                return [(L, R)] + res
                
        return None

    # Try increasing k
    # Note: generate_maximal_bicliques can be expensive.
    # We limit the number of candidates to explore?
    # For now, let's just run it.
    
    for k in range(1, m + 1):
        start_time = time.time()
        res = solve(full_mask, k)
        if res is not None:
            # Convert to required format
            result = []
            for L, R in res:
                left = [int(x) for x in sorted(L)]
                right = [int(x) for x in sorted(R)]
                result.append([left, right])
            return result
        if time.time() - start_time > 55:
            # Fallback to greedy if exact search is too slow
            break
            
    # Fallback: Greedy
    uncovered = set((min(a,b), max(a,b)) for a,b in edges)
    bicliques = []
    while uncovered:
        best_biclique = None
        best_size = 0
        for e in list(uncovered):
            u, v = e
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for w in range(n):
                    if w not in L and w not in R:
                        if all((min(w,x), max(w,x)) in uncovered for x in R):
                            L.add(w)
                            changed = True
                for w in range(n):
                    if w not in L and w not in R:
                        if all((min(w,x), max(w,x)) in uncovered for x in L):
                            R.add(w)
                            changed = True
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_biclique = (L, R)
        if best_biclique is None:
            break
        L, R = best_biclique
        bicliques.append([sorted(L), sorted(R)])
        for a in L:
            for b in R:
                uncovered.discard((min(a,b), max(a,b)))
                
    result = []
    for L, R in bicliques:
        left = [int(x) for x in L]
        right = [int(x) for x in R]
        result.append([left, right])
    return result

# EVOLVE-BLOCK-END
