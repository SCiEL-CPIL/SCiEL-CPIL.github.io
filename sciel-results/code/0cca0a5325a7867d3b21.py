"""Rewritten program using a greedy heuristic to find a large kissing configuration in R^11."""
import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate candidate vectors: weight 2 and weight 3
    candidates = []
    indices = range(D)
    
    # Weight 2 vectors: s1*e_i + s2*e_j
    for i, j in itertools.combinations(indices, 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i] = s1
                v[j] = s2
                candidates.append(v)
                
    # Weight 3 vectors: s1*e_i + s2*e_j + s3*e_k
    for i, j, k in itertools.combinations(indices, 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    candidates.append(v)
                    
    n = len(candidates)
    if n == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    # Precompute norms squared
    norms2 = np.array([sum(x*x for x in v) for v in candidates])
    
    # Build conflict graph
    # Conflict if dist2 < max(norm_u2, norm_v2)
    # dist2 = norm_u2 + norm_v2 - 2 dot(u,v)
    # Condition: norm_u2 + norm_v2 - 2 dot(u,v) < max(norm_u2, norm_v2)
    # Let A = norm_u2, B = norm_v2.
    # A + B - 2 dot < max(A,B)
    # If A >= B: A + B - 2 dot < A => B - 2 dot < 0 => 2 dot > B => dot > B/2
    # If B > A: A + B - 2 dot < B => A - 2 dot < 0 => 2 dot > A => dot > A/2
    # So conflict if dot > min(norm_u2, norm_v2) / 2
    
    # Since norms are integers (2 or 3), min(norm) is 2 or 3.
    # If both W2: min=2. Conflict if dot > 1. Max dot for distinct W2 is 1 (share 1 idx) or 2 (same idx, diff sign? No, distinct vectors).
    # Wait, if u=e1+e2, v=e1+e2 (same) -> dot 2. But they are distinct in list? No, unique vectors.
    # If u=e1+e2, v=e1-e2 -> dot 0.
    # If u=e1+e2, v=e1+e3 -> dot 1. 1 > 1 is False. So no conflict.
    # So W2-W2 never conflict?
    # Let's re-verify.
    # u=e1+e2, v=e1+e2 is not a pair.
    # u=e1+e2, v=e2+e1 is same.
    # Distinct W2 vectors share at most 1 index with same sign?
    # u=e1+e2, v=e1+e3. Share e1. Dot 1. min norm 2. 1 > 1 False. OK.
    # u=e1+e2, v=-e1+e2. Share e2. Dot 1. 1 > 1 False. OK.
    # u=e1+e2, v=e1-e2. Share e1. Dot 1. 1 > 1 False. OK.
    # So W2 nodes form an independent set.
    
    # W2-W3: min norm 2. Conflict if dot > 1.
    # u=W2, v=W3. Max dot 2.
    # If dot 2, conflict.
    
    # W3-W3: min norm 3. Conflict if dot > 1.5 => dot >= 2.
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dot = sum(candidates[i][k] * candidates[j][k] for k in range(D))
            min_norm2 = min(norms2[i], norms2[j])
            if dot > min_norm2 / 2.0:
                adj[i].add(j)
                adj[j].add(i)
                
    # MIS Solver (Branch and Bound with Coloring)
    # We want to find the largest independent set.
    # Independent set in G is Clique in complement G'.
    # Or just run MIS on G directly.
    
    best_set = set()
    
    def color_sort(vertices):
        # Greedy coloring of the subgraph induced by vertices
        # Returns list of (color, vertex) sorted by color
        # Colors are 1..k
        colors = {}
        order = []
        # Simple greedy: sort by degree in subgraph? Or just process in given order?
        # Tomita suggests sorting by degree first.
        # Let's just process in the order given, assigning smallest available color.
        # For efficiency, we can just assign colors 1..c such that no two adjacent have same color.
        # Actually, the bound is the number of colors used.
        # We return the vertices sorted by color number.
        
        used_colors = []
        # To make it a valid upper bound, we need a proper coloring.
        # Greedy coloring:
        for v in vertices:
            # Find neighbors of v that are in vertices
            neigh_colors = set()
            for u in adj[v]:
                if u in colors:
                    neigh_colors.add(colors[u])
            c = 1
            while c in neigh_colors:
                c += 1
            colors[v] = c
            used_colors.append(c)
            
        # Sort vertices by color
        sorted_vertices = sorted(vertices, key=lambda x: colors[x])
        return sorted_vertices

    def dfs(candidates_list, current_set):
        nonlocal best_set
        
        if not candidates_list:
            if len(current_set) > len(best_set):
                best_set = set(current_set)
            return
            
        # Upper bound: len(current_set) + len(candidates_list)
        # Better bound: len(current_set) + number of colors in greedy coloring of candidates_list
        sorted_cands = color_sort(candidates_list)
        
        # If even the max possible size is not better, prune
        # The max size of independent set in candidates_list is <= number of colors
        # Let max_color = max(colors.values())
        # We can just use the length of the color list? No, the number of distinct colors.
        # Actually, the bound is the number of colors used.
        
        # Optimization: If len(current_set) + len(sorted_cands) <= len(best_set), prune?
        # No, that's a weak bound.
        # The coloring bound is: max independent set size <= number of colors.
        # So if len(current_set) + num_colors <= len(best_set), prune.
        
        # Let's compute num_colors
        # In color_sort, we can return the max color used.
        
        # Re-implement color_sort to return max_color
        def color_sort_with_bound(vertices):
            if not vertices:
                return [], 0
            colors = {}
            max_c = 0
            for v in vertices:
                neigh_colors = set()
                for u in adj[v]:
                    if u in colors:
                        neigh_colors.add(colors[u])
                c = 1
                while c in neigh_colors:
                    c += 1
                colors[v] = c
                if c > max_c:
                    max_c = c
            sorted_vertices = sorted(vertices, key=lambda x: colors[x])
            return sorted_vertices, max_c

        sorted_cands, bound = color_sort_with_bound(candidates_list)
        
        if len(current_set) + bound <= len(best_set):
            return

        # Branch on the last vertex in the sorted list (highest color)
        # This is a common heuristic.
        # Actually, iterating from highest color down is standard.
        
        # We iterate through the sorted candidates.
        # For each candidate v, we include it and recurse on the remaining candidates
        # that are not adjacent to v.
        
        # To avoid duplicate work, we only consider v if it is the "first" vertex of its color in the remaining set?
        # Standard Tomita:
        # Sort vertices by color.
        # For i from len(sorted)-1 down to 0:
        #   v = sorted[i]
        #   if len(current_set) + i + 1 <= len(best_set): break
        #   Include v
        #   New candidates = { u in sorted[0..i-1] | u not in adj[v] }
        #   Recurse
        
        for i in range(len(sorted_cands) - 1, -1, -1):
            if len(current_set) + i + 1 <= len(best_set):
                break
            
            v = sorted_cands[i]
            current_set.add(v)
            
            # Filter candidates: must be in sorted_cands[0..i-1] and not adjacent to v
            new_cands = [u for u in sorted_cands[:i] if u not in adj[v]]
            
            dfs(new_cands, current_set)
            
            current_set.remove(v)

    # Initial call
    # We can start with all nodes.
    # To help the solver, we might want to sort initial nodes by degree?
    # The coloring heuristic inside will handle some of that.
    
    dfs(list(range(n)), set())
    
    # Extract vectors
    if not best_set:
        # Fallback to W2 only if solver fails or finds nothing (should not happen)
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result_vectors = [candidates[i] for i in best_set]
    return np.array(result_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
