"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s wall-clock limit

    # We need a subset S of [1, 256] such that for all distinct a, b in S,
    # (a*b) % (a+b) != 0.
    # This is equivalent to finding a Maximum Independent Set (MIS) in a graph G
    # where vertices are 1..256, and an edge exists between a and b if (a*b) % (a+b) == 0.
    #
    # We solve this using the Tomita Maximum Clique algorithm on the COMPLEMENT graph.
    # In the complement graph, an edge exists between a and b if (a*b) % (a+b) != 0.
    # A clique in the complement graph is an independent set in the original graph.
    #
    # The Tomita algorithm uses a greedy coloring bound to prune the search space.

    N = 256
    nodes = list(range(1, N + 1))
    
    # Build adjacency for the "compatible" graph (complement of conflict graph)
    # adj[u] is a set of v such that u and v are compatible (can both be in S)
    # i.e., (u*v) % (u+v) != 0
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) != 0:
                adj[a].add(b)
                adj[b].add(a)
    
    # Convert to sorted lists for faster intersection
    adj_list = [sorted(adj[i]) for i in range(N + 1)]
    
    # Tomita's Maximum Clique Algorithm
    # We work on the "compatible" graph, finding a maximum clique.
    
    best_clique = []
    best_size = 0
    best_clique_set = set()
    
    # Greedy coloring for upper bound
    # Returns a list of colors (upper bound on clique size in the subgraph)
    # and the order of vertices by color class
    def color_sort(vertices, adj_list):
        """
        Greedily color the subgraph induced by vertices.
        Returns (colors, order) where colors[i] is the color of vertices[order[i]],
        and order is sorted such that vertices with lower colors come first.
        Actually, for Tomita, we want vertices sorted by decreasing color.
        The color of a vertex is an upper bound on the clique size in its remaining subgraph.
        """
        # We'll implement a simple greedy coloring that also gives an order.
        # Standard approach: maintain uncolored set, color them one by one.
        # For efficiency in the branch and bound, we want to process vertices
        # in an order that allows tight bounds.
        
        # Simpler approach: just return a list of vertices and their color numbers.
        # The bound at a node with candidate set P is the maximum color assigned.
        
        uncolored = list(vertices)
        colors = {}
        order = []
        
        # Simple greedy coloring
        # We don't strictly need the optimal coloring, just a valid one for the bound.
        # A better ordering (higher colors first) helps prune.
        
        # Let's use a standard sequential coloring.
        # To make it fast, we just assign colors greedily.
        
        # Actually, for the Tomita algorithm, the specific implementation often
        # interleaves coloring and sorting.
        
        # Let's implement a simpler recursive search with a good bound.
        # Bound: max number of vertices in any independent set of the candidate graph?
        # No, we are finding a clique in the compatible graph.
        # The size of the maximum clique in a graph is bounded by the chromatic number.
        # So if we color the compatible graph subgraph, the max color is an upper bound on the clique size.
        
        # Greedy Coloring:
        # 1. Start with all vertices uncolored.
        # 2. Pick a vertex, assign the smallest available color.
        # 3. Remove it from uncolored.
        # To get a good bound, we should pick vertices in a specific order (e.g., high degree first).
        
        # Let's try a simple heuristic: sort vertices by degree in the subgraph descending.
        # Then color them.
        
        # But we also need the order for the search.
        # Tomita's algorithm sorts vertices by their color, descending.
        
        # Implementation:
        # 1. Compute degrees in the subgraph.
        # 2. Sort vertices by degree descending.
        # 3. Greedily assign colors.
        # 4. Return vertices sorted by color descending.
        
        if not uncolored:
            return [], []
            
        # Degrees in subgraph
        sub_adj = {v: adj_list[v] & set(uncolored) for v in uncolored}
        degrees = {v: len(sub_adj[v]) for v in uncolored}
        
        # Sort by degree descending
        ordered_vertices = sorted(uncolored, key=lambda x: degrees[x], reverse=True)
        
        # Greedy Coloring
        color_of = {}
        # We want to assign the smallest possible color to each vertex in the order.
        # But for the bound, we just need a valid coloring.
        # To maximize the bound (which helps pruning? No, we want the bound to be as tight as possible, i.e., small).
        # Wait, the bound is the max color. If we use fewer colors, the bound is smaller, which is better for pruning.
        # So we want a "good" coloring (few colors).
        # Standard greedy coloring: for each vertex in order, assign the smallest color not used by its neighbors.
        
        used_colors_by_vertex = {}
        max_color = 0
        
        for v in ordered_vertices:
            # Find colors used by neighbors of v that are already colored
            neighbor_colors = set()
            for u in sub_adj[v]:
                if u in color_of:
                    neighbor_colors.add(color_of[u])
            
            # Find smallest color not in neighbor_colors
            c = 1
            while c in neighbor_colors:
                c += 1
            
            color_of[v] = c
            max_color = max(max_color, c)
            used_colors_by_vertex[v] = c
            
        # Now, sort vertices by color descending for the search order.
        # Vertices with higher colors are more likely to be in large cliques?
        # Actually, in Tomita's algorithm, the order is by decreasing color.
        
        sorted_vertices = sorted(uncolored, key=lambda x: color_of[x], reverse=True)
        
        return color_of, sorted_vertices

    def expand(clique, candidates):
        nonlocal best_clique, best_size, best_clique_set
        nonlocal start_time
        
        if time.time() > DEADLINE:
            return
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
                best_clique_set = set(clique)
            return
        
        # Color the candidate subgraph
        color_of, ordered_candidates = color_sort(candidates, adj_list)
        
        if not ordered_candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
                best_clique_set = set(clique)
            return
        
        # Process vertices in order (decreasing color)
        # We iterate through the ordered candidates.
        # For each vertex v, we consider adding it to the clique.
        # The new candidate set will be the intersection of current candidates
        # and the neighbors of v in the compatible graph.
        
        # To maintain the order and bound, we typically iterate and update candidates.
        
        # Standard Tomita loop:
        # for i in range(len(ordered_candidates)):
        #     v = ordered_candidates[i]
        #     if len(clique) + color_of[v] <= best_size: break
        #     ...
        
        # Note: color_of[v] is the color assigned in the coloring of the *original* candidate set.
        # As we remove vertices, the bound might be loose, but it's still valid.
        
        for i in range(len(ordered_candidates)):
            v = ordered_candidates[i]
            
            # Pruning: if even with the max color we can't beat best_size
            if len(clique) + color_of[v] <= best_size:
                break
            
            # Add v to clique
            new_clique = clique + [v]
            
            # New candidates: intersection of current candidates (excluding v and previous ones?)
            # Actually, we should intersect the *remaining* candidates (from i+1 onwards?) 
            # No, in Tomita, the candidates are the neighbors of v among the *current* candidate set.
            # But we must ensure we don't revisit vertices.
            # The standard implementation iterates over the list, and for each v,
            # the new candidate set is {u in candidates | u > v in order? No.}
            
            # Let's stick to a clean recursive formulation:
            # Candidates is a set.
            # We pick a vertex v from candidates.
            # New candidates = candidates ∩ neighbors(v).
            # But to avoid symmetry, we usually process vertices in a specific order and only consider neighbors that come "after" v in that order?
            # Or simply:
            # for v in candidates (in some order):
            #    new_candidates = candidates ∩ adj[v]
            #    expand(clique + [v], new_candidates)
            #    candidates.remove(v)  # to avoid re-adding v in subsequent iterations?
            
            # Let's implement it this way:
            # We iterate through the ordered list.
            # For each v, we compute new_candidates as the intersection of the *current* candidate set
            # (which shrinks as we iterate) and adj[v].
            
            # Wait, the `candidates` passed to this function is a set.
            # We should convert it to a list or work with the set directly.
            
            # Let's refine the expand function to work with a list of candidates that we mutate or pass copies.
            # Passing copies is expensive. Mutating is tricky with recursion.
            
            # Alternative: Pass a set of candidates.
            # new_candidates = candidates & adj_set[v]
            # This is O(|candidates|).
            
            # Let's try this.
            
            # Optimization: Only consider neighbors that are in the candidate set.
            # adj_set is not built, we have adj_list.
            # We can build a set of neighbors for v on the fly or pre-compute.
            # Pre-computing adj_set is faster for intersection.
            
            # Let's assume adj_set is available.
            
            # For now, let's just do:
            # neighbors_in_candidates = [u for u in candidates if u in adj_set[v]]
            # This is slow.
            
            # Better: adj_set[v] & candidates
            
            # Let's pre-compute adj_set.
            
            pass # Placeholder for logic

    # Pre-compute adj_set for faster intersection
    adj_set = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        adj_set[i] = set(adj_list[i])
    
    # Revised Expand
    def expand(clique, candidates):
        nonlocal best_clique, best_size, best_clique_set
        nonlocal start_time
        
        if time.time() > DEADLINE:
            return
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
                best_clique_set = set(clique)
            return
        
        # Color the candidate subgraph
        color_of, ordered_candidates = color_sort(candidates, adj_list)
        
        if not ordered_candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
                best_clique_set = set(clique)
            return
        
        # Iterate through ordered candidates
        # We need to be careful: as we pick v, we remove it from the candidate pool for subsequent iterations?
        # In the standard algorithm, we iterate through the list.
        # The bound check uses the color of v.
        # The new candidate set for the recursive call is the intersection of the *current* candidate set
        # and the neighbors of v.
        
        # To ensure we explore all maximal cliques, we typically remove v from the candidate set
        # after processing it, or rely on the fact that we are generating cliques in a specific order.
        
        # Let's use a mutable list for candidates to allow removal?
        # Or just use a set and create a new set for the recursion.
        
        for i in range(len(ordered_candidates)):
            v = ordered_candidates[i]
            
            # Pruning
            if len(clique) + color_of[v] <= best_size:
                break
            
            # New candidates: neighbors of v in the candidate set
            # Note: We should probably exclude v itself (it's not a neighbor of itself)
            # And we should probably only consider candidates that are "available".
            
            # In the standard Tomita algorithm, the candidate set P is a set.
            # We pick v from P.
            # New P' = P ∩ N(v).
            # We then recurse with P'.
            # After returning, we remove v from P.
            # This ensures we don't pick v again in the same level.
            
            new_candidates = candidates & adj_set[v]
            
            expand(clique + [v], new_candidates)
            
            # Remove v from candidates for the next iteration
            # This is crucial to avoid finding the same clique multiple times
            # or to respect the ordering.
            candidates.discard(v)
            
            if time.time() > DEADLINE:
                return

    # Initial call
    # Start with an empty clique and all nodes as candidates
    initial_candidates = set(range(1, N + 1))
    expand([], initial_candidates)
    
    if best_clique:
        return sorted(best_clique)
    else:
        # Fallback: return a valid non-empty subset
        # e.g., just {1}
        return [1]

# EVOLVE-BLOCK-END
