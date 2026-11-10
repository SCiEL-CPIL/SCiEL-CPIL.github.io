import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    N = 256
    nodes = list(range(1, N + 1))
    
    # 1. Filter out nodes where 2*a is a perfect square
    valid_nodes = []
    for a in nodes:
        s = 2 * a
        r = int(math.isqrt(s))
        if r * r != s:
            valid_nodes.append(a)
            
    # Map original values to indices in valid_nodes
    val_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    n_valid = len(valid_nodes)
    
    # 2. Build conflict graph G (edges if a+b is square)
    # We will build the complement graph's adjacency for Max Clique
    # adj[i] is a set of indices j such that i and j are compatible (a_i + a_j is NOT square)
    # Note: In Max Clique on complement, we want nodes that are NOT connected in G.
    # So if a_i + a_j is square, they are incompatible (edge in G, no edge in complement).
    # If a_i + a_j is NOT square, they are compatible (no edge in G, edge in complement).
    
    # Precompute squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
        
    # Initialize adjacency for complement graph
    # Using list of sets for flexibility, or list of lists?
    # For Tomita, we need to efficiently intersect sets.
    # Let's use list of sets for neighbors in complement graph.
    comp_adj = [set() for _ in range(n_valid)]
    
    for i in range(n_valid):
        a = valid_nodes[i]
        for j in range(i + 1, n_valid):
            b = valid_nodes[j]
            s = a + b
            if s in squares:
                # Conflict in G -> No edge in complement
                pass
            else:
                # Compatible -> Edge in complement
                comp_adj[i].add(j)
                comp_adj[j].add(i)
                
    # 3. Max Clique Algorithm on comp_adj
    # Tomita's algorithm with greedy coloring
    
    best_clique = []
    
    def greedy_coloring(vertices):
        """
        Greedy coloring of the subgraph induced by vertices.
        Returns a list of colors assigned to vertices in the order they are processed.
        We want to order vertices such that vertices with high color come first in the branch?
        Standard Tomita:
        1. Order vertices by greedy coloring.
        2. Process in reverse order of colors?
        
        Actually, the standard implementation:
        - Maintain a list P of candidate vertices.
        - Color P greedily.
        - The color of a vertex gives an upper bound on the clique size extending from that vertex.
        - We iterate through P in reverse order of their colors (highest color first).
        """
        colors = {}
        color_map = {} # vertex -> color index
        # Simple greedy: sort by degree in induced subgraph? Or just iterate?
        # To get a tight bound, we should color carefully.
        # Standard: 
        #   available = set(vertices)
        #   color_idx = 0
        #   while available:
        #       color_idx += 1
        #       # Pick a maximal independent set in the remaining graph? No, that's hard.
        #       # Greedy: pick a vertex, assign color, remove from available, remove neighbors from available?
        #       # This is not standard coloring.
        
        # Let's use the standard "Color Sort" from Tomita et al.
        # 1. Sort vertices by degree in the induced subgraph (descending).
        # 2. Assign colors greedily.
        
        # For simplicity and speed in Python, we might just use a simple heuristic:
        # Iterate through vertices. If it's not colored, assign the smallest possible color?
        # No, we want to group compatible vertices into same color?
        # In Clique finding, the bound is the number of colors.
        # If we color the graph such that no two adjacent vertices share a color,
        # then a clique can contain at most one vertex from each color class.
        # So the max clique size <= number of colors.
        
        # Greedy coloring algorithm:
        # 1. Order vertices (e.g., by degree descending).
        # 2. For each vertex in order, assign the smallest color not used by its already-colored neighbors.
        
        # To make the bound tight, we want to use as FEW colors as possible?
        # No, the bound is the MAX color index used.
        # If we use fewer colors, the bound is tighter (smaller).
        # So we want a good coloring (few colors).
        
        # However, for the branching order, we process vertices in decreasing order of their assigned color.
        
        # Let's implement a simple greedy coloring.
        # Since n is small, we can be a bit slow here.
        
        # Sort by degree in induced subgraph
        degs = []
        for v in vertices:
            deg = 0
            for u in vertices:
                if u != v and u in comp_adj[v]:
                    deg += 1
            degs.append((deg, v))
        degs.sort(reverse=True)
        
        vertex_order = [v for _, v in degs]
        
        color_of = {}
        next_color = 0
        for v in vertex_order:
            # Find neighbors of v that are already colored
            neighbor_colors = set()
            for u in vertex_order:
                if u in color_of and u in comp_adj[v]:
                    neighbor_colors.add(color_of[u])
            
            # Assign smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            if c > next_color:
                next_color = c
                
        # Return list of (vertex, color) sorted by color descending
        # We will process in this order.
        # The bound for a vertex v is color_of[v] + 1 (since colors are 0-indexed, max size is max_color+1)
        # Actually, if colors are 0..k, max clique size is <= k+1.
        
        sorted_vertices = sorted(vertex_order, key=lambda x: color_of[x], reverse=True)
        return sorted_vertices, color_of, next_color + 1

    def expand(clique, candidates):
        nonlocal best_clique, deadline
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if len(clique) > len(best_clique):
                best_clique = clique.copy()
            return
            
        # Greedy coloring
        sorted_cands, colors, max_bound = greedy_coloring(candidates)
        
        # If current clique size + max possible (max_bound) <= best_clique size, prune
        if len(clique) + max_bound <= len(best_clique):
            return
            
        for v in sorted_cands:
            if time.time() > deadline:
                return
                
            # Prune if even taking this vertex and all remaining colors doesn't beat best
            # The bound for this specific branch is len(clique) + 1 + (max color of remaining candidates compatible with v)
            # Simpler: The global bound for this node was max_bound.
            # When we pick v, the new candidates are candidates that are compatible with v and come AFTER v in the sorted list?
            # No, in Tomita, we maintain the candidate set.
            
            # Let's stick to the standard recursive structure.
            # The bound for the current node is max_bound.
            # If len(clique) + max_bound <= len(best_clique), we would have pruned above.
            
            # We branch on v.
            # New clique: clique + [v]
            # New candidates: candidates that are neighbors of v in comp_adj AND were in candidates.
            # To maintain the ordering and bound, we should filter the sorted_cands.
            
            new_clique = clique + [v]
            
            # Find neighbors of v in the current candidate set
            # Note: candidates is a set/list. We need to check membership in comp_adj[v].
            new_candidates = set()
            for u in candidates:
                if u in comp_adj[v]:
                    new_candidates.add(u)
            
            # Recursive call
            expand(new_clique, new_candidates)
            
            # Remove v from candidates for subsequent siblings
            # In the loop, we just move to the next v.
            # But we must ensure that v is not considered again.
            # The loop iterates over sorted_cands.
            # We should remove v from the set of candidates available for subsequent branches?
            # Yes.
            candidates.discard(v)
            
            # Update bound?
            # The bound for the next iteration is determined by the new candidates.
            # We could re-color, but that's expensive.
            # Standard Tomita re-colors at each node.
            # Given N=245, re-coloring at each node might be too slow if depth is large.
            # However, the depth is bounded by the clique size.
            # If the clique size is ~100, and branching factor is high, it explodes.
            
            # Optimization:
            # If len(new_clique) + len(new_candidates) <= len(best_clique):
            #     continue # Simple bound
            
    # Initial call
    # candidates is all valid nodes
    initial_candidates = set(range(n_valid))
    expand([], initial_candidates)
    
    # Map indices back to values
    result_vals = [valid_nodes[i] for i in best_clique]
    
    # Verify
    def is_valid(S):
        S_set = set(S)
        for a in S:
            if 2*a in squares: return False
            for b in S:
                if a < b and (a+b) in squares: return False
        return True
        
    if not is_valid(result_vals):
        # Fallback to seed
        return list(range(1, 257, 3))
        
    return result_vals
