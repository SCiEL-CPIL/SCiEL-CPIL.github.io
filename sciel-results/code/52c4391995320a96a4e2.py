"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses the Tomita Algorithm (Maximum Clique with Greedy Coloring) for efficiency.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # Build adjacency matrix
    # adj[i] will be a set of neighbors for fast lookup, but we also need ordered lists for Tomita
    adj_sets = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            l = (i * j) // gcd(i, j)
            if l <= N:
                adj_sets[i].add(j)
                adj_sets[j].add(i)
                
    # Sort nodes by degree descending for better initial ordering
    # This helps in the coloring step and general B&B efficiency
    sorted_nodes = sorted(nodes, key=lambda x: len(adj_sets[x]), reverse=True)
    
    # Map original node ID to its index in sorted_nodes? 
    # Actually, Tomita works on the list of candidates directly.
    # We will maintain a list of candidates.
    
    best_clique = []
    best_size = 0
    
    # Greedy Coloring for the subgraph induced by candidates
    # Returns a list of colors for each vertex in candidates, such that
    # vertices with the same color are not adjacent.
    # The maximum color used is an upper bound on the clique size.
    def color_sort(candidates):
        # candidates is a list of node IDs
        # We want to return (ordered_candidates, color_bounds)
        # where color_bounds[i] is the color assigned to ordered_candidates[i]
        # and the list is ordered such that colors are non-decreasing? 
        # Actually, standard Tomita orders by color, then by degree within color.
        # Let's implement a simple greedy coloring that returns the color of each vertex.
        
        uncolored = list(candidates)
        colors = {}
        color_classes = [] # List of lists, color_classes[c] contains vertices with color c+1
        
        # Simple greedy: iterate through uncolored, assign smallest possible color
        # To make it effective for pruning, we often sort vertices by degree first?
        # The Tomita algorithm typically processes vertices in a specific order.
        # Let's use the standard approach:
        # 1. Order vertices by degree in the subgraph (descending).
        # 2. Assign colors greedily.
        
        # Compute degrees in subgraph for sorting
        sub_degrees = {}
        for u in candidates:
            sub_degrees[u] = len(adj_sets[u] & set(candidates))
            
        # Sort by degree descending
        ordered_vertices = sorted(candidates, key=lambda x: sub_degrees[x], reverse=True)
        
        # Greedy Coloring
        # We assign colors 1, 2, ...
        # A vertex can have color c if it is not adjacent to any vertex with color c.
        # We maintain for each color the set of vertices that have that color.
        # To speed up, we can just check against the list of vertices with that color.
        
        vertex_colors = {}
        color_vertices = {} # color -> list of vertices
        
        for v in ordered_vertices:
            # Try to assign the smallest possible color
            assigned_color = None
            for c in range(1, len(color_vertices) + 1):
                # Check if v is adjacent to any vertex in color_vertices[c]
                # If not, v can take this color
                # Optimization: If we just check existence, it's O(V). 
                # For N=512, this is fine.
                is_conflict = False
                for u in color_vertices.get(c, []):
                    if u in adj_sets[v]:
                        is_conflict = True
                        break
                if not is_conflict:
                    assigned_color = c
                    break
            
            if assigned_color is None:
                assigned_color = len(color_vertices) + 1
                
            vertex_colors[v] = assigned_color
            if assigned_color not in color_vertices:
                color_vertices[assigned_color] = []
            color_vertices[assigned_color].append(v)
        
        # Now we need to order the candidates for the B&B search.
        # Tomita algorithm searches in reverse order of colors?
        # Usually, we expand the branch with the highest color bound first.
        # So we sort vertices by their color.
        # Vertices with higher colors are more "constrained" or "valuable" for the bound.
        # We want to try vertices that allow for larger cliques first.
        # Standard Tomita:
        # Expand vertex v.
        # New candidates = neighbors of v in current candidates.
        # The upper bound is the max color in the new candidates.
        
        # To make the pruning effective, we should process vertices with HIGHER colors first?
        # No, the bound is derived from the coloring of the *remaining* candidates.
        # The algorithm usually iterates through the candidate list.
        # If we sort candidates by color ascending, the "remaining" part has higher max colors?
        # Let's follow the standard "MCQ" or "Tomita" implementation:
        # 1. Color the graph.
        # 2. Order vertices by color (ascending).
        # 3. In the recursive step, iterate from the END of the list (highest color) to the beginning.
        #    This ensures that when we pick a vertex, the remaining candidates (which are a prefix of the list)
        #    have a max color that provides a tight upper bound.
        
        # Let's create the ordered list: sorted by color ascending.
        ordered_candidates = sorted(candidates, key=lambda x: vertex_colors[x])
        
        # color_bounds[i] = max color in ordered_candidates[0..i]
        # Actually, for the bound in the loop, we need the max color of the *remaining* candidates.
        # If we iterate from the end, the remaining candidates are ordered_candidates[0..i-1].
        # The max color in that prefix is vertex_colors[ordered_candidates[i-1]]?
        # No, because colors are not necessarily monotonic in the sorted list if we just sort by color.
        # Wait, if we sort by color, then ordered_candidates[0] has color 1, ..., ordered_candidates[-1] has color K.
        # The max color in the prefix [0..i] is simply the color of the element at i (since they are sorted by color).
        # So if we iterate i from len-1 down to 0:
        #   Pick v = ordered_candidates[i]
        #   Remaining candidates for next step = neighbors of v in ordered_candidates[0..i-1]
        #   Upper bound for this branch = len(current_clique) + max_color(remaining)
        #   Since remaining is a subset of ordered_candidates[0..i-1], its max color is <= vertex_colors[ordered_candidates[i-1]].
        #   Actually, we can precompute the max color for each prefix.
        
        # Let's return the ordered list and the color of each vertex.
        return ordered_candidates, vertex_colors

    def expand(clique, candidates_list):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if not candidates_list:
            if len(clique) > best_size:
                best_clique = clique[:]
                best_size = len(clique)
            return
            
        # Color the subgraph induced by candidates_list
        ordered_cands, colors = color_sort(candidates_list)
        
        # Iterate from the end (highest color) to the beginning
        # This is crucial for the pruning bound
        for i in range(len(ordered_cands) - 1, -1, -1):
            if time.time() > deadline:
                return
                
            v = ordered_cands[i]
            
            # Pruning:
            # The max possible size of a clique from candidates_list[0..i] is bounded by colors[v]
            # because colors[v] is the max color in the prefix [0..i] (due to sorting).
            # Wait, is it the max color?
            # Yes, if we sort by color, the color of element i is the max color in [0..i].
            if len(clique) + colors[v] <= best_size:
                return # Since we are going downwards, all subsequent i will have <= colors[v], so we can break/return
            
            # Include v
            clique.append(v)
            
            # New candidates: neighbors of v that are in candidates_list[0..i-1]
            # We need to filter the prefix ordered_cands[0..i-1] to only those adjacent to v.
            new_cands = [u for u in ordered_cands[:i] if u in adj_sets[v]]
            
            expand(clique, new_cands)
            
            clique.pop()
            
            # Remove v from candidates for the "exclude v" branch
            # In the loop structure, we don't explicitly "remove" v from the list for the next iteration
            # because the next iteration uses i-1, effectively ignoring v.
            # However, we must ensure that the "remaining" candidates for the NEXT iteration (i-1)
            # do NOT include v. Since we are iterating over a static list `ordered_cands` and using index `i`,
            # and the recursive call uses a *new* list `new_cands` derived from the prefix,
            # the "exclude v" branch is handled by simply decrementing i in the loop.
            # The candidates for the next iteration (i-1) are ordered_cands[0..i-1], which excludes v.
            
            # Note: We do NOT need to explicitly remove v from a shared list because
            # the next recursive call gets a fresh list, and the loop variable i controls the prefix.

    # Start
    # Initial candidates: all nodes
    # We should pass them in a way that allows coloring.
    expand([], sorted_nodes)
    
    return sorted(best_clique)

# EVOLVE-BLOCK-END
