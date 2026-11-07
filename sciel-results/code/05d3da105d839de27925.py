import math
import time

def construct():
    n = 256
    # 1. Build the complement graph adjacency list.
    # In the complement graph, an edge exists if gcd(u, v) == 1.
    # We are looking for a Maximum Weight Clique in this graph.
    
    adj = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if math.gcd(i, j) == 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Sort vertices by weight (value) descending.
    # We will process them in this order to find high weight solutions early.
    vertices = list(range(n, 0, -1))
    
    best_weight = 0
    best_clique = []
    
    # Time limit guard
    start_time = time.time()
    time_limit = 55.0 # 55 seconds to be safe
    
    def greedy_color_bound(candidates):
        """
        Returns an upper bound on the maximum weight clique in the subgraph
        induced by 'candidates' in the complement graph.
        Uses greedy coloring on the complement graph subgraph.
        Vertices in the same color class are NOT adjacent in the complement graph,
        meaning they ARE adjacent in the original graph (not coprime).
        Wait, let's re-verify the logic.
        
        We are finding a Clique in G_bar.
        We color G_bar.
        If u, v have the same color in G_bar, they are NOT adjacent in G_bar.
        Therefore, they cannot both be in a Clique of G_bar.
        So a Clique can have at most one vertex from each color class.
        The upper bound is the sum of the max weight vertex in each color class.
        """
        if not candidates:
            return 0
            
        # Order candidates by weight descending for tighter bound?
        # Usually, greedy coloring works better if we process high-degree or high-weight nodes first.
        # Let's just iterate through the sorted candidates list.
        # 'candidates' is a list of ints, already sorted descending by value.
        
        color_classes = [] # List of lists of vertices
        vertex_to_color = {}
        
        for v in candidates:
            # Find a color for v
            assigned = False
            for k, color_verts in enumerate(color_classes):
                # Check if v is adjacent to any vertex in this color class in G_bar
                # If it is NOT adjacent to any, it can take this color.
                # In G_bar, edge means coprime.
                # So we check if gcd(v, u) == 1 for any u in color_verts.
                # If gcd(v, u) == 1 for any u, then v is adjacent to u in G_bar, so they must have different colors.
                # We want a color where v is NOT adjacent to any existing vertex.
                is_conflict = False
                for u in color_verts:
                    if u in adj[v]: # u is in adj[v] means gcd(u,v)==1, i.e., adjacent in G_bar
                        is_conflict = True
                        break
                if not is_conflict:
                    color_classes[k].append(v)
                    vertex_to_color[v] = k
                    assigned = True
                    break
            
            if not assigned:
                color_classes.append([v])
                vertex_to_color[v] = len(color_classes) - 1
        
        # Calculate bound: sum of max weight in each color class
        bound = 0
        for color_verts in color_classes:
            # Since candidates are sorted descending, the first one added to a color class 
            # isn't necessarily the max weight if we just append. 
            # But we iterate 'candidates' in descending weight order.
            # So the first vertex added to a color class is the highest weight in that class?
            # Yes, because we iterate from highest to lowest.
            # So max weight is color_verts[0].
            bound += color_verts[0]
            
        return bound

    def expand(candidates, current_weight, current_clique):
        nonlocal best_weight, best_clique
        
        if time.time() - start_time > time_limit:
            return
            
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_clique[:]
            return
        
        # Calculate upper bound
        ub = greedy_color_bound(candidates)
        if current_weight + ub <= best_weight:
            return
        
        # Pick a vertex to branch on.
        # Heuristic: Pick the vertex with the highest weight.
        # Since candidates is sorted descending, candidates[0] is the highest.
        v = candidates[0]
        
        # Branch 1: Include v
        # New candidates must be neighbors of v in G_bar.
        new_candidates = [u for u in candidates[1:] if u in adj[v]]
        expand(new_candidates, current_weight + v, current_clique + [v])
        
        # Branch 2: Exclude v
        # New candidates are just the rest.
        expand(candidates[1:], current_weight, current_clique)

    # Start the search
    # We include vertex 1? 
    # Vertex 1 is coprime with everything.
    # It has weight 1.
    # The algorithm will handle it.
    
    expand(vertices, 0, [])
    
    # The result is the list of integers in the best clique found.
    # The task asks for a list of ints.
    return best_clique

# EVOLVE-BLOCK-END
