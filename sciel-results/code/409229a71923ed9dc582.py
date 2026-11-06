import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # 1. Determine valid vertices
    # a is valid if 2*a is not a perfect square
    valid_vertices = []
    for i in range(1, 257):
        s = 2 * i
        r = int(math.isqrt(s))
        if r * r != s:
            valid_vertices.append(i)
    
    n = len(valid_vertices)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 2. Build complement graph adjacency bitsets
    # adj[i] has bit j set if valid_vertices[i] + valid_vertices[j] is NOT a perfect square
    adj = [0] * n
    
    # Precompute squares up to 512
    squares = set()
    for k in range(2, 23): # 22^2 = 484, 23^2 = 529 > 512
        squares.add(k * k)
        
    for i in range(n):
        vi = valid_vertices[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = valid_vertices[j]
            if vi + vj not in squares:
                mask |= (1 << j)
        adj[i] = mask
        
    # 3. Maximum Clique Search on Complement Graph
    # We use a branch and bound algorithm with greedy coloring.
    
    best_clique = []
    best_size = 0
    
    # Global variables for recursion (using list for mutability or nonlocal)
    # To avoid overhead, we might use iterative or optimized recursive
    
    def greedy_coloring(P):
        # Returns list of (color, vertex) sorted by color descending
        # P is a bitmask of candidate vertices
        # We perform greedy coloring on the subgraph induced by P in the complement graph
        # Note: We are finding a Clique in Complement Graph.
        # The bound for clique size in G is the chromatic number of G.
        # So we color the graph G (which is the complement of the original conflict graph, i.e., the "compatibility" graph).
        # Wait.
        # Original Problem: Max Independent Set in Conflict Graph G_conflict.
        # G_conflict has edge if sum is square.
        # MIS in G_conflict is Max Clique in G_complement.
        # G_complement has edge if sum is NOT square.
        # So we are finding Max Clique in G_complement.
        # The upper bound for a clique in a graph H is the chromatic number of H.
        # So we need to color G_complement[P].
        
        colors = {}
        remaining = P
        color_id = 0
        # We need an ordering. A common heuristic is to order by degree in the current subgraph.
        # For simplicity and speed, we can just iterate through bits and assign colors.
        # To get a good bound, we should process vertices in a way that colors are assigned tightly.
        # Standard Tomita: Order by degree, then color.
        
        # Let's extract vertices from P
        verts = []
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            verts.append(v)
            temp_P ^= lsb
            
        # Order by degree in G_complement[P]
        # deg(v) = popcount(adj[v] & P)
        # Python 3.10+ has int.bit_count()
        
        def get_deg(v):
            return (adj[v] & P).bit_count()
            
        verts.sort(key=get_deg, reverse=True)
        
        # Greedy coloring
        # Assign smallest available color
        color_map = {}
        max_color = 0
        for v in verts:
            # Find colors of neighbors in G_complement[P]
            # Neighbor u is in P and (adj[v] >> u) & 1
            # We want to avoid colors used by neighbors.
            used_colors = set()
            neighbors_mask = adj[v] & P
            # Iterate neighbors
            temp_N = neighbors_mask
            while temp_N:
                lsb = temp_N & -temp_N
                u = lsb.bit_length() - 1
                if u in color_map:
                    used_colors.add(color_map[u])
                temp_N ^= lsb
                
            # Find smallest color not in used_colors
            c = 0
            while c in used_colors:
                c += 1
            color_map[v] = c
            if c > max_color:
                max_color = c
                
        # Create list of (color, vertex)
        colored_verts = [(color_map[v], v) for v in verts]
        # Sort by color descending to try high color vertices first (tighter bound pruning)
        colored_verts.sort(key=lambda x: x[0], reverse=True)
        return colored_verts, max_color

    def expand(P, clique):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique.copy()
            return
            
        # Get colored vertices and bound
        colored_verts, bound = greedy_coloring(P)
        
        # If current clique size + bound <= best_size, prune
        if len(clique) + bound <= best_size:
            return
            
        # Iterate through vertices in order of color (descending)
        for color, v in colored_verts:
            if time.time() > deadline:
                return
                
            # Pruning check: if even taking this vertex and all remaining higher/equal color vertices
            # doesn't beat best, stop.
            # The bound for the subgraph containing v and subsequent vertices is 'color' (since we sorted by color)
            # Actually, the max color in the remaining set is 'color'.
            if len(clique) + color <= best_size:
                break
                
            # Branch on v
            new_clique = clique + [v]
            # New candidate set: neighbors of v in G_complement[P]
            new_P = P & adj[v]
            
            # Remove v from P for the next iteration in the loop
            P &= ~(1 << v)
            
            expand(new_P, new_clique)

    # Initial call
    # We can start with an empty set, P = all vertices
    initial_P = (1 << n) - 1
    expand(initial_P, [])
    
    # Convert best_clique (indices) back to values
    result = [valid_vertices[i] for i in best_clique]
    
    # If no result found (should not happen), return seed
    if not result:
        return list(range(1, 257, 3))
        
    return result
