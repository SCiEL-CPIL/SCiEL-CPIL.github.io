import math
import time

def construct():
    N = 512
    limit = 512
    
    # Build adjacency bitmask for valid pairs (lcm(i, j) <= 512)
    # adj[i] is a bitmask of vertices j such that lcm(i, j) <= 512
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= limit:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
                
    # Pre-compute degree for initial sorting or just rely on coloring
    # We will use a standard Tomita-style algorithm with greedy coloring bound.
    
    best_size = 0
    best_set_mask = 0
    
    # Helper to perform greedy coloring on a set of vertices (bitmask)
    # Returns a list of (vertex_index, color) sorted by color, then by index (or degree)
    # The max color is the upper bound on clique size.
    def greedy_coloring(P_mask):
        # We want to assign colors to vertices in P_mask.
        # Standard greedy: 
        # 1. Order vertices (e.g., by degree descending in P, or just index).
        # 2. Assign smallest possible color.
        # 3. The result is a list of vertices sorted by color.
        
        # To make it fast, we extract vertices from P_mask.
        # For N=512, extracting bits is O(N) per node, which is acceptable if pruning is strong.
        
        # Optimization: We can pre-sort vertices globally by degree descending.
        # Then iterate through that global order, picking those in P_mask.
        # This usually yields better coloring bounds than arbitrary order.
        
        # Let's pre-compute a global order once outside.
        pass 

    # Pre-compute global degree order for better coloring
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = adj[i].bit_count()
        
    # Sort vertices 1..N by degree descending
    global_order = list(range(1, N + 1))
    global_order.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map vertex value to bit
    def val_to_bit(v):
        return 1 << (v - 1)
        
    # We need a way to get the list of vertices in P_mask sorted by global_order.
    # Since P_mask changes, we can't pre-compute this list for every mask.
    # Instead, inside expand, we construct the candidate list.
    
    def expand(P_mask, current_size, current_set_mask):
        nonlocal best_size, best_set_mask
        
        if current_size > best_size:
            best_size = current_size
            best_set_mask = current_set_mask
            
        if P_mask == 0:
            return
            
        # Greedy Coloring on P_mask
        # We process vertices in the order of global_order.
        # We maintain a set of "available" vertices (initially P_mask).
        # We assign colors 1, 2, ...
        # For each color k, we pick a maximal independent set from available vertices?
        # No, standard greedy coloring for clique bound:
        # 1. Initialize color = 0.
        # 2. While P_mask is not empty:
        #    color += 1
        #    U = P_mask
        #    While U is not empty:
        #       Pick a vertex v from U (e.g., first in global_order that is in U)
        #       Assign color 'color' to v.
        #       Remove v from U.
        #       Remove all neighbors of v from U (because they can't have the same color in an independent set? 
        #       Wait. The coloring is for the graph G. An independent set in G is a clique in G_bar?
        #       No. We are finding a CLIQUE in G (valid pairs).
        #       The bound comes from coloring the graph G.
        #       If G can be colored with k colors, the maximum clique size is <= k.
        #       Greedy coloring: 
        #       - Start with uncolored set P.
        #       - Color 1: Find a maximal independent set in P? No.
        #       - Standard Tomita: 
        #         - Order vertices.
        #         - Assign smallest color such that it differs from neighbors already colored.
        #         - This requires knowing neighbors' colors.
        
        # Let's implement the standard simple greedy coloring used in MCQ:
        # 1. Extract vertices from P_mask into a list, sorted by global_order.
        # 2. Initialize color array to 0.
        # 3. For each vertex v in the list:
        #      Determine the set of colors used by neighbors of v that are already in the list (and thus colored).
        #      Assign the smallest positive integer not in that set.
        # 4. The maximum color assigned is the bound.
        # 5. We also need to sort the vertices by their assigned color for the branching loop.
        
        # This O(N^2) per node might be too slow if the tree is deep.
        # However, with N=512 and strong pruning, it should be okay.
        
        # Optimization: Instead of full coloring, we can use a simpler bound if needed, 
        # but coloring is the key to "bitset-based clique cover lower bounds" (actually upper bounds on clique size).
        
        # Let's try a slightly faster coloring:
        # We process vertices in reverse global_order? Or forward?
        # Usually, high degree first helps.
        
        # Get list of vertices in P_mask
        # To avoid creating a list every time, we can iterate through global_order and check bit.
        # But we need to store colors.
        
        verts_in_P = []
        for v in global_order:
            if P_mask & val_to_bit(v):
                verts_in_P.append(v)
                
        if not verts_in_P:
            return
            
        # If the number of vertices is small, we can brute force? 
        # No, let's stick to coloring.
        
        # colors[v] for v in verts_in_P
        # We need a map or dict. Since N is small, an array of size N+1 is fine.
        colors = [0] * (N + 1)
        
        max_color = 0
        
        # We need to know which neighbors are already colored.
        # Since we process in a specific order, we can just check the colors of neighbors that are in verts_in_P.
        # To make this efficient, we can maintain a bitmask of "colored" vertices?
        # Or simply, for each v, check all its neighbors u in adj[v], and if u is in verts_in_P (and thus has a color), 
        # add that color to a forbidden set.
        
        # To check if u is in verts_in_P quickly, we can use a bitmask `colored_mask`?
        # Actually, verts_in_P is derived from P_mask. So if u is in P_mask, it will be processed.
        # But we only want neighbors that are *already processed* (i.e., appear earlier in our iteration order).
        
        # Let's iterate through verts_in_P.
        # We need to know if a neighbor u is "before" v in verts_in_P.
        # We can maintain a bitmask `processed_mask` of vertices already assigned a color in this step.
        
        processed_mask = 0
        
        for v in verts_in_P:
            v_bit = val_to_bit(v)
            
            # Find forbidden colors from neighbors in processed_mask
            # Neighbors of v are in adj[v].
            # Intersection with processed_mask gives neighbors already colored.
            neigh_in_processed = adj[v] & processed_mask
            
            # Get the set of colors used by these neighbors
            forbidden = set()
            # Iterate through bits in neigh_in_processed
            # Since degree is at most 512, this is fast.
            mask = neigh_in_processed
            while mask:
                lsb = mask & -mask
                u_idx = lsb.bit_length() - 1
                u_val = u_idx + 1
                forbidden.add(colors[u_val])
                mask ^= lsb
            
            # Assign smallest color not in forbidden
            c = 1
            while c in forbidden:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
            processed_mask |= v_bit
            
        # Upper bound check
        if current_size + max_color <= best_size:
            return
            
        # Now, we branch.
        # We should process vertices in order of decreasing color (or increasing? 
        # Tomita usually processes in the order of coloring, often lowest color first to fail fast?
        # Actually, to find a large clique, we want to pick vertices that are "central".
        # The standard MCQ algorithm sorts vertices by color (lowest first) and then by degree.
        # Let's sort verts_in_P by (color, -degree) or just color.
        # If we pick a vertex with low color, it means it has few colored neighbors, 
        # which doesn't directly imply high degree in the remaining graph.
        
        # Let's sort by color descending? 
        # If we pick a vertex with the highest color, it's "harder" to place, 
        # but might be part of a large clique?
        # Actually, the bound `current_size + max_color` is tight if we can find a clique of size `max_color`.
        # The standard approach: 
        # 1. Color the graph.
        # 2. Order vertices by color (increasing).
        # 3. For each vertex in this order:
        #    - If current_size + color[v] <= best_size, break (prune).
        #    - Recurse on N(v) \ {vertices processed so far}.
        
        # Let's create a sorted list of (v, color[v])
        sorted_verts = sorted(verts_in_P, key=lambda x: colors[x])
        
        # We need to maintain the set of "remaining" candidates as we iterate.
        # Let R = P_mask
        R_mask = P_mask
        
        for v in sorted_verts:
            v_bit = val_to_bit(v)
            
            # Prune if even with this vertex and all remaining vertices of this color or higher, we can't beat best
            # The color of v is an upper bound on the clique size in the subgraph induced by {v} U {vertices with color >= color[v] in the original ordering?}
            # Actually, the property is: The vertices with color <= c form a subgraph where the max clique is <= c.
            # So if we process in increasing color, and we are at vertex v with color c,
            # the remaining candidates (those not yet processed in this loop) have colors >= c? 
            # No, the remaining candidates are those in R_mask.
            # The bound for the subgraph induced by R_mask is the max color in R_mask.
            # However, a tighter bound is often used: 
            # If we pick v, the new candidate set is N(v) & R_mask_without_v.
            # The max clique in that set is <= max_color_in(N(v) & R_mask_without_v).
            
            # Standard Tomita:
            # We have a list of vertices sorted by color.
            # We iterate through them.
            # For the i-th vertex v_i, we know that the maximum clique in the subgraph induced by {v_i, v_{i+1}, ...} is <= color[v_i].
            # So if current_size + color[v_i] <= best_size, we can stop.
            
            if current_size + colors[v] <= best_size:
                break
                
            # Remove v from R_mask
            R_mask ^= v_bit
            
            # Recurse
            new_P = R_mask & adj[v]
            expand(new_P, current_size + 1, current_set_mask | v_bit)
            
    # Initial call
    all_vertices = (1 << N) - 1
    expand(all_vertices, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    if not result:
        return [1]
        
    return result
