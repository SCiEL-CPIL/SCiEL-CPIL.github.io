import time
import sys

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Map edges to indices
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # Build conflict graph
    # conflict_mask[e] is a bitmask of edges conflicting with e
    conflict_mask = [0] * n_edges
    
    # Generate faces and update conflict masks
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_edges = [e1, e2, e3, e4]
                
                # Each pair in face_edges conflicts
                for k in range(4):
                    for l in range(k + 1, 4):
                        ek = face_edges[k]
                        el = face_edges[l]
                        conflict_mask[ek] |= (1 << el)
                        conflict_mask[el] |= (1 << ek)

    # Precompute bit masks for each edge
    # bit_mask[e] = 1 << e
    
    # MIS Branch and Bound
    # We want to find the largest independent set.
    
    best_size = 0
    best_set_mask = 0
    
    # Greedy coloring to get an upper bound on the size of an independent set
    # in the subgraph induced by 'candidates'
    def greedy_color_bound(candidates):
        # Returns the number of colors used in a greedy coloring of the subgraph
        # This is an upper bound on the independence number.
        colors = {}
        uncolored = candidates
        color_count = 0
        
        while uncolored:
            color_count += 1
            # Pick a vertex from uncolored
            # Heuristic: pick vertex with max degree in uncolored?
            # For speed, just pick the lowest set bit
            v = (uncolored & -uncolored).bit_length() - 1
            v_bit = 1 << v
            
            # Assign this color to v and all vertices in uncolored that do NOT conflict with v
            # Wait, in a coloring of the conflict graph, adjacent vertices must have different colors.
            # So if we assign color C to v, we can assign color C to any u in uncolored such that u is NOT adjacent to v.
            # The set of vertices that CAN have color C is {v} union {u in uncolored | u not in conflict_mask[v]}
            
            compatible = uncolored & ~conflict_mask[v]
            # Assign color to all in compatible? 
            # In standard greedy coloring for MIS bound, we assign the color to a maximal independent set in the remaining graph.
            # So, we pick v, then we pick as many non-conflicting vertices as possible to give them this color.
            
            current_color_set = 0
            temp_uncolored = uncolored
            # Greedily add vertices to current_color_set
            while temp_uncolored:
                # Pick a vertex
                # To be fast, pick lowest bit
                u = (temp_uncolored & -temp_uncolored).bit_length() - 1
                u_bit = 1 << u
                
                # Check if u is compatible with all vertices already in current_color_set
                # Since we are building an independent set for this color class,
                # u must not conflict with any vertex in current_color_set.
                # However, if we just pick them one by one and remove their neighbors, it's simpler.
                
                # Actually, the standard Tomita algorithm colors the subgraph.
                # Let's stick to a simpler bound: 
                # The independence number is at most the number of colors in a greedy coloring.
                # Greedy coloring: 
                # 1. Order vertices (e.g., by degree).
                # 2. Assign smallest possible color.
                
                # This might be too slow inside the B&B loop if done from scratch every time.
                # Let's use a simpler bound: 
                # Upper bound = popcount(candidates). This is weak.
                # Better bound: Clique cover? No.
                
                # Let's try a simple heuristic bound:
                # If we pick a vertex v, the max independent set size is at most 1 + alpha(candidates \ N[v]).
                # We can use the fact that the graph is sparse?
                
                # For 448 nodes, a recursive B&B with bitsets might work if we prune well.
                # Let's implement a basic Tomita-like coloring.
                
                # To make it fast, we will approximate the bound.
                # Bound: popcount(candidates) is safe but weak.
                # Let's try: 
                # Sort vertices in candidates by degree (descending).
                # This is expensive to compute every call.
                
                # Alternative: Use the fact that we are looking for a very large independent set (~300 out of 448).
                # This means we are looking for a very small vertex cover in the complement?
                # No.
                
                # Let's optimize the B&B structure itself.
                # We will use the "include/exclude" strategy.
                
                pass # Placeholder for logic below
                
            # Remove current_color_set from uncolored
            uncolored = uncolored & ~current_color_set
            
        return color_count

    # The above greedy_color_bound implementation was getting complex and slow.
    # Let's use a different strategy:
    # Since we know the answer is around 304, and total edges 448,
    # we can search for the MINIMUM number of edges to REMOVE (hitting set).
    # This is what the original code did.
    # BUT the directive says: "Model the problem as a maximum independent set... solve it using a branch-and-bound".
    
    # Okay, I will implement a standard B&B for MIS.
    # To make it fast in Python, I will rely on the bitset operations being fast.
    
    # Heuristic for ordering:
    # We will maintain a list of candidate edges.
    # At each step, pick an edge e.
    # Branch 1: Include e. Remove e and all its neighbors from candidates.
    # Branch 2: Exclude e. Remove e from candidates.
    
    # To prune, we need an upper bound on the size of the independent set in the remaining candidates.
    # A good upper bound is the size of a maximum matching? No, that's for vertex cover.
    # For Independent Set, a simple bound is the number of vertices.
    # A tighter bound: The graph is the line graph of a bipartite graph? No, Q7 is bipartite, so its line graph is...
    # Actually, the conflict graph is the union of cliques of size 4.
    
    # Let's use a simple greedy bound:
    # Greedily select vertices that don't conflict, count them. This gives a LOWER bound on alpha.
    # We need an UPPER bound to prune.
    # Upper bound: 
    # 1. popcount(candidates)
    # 2. Sum of 1/degree? No.
    # 3. Coloring bound.
    
    # Let's implement a fast coloring bound.
    def get_color_bound(candidates):
        # Greedy coloring of the subgraph induced by candidates
        # Returns the number of colors
        uncolored = candidates
        colors = 0
        # To speed up, we can process vertices in a fixed order (e.g., index order)
        # This is valid for an upper bound.
        while uncolored:
            colors += 1
            # Find a maximal independent set in uncolored to assign this color
            # Greedily pick vertices
            mis = 0
            temp = uncolored
            while temp:
                # Pick lowest set bit
                v = (temp & -temp).bit_length() - 1
                v_bit = 1 << v
                mis |= v_bit
                # Remove v and its neighbors from temp
                # Neighbors in the conflict graph
                temp = temp & ~conflict_mask[v] & ~v_bit
            uncolored = uncolored & ~mis
        return colors

    # This coloring bound might be expensive if called too often.
    # We will call it at the root and at each node of the B&B tree.
    
    def solve(candidates, current_size):
        nonlocal best_size, best_set_mask
        
        # Pruning
        if current_size + get_color_bound(candidates) <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                # We need to reconstruct the set. 
                # For now, just store size. We can store the mask too.
                # To store the mask, we need to pass it or reconstruct.
                # Let's pass a 'selected_mask' or just return the mask.
                # Passing mask makes recursion heavier.
                # Let's store the path? No, too much memory.
                # Let's just find the best size first, then maybe run again to find the set?
                # Or, we can store the mask in best_set_mask.
                # But we don't have the mask here.
                # Let's modify solve to return the best mask found in this branch?
                # Or, we can just output the edges based on a heuristic if we can't track the mask easily?
                # No, we need the exact edges.
                
                # Let's change the signature to pass the selected mask.
                # This doubles the integer size but is fine.
                pass # Logic adjusted below

    # Revised Solve with Mask
    def solve(candidates, current_size, selected_mask):
        nonlocal best_size, best_set_mask
        
        # Pruning
        bound = get_color_bound(candidates)
        if current_size + bound <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = selected_mask
            return
            
        # Heuristic: Pick a vertex to branch on
        # Choose vertex with max degree in the subgraph induced by candidates?
        # Or just pick the lowest set bit for simplicity and speed (bit ops are fast).
        # Picking high degree might prune better.
        # Let's pick the vertex with the highest number of neighbors in 'candidates'.
        
        # To find max degree vertex quickly:
        # We can iterate through set bits.
        # 448 bits is small enough.
        
        max_deg = -1
        v = -1
        temp = candidates
        while temp:
            u = (temp & -temp).bit_length() - 1
            # Count neighbors of u in candidates
            neighbors = conflict_mask[u] & candidates
            deg = neighbors.bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
            temp &= temp - 1 # Clear lowest bit
            
        if v == -1:
            return
            
        v_bit = 1 << v
        
        # Branch 1: Include v
        # New candidates: remove v and all its neighbors
        new_candidates_1 = candidates & ~conflict_mask[v] & ~v_bit
        solve(new_candidates_1, current_size + 1, selected_mask | v_bit)
        
        # Branch 2: Exclude v
        # New candidates: remove v
        new_candidates_2 = candidates & ~v_bit
        solve(new_candidates_2, current_size, selected_mask)

    # Initial call
    all_edges_mask = (1 << n_edges) - 1
    start_time = time.time()
    
    # Run B&B
    # To prevent timeout, we can add a check inside solve?
    # Python recursion limit might be an issue? Depth is max 448. Default is 1000. Should be fine.
    
    solve(all_edges_mask, 0, 0)
    
    # If we didn't find anything (shouldn't happen), fallback
    if best_size == 0:
         return []

    # Construct result
    result = []
    for i in range(n_edges):
        if best_set_mask & (1 << i):
            u, v = edges[i]
            result.append([u, v])
            
    return result
