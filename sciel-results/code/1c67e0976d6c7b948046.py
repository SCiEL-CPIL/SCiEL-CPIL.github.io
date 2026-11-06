import time
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    # Leave a safety margin before the 90s kill. 85s is a good target.
    deadline = start + 85.0
    n = 256

    # 1. Precompute perfect squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1

    # 2. Identify valid nodes (2*a is not a perfect square)
    valid_nodes = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
    
    # Map node value to index for internal representation
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
    m = len(valid_nodes)
    
    if m == 0:
        return [1] # Fallback, though 1 is valid (2!=square)

    # 3. Build adjacency bitmasks
    # adj_bits[i] is an integer where bit j is 1 if valid_nodes[i] and valid_nodes[j] conflict
    adj_bits = [0] * m
    for i, a in enumerate(valid_nodes):
        for j in range(i + 1, m):
            b = valid_nodes[j]
            if a + b in squares:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)

    # 4. Greedy Initial Solution
    # We want a maximum independent set.
    # Greedy: Pick vertex with min degree, add to set, remove it and its neighbors.
    remaining_mask = (1 << m) - 1
    selected_mask = 0
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')

    # Greedy Min-Degree
    while remaining_mask:
        best_v = -1
        best_deg = m + 1
        # Iterate through remaining vertices to find min degree in remaining subgraph
        # Optimization: We can just iterate all and check if in remaining
        for i in range(m):
            if (remaining_mask >> i) & 1:
                # Degree of i in remaining subgraph
                deg = popcount(adj_bits[i] & remaining_mask)
                if deg < best_deg:
                    best_deg = deg
                    best_v = i
        
        if best_v == -1:
            break
        
        selected_mask |= (1 << best_v)
        # Remove best_v and its neighbors from remaining
        remaining_mask &= ~(1 << best_v)
        remaining_mask &= ~adj_bits[best_v]

    best_size = popcount(selected_mask)
    best_solution = selected_mask

    # 5. Local Search: Greedy Add
    # Try to add any vertex that doesn't conflict with current selected
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(m):
            if (best_solution >> i) & 1:
                continue
            # Check if i conflicts with any in best_solution
            if adj_bits[i] & best_solution:
                continue
            best_solution |= (1 << i)
            best_size += 1
            improved = True

    # 6. Branch and Bound for Maximum Independent Set
    # We use a standard B&B with a coloring bound.
    # State: available vertices (bitmask), current selected count.
    # Bound: Greedy coloring of the available subgraph. The max independent set
    # size is at most the number of colors in a greedy coloring (which is an
    # upper bound for the clique number, but for MIS we use the fact that
    # an independent set in G is a clique in complement G. 
    # Actually, a simpler bound: The maximum independent set size in a graph
    # is at most the size of the largest independent set in the remaining graph.
    # A common heuristic upper bound is the greedy coloring number of the
    # *complement* of the remaining graph? No.
    # Let's use the standard Tomita bound: 
    # For a graph G, the size of the MIS is <= the number of colors in a
    # greedy coloring of G. Wait, no. 
    # In a graph G, a clique in G is an independent set in Complement(G).
    # So we want the Max Clique in Complement(G).
    # Alternatively, we can just use a simple bound: 
    # If we have a set of vertices V, the max independent set size is at most
    # the number of vertices. That's weak.
    # Better: Greedy coloring of the *conflict graph* gives a lower bound on
    # the number of colors needed. This doesn't directly bound the MIS size.
    # 
    # Correct Bound for MIS:
    # The size of the Maximum Independent Set in G is equal to the size of the
    # Maximum Clique in the Complement of G.
    # So we can run a Max Clique B&B on the Complement Graph.
    # Let's build the complement adjacency bits.
    
    # Complement adjacency: c_adj[i] has bit j set if i and j are NOT connected in original
    # Note: In original, i is not connected to itself. In complement, i is connected to itself?
    # No, simple graph. 
    # Let's stick to MIS on original graph.
    # Upper bound for MIS(G): 
    # A common heuristic is to use a greedy coloring of G. Let k be the number of colors.
    # This is an upper bound for the *Chromatic Number*, not directly MIS.
    # However, if we color G, each color class is an independent set.
    # This doesn't help bound the *size* of the MIS directly from the color count.
    
    # Let's use the "Petersen" bound or similar?
    # Actually, a very effective B&B for MIS uses the following:
    # 1. Order vertices by degree.
    # 2. Use a greedy coloring of the *remaining* graph to get an upper bound.
    # Wait, the greedy coloring number of G is an upper bound on the size of the
    # *Maximum Clique* in the complement? No.
    
    # Let's switch to Max Clique on the Complement Graph.
    # Vertices: 0..m-1
    # Edge in Complement: (i, j) if i != j and NOT (adj_bits[i] has bit j)
    
    comp_adj = [0] * m
    for i in range(m):
        # All other vertices
        all_others = ((1 << m) - 1) ^ (1 << i)
        # Remove neighbors from original
        comp_adj[i] = all_others & ~adj_bits[i]

    # Max Clique B&B on comp_adj
    # We want to find the largest set of vertices that are pairwise connected in comp_adj.
    # This corresponds to a set of vertices that are pairwise NON-connected in adj_bits (original).
    # Which is exactly an Independent Set in the original graph.
    
    # Tomita's Algorithm for Max Clique
    # We maintain a list of candidates.
    # We use a greedy coloring of the candidate set to prune.
    # The color index of a vertex is an upper bound on the size of the clique
    # that can be formed from that vertex and its successors in the ordering.
    
    # To make it fast, we process vertices in a specific order.
    # Let's reorder vertices by degree in the complement graph (descending) to help coloring.
    # Or just use the original order and a standard coloring routine.
    
    # Given time constraints, a simple recursive B&B with a good initial bound
    # and greedy coloring pruning is best.
    
    # Greedy Coloring function for a set of vertices (bitmask)
    # Returns the number of colors used.
    def greedy_color_count(vertices_mask):
        if vertices_mask == 0:
            return 0
        colors = 0
        uncolored = vertices_mask
        while uncolored:
            colors += 1
            # Greedily pick a vertex for this color
            # Pick the one with highest degree in uncolored? Or just first?
            # For upper bound, any valid coloring works.
            # Let's pick the vertex with max degree in the subgraph induced by uncolored.
            best_v = -1
            best_deg = -1
            # Iterate
            v_iter = uncolored
            while v_iter:
                v = (v_iter & -v_iter).bit_length() - 1
                v_iter &= v_iter - 1
                deg = popcount(comp_adj[v] & uncolored)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            uncolored &= ~(1 << best_v)
            # Remove neighbors of best_v from uncolored for this color class
            # In clique search, we want to find a clique.
            # The coloring is on the graph we are searching for cliques in (Complement).
            # So we remove neighbors in the Complement graph.
            uncolored &= ~comp_adj[best_v]
            
        return colors

    # B&B State
    # best_size is global
    # current_clique_size
    # candidates_mask: vertices that can be added to the current clique
    
    # We use a global variable for best_size to allow pruning
    global_best_size = best_size
    global_best_clique_mask = best_solution # This is the MIS mask, which is the Clique mask in Comp
    
    # To speed up B&B, we should order vertices.
    # Let's sort vertices by degree in Complement (descending).
    # But we are using bitmasks based on indices 0..m-1.
    # Reordering indices is complex with bitmasks.
    # Instead, we can just rely on the coloring bound.
    
    def expand_clique(current_mask, candidates_mask, current_size):
        nonlocal global_best_size, global_best_clique_mask
        
        if time.time() > deadline - 2.0:
            return False # Timeout
        
        if current_size + popcount(candidates_mask) <= global_best_size:
            return True
        
        if candidates_mask == 0:
            if current_size > global_best_size:
                global_best_size = current_size
                global_best_clique_mask = current_mask
            return True
        
        # Greedy coloring bound
        # We need a tighter bound than just popcount.
        # The number of colors in a greedy coloring of the subgraph induced by candidates_mask
        # is an upper bound on the max clique size in that subgraph.
        
        # Optimization: If the bound is not tight, we might still prune.
        # Let's calculate the color bound.
        # To make coloring fast, we can use a simplified version.
        
        # Heuristic: Order vertices in candidates by degree in candidates subgraph.
        # But for simplicity and speed in Python, let's just use the greedy coloring
        # described above. It might be slow if called too often.
        
        # Alternative: Use a precomputed upper bound? No, dynamic.
        
        # Let's try a simpler pruning first:
        # If current_size + popcount(candidates_mask) <= global_best_size, prune.
        # We already did that.
        
        # Let's use the coloring bound.
        # To avoid expensive coloring every node, we can use it selectively or optimize it.
        
        # Let's implement a fast greedy coloring count.
        color_bound = greedy_color_count(candidates_mask)
        
        if current_size + color_bound <= global_best_size:
            return True
            
        # Select a vertex to branch on
        # Usually the last vertex in the coloring order or highest degree.
        # Let's pick the vertex with the highest color index in the greedy coloring?
        # Or just the first available one?
        # Standard Tomita: Use the ordering from the coloring.
        
        # For simplicity, let's pick the vertex with the highest degree in the candidate set.
        best_v = -1
        best_deg = -1
        c_iter = candidates_mask
        while c_iter:
            v = (c_iter & -c_iter).bit_length() - 1
            c_iter &= c_iter - 1
            deg = popcount(comp_adj[v] & candidates_mask)
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            return True
            
        # Branch 1: Include best_v
        new_candidates = candidates_mask & comp_adj[best_v]
        new_mask = current_mask | (1 << best_v)
        expand_clique(new_mask, new_candidates, current_size + 1)
        
        # Branch 2: Exclude best_v
        candidates_mask &= ~(1 << best_v)
        expand_clique(current_mask, candidates_mask, current_size)
        
        return True

    # Start B&B
    # Initial candidates: all vertices
    # Initial mask: 0
    # We want to find a clique larger than best_size.
    # So we start with empty clique.
    
    # To improve performance, we can start with the vertices from the greedy solution
    # as a "core" and try to expand? No, standard B&B is safer.
    
    # However, Python recursion might hit limits or be slow.
    # Let's add a recursion limit or use iterative if needed.
    # m is at most 256. Recursion depth is at most 256. Safe.
    
    # One optimization: If we find a clique of size global_best_size + 1, we update.
    
    # Let's run the B&B.
    # We pass the full set of vertices as candidates.
    expand_clique(0, (1 << m) - 1, 0)
    
    # Update best_solution if B&B found something better
    if global_best_size > best_size:
        best_size = global_best_size
        best_solution = global_best_clique_mask

    # Final Local Search Polish on the result
    # Try to add any vertex that fits
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in range(m):
            if (best_solution >> i) & 1:
                continue
            if adj_bits[i] & best_solution:
                continue
            best_solution |= (1 << i)
            best_size += 1
            improved = True

    # Convert bitmask back to list of values
    result = []
    for i in range(m):
        if (best_solution >> i) & 1:
            result.append(valid_nodes[i])
    
    # Verify
    if not result:
        return [1]
        
    square_set = squares
    valid = True
    for idx, a in enumerate(result):
        for b in result[idx:]:
            if a + b in square_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid set (e.g., numbers 1, 2, 4, 8... powers of 2? No, sums can be squares)
        # Just return the first few valid nodes
        return valid_nodes[:10]

    return result

# EVOLVE-BLOCK-END
