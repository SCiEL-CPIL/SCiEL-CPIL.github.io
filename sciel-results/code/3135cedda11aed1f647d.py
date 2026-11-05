import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s limit
    
    N = 256
    # Vertices are 0..255 representing numbers 1..256
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build conflict graph using bitsets
    # conflict[i] is a bitmask where bit j is 1 if vertex i conflicts with vertex j
    # Note: If 2*(i+1) is a square, vertex i conflicts with itself.
    # In MIS, self-loops mean the vertex cannot be selected.
    # We handle this by removing self-conflicting vertices from the candidate pool entirely,
    # or by treating them as already "removed" in the B&B logic.
    # Actually, standard MIS assumes simple graph. If a vertex has a self-loop, it cannot be in any independent set.
    # So we can simply exclude them from the graph.
    
    valid_vertices = []
    vertex_to_index = {}
    for i in range(N):
        a = i + 1
        if 2 * a in squares:
            continue # Self-conflict, cannot be in set
        valid_vertices.append(i)
        vertex_to_index[i] = len(valid_vertices)
    
    M = len(valid_vertices)
    
    # Build adjacency bitmasks for the reduced graph
    # adj[i] is a Python int representing a bitmask of neighbors of vertex i in the reduced graph
    adj = [0] * M
    
    for idx1, v1 in enumerate(valid_vertices):
        a = v1 + 1
        # Find all v2 such that a + (v2+1) is a square
        # v2 = s - a - 1
        # We need to check if v2 is in valid_vertices and v2 != v1 (self-loop already handled)
        neighbors_mask = 0
        for s in squares:
            b = s - a
            if 1 <= b <= N:
                v2 = b - 1
                if v2 in vertex_to_index:
                    idx2 = vertex_to_index[v2]
                    if idx1 != idx2:
                        neighbors_mask |= (1 << idx2)
        adj[idx1] = neighbors_mask
    
    # Tomita's Algorithm for Maximum Independent Set
    # State: P (set of available vertices), X (set of vertices that conflict with P, used for coloring bound)
    # We use bitmasks for P and X.
    
    best_set = 0
    best_size = 0
    
    def greedy_coloring(P):
        """
        Greedy coloring of graph induced by P.
        Returns list of color classes (masks) and the number of colors (upper bound on MIS size).
        """
        colors = []
        uncolored = P
        while uncolored:
            # Pick a vertex from uncolored (e.g., lowest bit)
            # To make it deterministic and potentially faster, we can pick based on degree or just lowest
            v_bit = uncolored & (-uncolored)
            v_idx = v_bit.bit_length() - 1
            
            # Find all vertices in uncolored that do NOT conflict with v
            # Non-neighbors of v in P
            non_neighbors = uncolored & ~adj[v_idx] & ~v_bit
            
            # Create a new color class containing v and some non-neighbors?
            # Standard greedy coloring:
            # 1. Start with empty color class C = {v}
            # 2. While there are uncolored vertices not adjacent to v and not in C, add one?
            # Actually, the standard Tomita coloring is:
            # Iterate through vertices in P in some order.
            # Assign smallest possible color.
            # Here, we construct color classes one by one.
            # Class k consists of vertices that are independent.
            # A simple greedy approach:
            # C = {v}
            # available = uncolored \ C
            # While available is not empty:
            #   Pick u from available such that u is not adjacent to any vertex in C
            #   If such u exists, add to C, remove from available
            #   Else break
            
            C = v_bit
            remaining = uncolored & ~C
            while remaining:
                # Find any vertex in remaining that is not adjacent to any vertex in C
                # A vertex u is compatible with C if (adj[u] & C) == 0
                # We need to find u in remaining such that (adj[u] & C) == 0
                
                # This can be slow if done naively.
                # Optimization: Maintain a mask of vertices in 'remaining' that are compatible with current C.
                # Initially, compatible = remaining
                # When we add u to C, we must remove from 'compatible' any vertex w that is adjacent to u.
                # i.e., compatible &= ~adj[u]
                
                # Let's re-structure:
                pass
            
            # Let's implement a standard sequential greedy coloring which is often faster to code correctly
            # but might be slower per step.
            # Alternative: The "max clique" style coloring used in Tomita's MIS.
            # We color the graph G[P].
            # The color number is an upper bound on the size of MIS in G[P].
            
            # Let's stick to a simpler, robust greedy coloring:
            # Order vertices by index (or degree).
            # For each vertex, assign the smallest color not used by its neighbors.
            
            # However, for the bound, we just need *a* coloring.
            # Let's use the "sequential" method on the bitset.
            
            # Actually, the most efficient bound in Tomita's algorithm uses a specific greedy coloring
            # that processes vertices in a specific order (often by degree or just sequential)
            # and assigns colors. The number of colors is the bound.
            
            # Let's implement the coloring that returns the number of colors directly.
            # We will use a list of color classes.
            # To speed up, we can use the fact that if we process vertices in order,
            # we can maintain for each color the mask of vertices that conflict with it.
            
            # Let's try a different, very fast bound:
            # The size of the maximum independent set is at most the number of colors in any proper coloring.
            # We can compute a greedy coloring.
            
            # Implementation:
            # colors: list of masks
            # For each vertex v in P (in some order):
            #    find first color c such that v is not adjacent to any vertex in color c.
            #    if none, create new color.
            
            # To make it fast, we iterate through P.
            # P is a bitmask.
            
            color_classes = []
            # We need to iterate through vertices in P.
            # Let's just iterate through all M vertices and check if they are in P.
            # This is O(M * M) in worst case if M is large, but M <= 256.
            # 256 * 256 = 65536 ops per coloring. This is acceptable.
            
            # However, we can do better.
            # Let's use the "greedy coloring" that picks a vertex, adds it to a color class,
            # removes it and its neighbors from the candidate set for that color,
            # and repeats.
            
            # This is the "maximal independent set" approach for coloring?
            # No, that gives a partition into independent sets.
            
            # Let's use the standard sequential greedy coloring.
            # Order: natural order of indices.
            
            # To optimize, we precompute nothing, just loop.
            
            # Wait, there is a specific greedy coloring in Tomita's algorithm that is very fast.
            # It processes vertices in an order (often descending degree, but natural order works for bound)
            # and assigns the lowest available color.
            
            # Let's implement a simple sequential coloring.
            # We need to know for each color, which vertices are in it.
            # And for each vertex, which colors are forbidden by its neighbors.
            
            # Given the small size (M <= 256), a simple implementation might be fine.
            
            # But we are inside a recursive B&B, so it needs to be fast.
            
            # Let's try this:
            # 1. Get list of vertices in P.
            # 2. For each vertex v in that list:
            #    a. Determine which colors are used by neighbors of v.
            #    b. Assign smallest unused color.
            
            # To avoid building the list every time, we can iterate bits.
            
            # Let's try a different bound:
            # Upper bound = size of P? No, too loose.
            # Upper bound = number of colors.
            
            # Let's implement the coloring carefully.
            
            # Actually, there is a simpler bound often used:
            # If we have a maximal independent set found greedily, its size is a LOWER bound.
            # For UPPER bound, we can use the greedy coloring.
            
            # Let's implement greedy coloring returning the number of colors.
            
            # We'll use an array `color_of` of size M, initialized to -1.
            # But we only care about vertices in P.
            
            # To speed up, we can use bit operations.
            
            # Let's use the following heuristic:
            # The number of colors is at least the size of the maximum clique.
            # But computing max clique is hard.
            
            # Let's just do sequential greedy coloring.
            
            # Optimization: Process vertices in P in order of increasing index.
            
            # We maintain `used_colors` for each vertex? No.
            # We maintain for each color `k`, a mask `color_mask[k]` of vertices in that color.
            # When considering vertex `v`, we check if `v` is adjacent to any vertex in `color_mask[k]`.
            # i.e., `(adj[v] & color_mask[k]) == 0`.
            
            # This is fast enough if we limit the number of colors.
            
            # Let's implement it.
            
            color_masks = []
            
            # Iterate through vertices in P
            # We can do this by iterating through all M vertices and checking if in P.
            # Or by extracting bits.
            
            # Let's iterate through bits of P.
            temp_p = P
            while temp_p:
                v_bit = temp_p & (-temp_p)
                v_idx = v_bit.bit_length() - 1
                temp_p ^= v_bit
                
                # Find smallest color k such that v_idx is not adjacent to any vertex in color k
                assigned = False
                for k in range(len(color_masks)):
                    if (adj[v_idx] & color_masks[k]) == 0:
                        color_masks[k] |= v_bit
                        assigned = True
                        break
                
                if not assigned:
                    color_masks.append(v_bit)
            
            return len(color_masks)
    
    def branch_and_bound(P, X, current_size):
        """
        P: bitmask of available vertices
        X: bitmask of vertices that conflict with P (not strictly needed for standard Tomita, 
           but used in some variants for symmetry breaking or specific bounds. 
           In standard Tomita MIS, we just use P and the bound on P).
        current_size: size of the independent set chosen so far
        """
        nonlocal best_set, best_size
        
        if P == 0:
            if current_size > best_size:
                best_size = current_size
                # We need to recover the set. 
                # Since we are using bitmasks, we can track the path or just store the mask.
                # Storing the mask is easier.
                # But we need the final set of vertices.
                # Let's store the best_mask instead of best_set.
                # We'll return the mask.
                # Actually, let's just update a global best_mask.
                pass
            return
        
        # Pruning: if current_size + popcount(P) <= best_size, return
        if current_size + P.bit_count() <= best_size:
            return
        
        # Bound: Greedy coloring of G[P]
        # The number of colors is an upper bound on the size of MIS in G[P]
        bound = greedy_coloring(P)
        if current_size + bound <= best_size:
            return
        
        # Time check
        if time.time() > DEADLINE:
            return
        
        # Choose a vertex v from P
        # Heuristic: Choose vertex with maximum degree in G[P] or minimum?
        # Tomita's algorithm often picks a vertex with high degree to reduce P quickly.
        # Or simply the first one.
        # Let's pick the vertex with the highest degree in the subgraph P.
        # To find degree in P: (adj[v] & P).bit_count()
        
        # Finding max degree vertex in P can be expensive if done naively for all v in P.
        # But M is small (<=256).
        
        best_v = -1
        max_deg = -1
        temp_p = P
        while temp_p:
            v_bit = temp_p & (-temp_p)
            v_idx = v_bit.bit_length() - 1
            temp_p ^= v_bit
            
            deg = (adj[v_idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v_idx
        
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include v
        # New P = P & ~adj[v] & ~v_bit
        new_P = P & ~adj[best_v] & ~v_bit
        branch_and_bound(new_P, X, current_size + 1)
        
        # Branch 2: Exclude v
        # New P = P & ~v_bit
        # Note: X is not updated in standard Tomita for MIS, it's for Max Clique.
        # For MIS, we just remove v from P.
        branch_and_bound(P & ~v_bit, X, current_size)
        
    # We need to track the best mask.
    # Let's modify the logic to store best_mask.
    
    global_best_mask = 0
    global_best_size = 0
    
    def bbb(P, current_size, current_mask):
        nonlocal global_best_mask, global_best_size
        
        if P == 0:
            if current_size > global_best_size:
                global_best_size = current_size
                global_best_mask = current_mask
            return
        
        if current_size + P.bit_count() <= global_best_size:
            return
        
        # Bound
        # We can use a simple bound: P.bit_count()
        # But coloring is better.
        # Let's use coloring.
        
        # To save time, we might skip coloring if P is small or if time is tight.
        # For now, use coloring.
        
        # Optimization: If P.bit_count() + current_size <= global_best_size, prune.
        # We already did that.
        
        # Let's use the greedy coloring bound.
        # If it's too slow, we can fall back to P.bit_count()
        
        # Let's try to be efficient.
        # We'll use the coloring function defined above.
        
        bound = greedy_coloring(P)
        if current_size + bound <= global_best_size:
            return
        
        if time.time() > DEADLINE:
            return
        
        # Select vertex
        best_v = -1
        max_deg = -1
        temp_p = P
        while temp_p:
            v_bit = temp_p & (-temp_p)
            v_idx = v_bit.bit_length() - 1
            temp_p ^= v_bit
            
            deg = (adj[v_idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v_idx
        
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Include v
        new_P = P & ~adj[best_v] & ~v_bit
        bbb(new_P, current_size + 1, current_mask | v_bit)
        
        # Exclude v
        bbb(P & ~v_bit, current_size, current_mask)
        
    # Initial call
    # P is all valid vertices
    initial_P = (1 << M) - 1
    bbb(initial_P, 0, 0)
    
    # Recover the set
    result_indices = []
    mask = global_best_mask
    while mask:
        v_bit = mask & (-mask)
        v_idx = v_bit.bit_length() - 1
        mask ^= v_bit
        result_indices.append(valid_vertices[v_idx])
    
    # Convert to 1-based values
    result = sorted(v + 1 for v in result_indices)
    
    # Fallback if empty (should not happen as M>0)
    if not result:
        result = [1]
        
    # Verification
    squares_set = squares
    for i in range(len(result)):
        for j in range(i, len(result)):
            if result[i] + result[j] in squares_set:
                # Invalid, fallback
                return [1]
                
    return result

