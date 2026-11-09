import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe within 60s limit
    
    N = 256
    # Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is 1 if (i+1, j+1) is an edge (i.e., (i+1)*(j+1) % ((i+1)+(j+1)) == 0)
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        adj[i] = mask

    best_set = []
    best_size = 0
    # Initial candidate set: all vertices
    all_vertices = (1 << N) - 1
    
    # Tomita's Branch and Bound for Maximum Independent Set
    # C: bitmask of candidates
    # S: current independent set (list)
    # bound: upper bound on size of MIS in C
    
    # We use a simple recursive approach with global variables for speed
    # Python recursion depth might be an issue for N=256 if not careful, but MIS depth is usually small relative to N.
    # However, we can use an iterative stack or increase recursion limit.
    sys.setrecursionlimit(3000)
    
    def greedy_color_bound(C):
        """
        Returns the greedy coloring bound for the subgraph induced by C.
        Also returns the coloring (list of color classes) for ordering.
        """
        colors = []
        remaining = C
        color_idx = 0
        while remaining:
            color_class = 0
            # Find maximal independent set in remaining for this color
            # Greedy: pick vertex with lowest degree in remaining? Or just first?
            # Standard Tomita uses a specific ordering.
            # For simplicity and speed, we can just pick vertices greedily.
            # A better heuristic: pick vertex with minimum degree in the current remaining set.
            # But bit operations make degree counting expensive.
            # Let's use a simple greedy: iterate through bits, if not adjacent to any in color_class, add it.
            # To make it effective, we should order vertices.
            # Let's stick to a standard greedy coloring on the induced subgraph.
            
            # We need to assign colors such that adjacent vertices have different colors.
            # The number of colors is the upper bound.
            
            # Efficient way:
            # 1. Extract vertices from remaining.
            # 2. For each vertex, check if it can be added to current color class.
            
            # To speed up, we can pre-compute or just iterate.
            # Since N=256, iterating is fine.
            
            # Let's use a list of vertices in remaining
            verts = []
            temp = remaining
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                verts.append(idx)
                temp ^= lsb
            
            # Greedy coloring:
            # Assign color 0 to as many non-adjacent vertices as possible.
            # Then color 1 to the rest, etc.
            
            # We'll do this by maintaining the 'remaining' for the current color.
            # Actually, the standard greedy coloring for bound is:
            # 1. Order vertices (e.g., by degree).
            # 2. Assign smallest possible color.
            
            # Let's implement a simple greedy that returns the number of colors.
            # This is often sufficient for pruning.
            
            # Alternative: Just count the max independent set size in the complement? No, that's the problem itself.
            # The bound is the chromatic number of the subgraph.
            
            # Simple greedy coloring:
            # color_map = {}
            # for v in verts:
            #    c = 0
            #    while any(u in color_class and adj[v] & (1<<u) for u in color_class):
            #        c += 1
            #    color_map[v] = c
            #    color_class.add(v)
            #    if c == max_color: max_color += 1
            
            # To be faster, we can use bitmasks for color classes.
            
            color_classes = []
            uncolored = remaining
            c_idx = 0
            while uncolored:
                # Pick a vertex from uncolored
                # Heuristic: pick vertex with max degree in uncolored subgraph?
                # For now, pick the first one (lowest index) to keep it simple and deterministic.
                lsb = uncolored & -uncolored
                v = lsb.bit_length() - 1
                uncolored ^= lsb
                
                # Try to fit v into existing color classes
                placed = False
                for i in range(c_idx):
                    # Check if v is adjacent to any vertex in color_classes[i]
                    # color_classes[i] is a bitmask
                    if adj[v] & color_classes[i]:
                        continue
                    # v can go into color i
                    color_classes[i] |= (1 << v)
                    placed = True
                    break
                
                if not placed:
                    color_classes.append(1 << v)
                    c_idx += 1
            
            return c_idx, color_classes

    def search(C, S, bound):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if C == 0:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
        
        # Prune if potential size doesn't beat best
        if len(S) + bound <= best_size:
            return
            
        # Get coloring bound and classes
        # We need to order vertices by color class to guide the search
        # Tomita's algorithm uses the coloring to order the vertices.
        # Vertices with higher colors are chosen first.
        
        # Let's get the coloring
        c_bound, color_classes = greedy_color_bound(C)
        
        if len(S) + c_bound <= best_size:
            return
            
        # Order vertices: group by color class, highest color first.
        # Within a color class, any order.
        ordered_verts = []
        # We need to map vertices to their color index
        # color_classes is a list of bitmasks.
        # Let's create a list of (color_idx, vertex)
        vert_color = {}
        for c_idx, mask in enumerate(color_classes):
            temp = mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                vert_color[v] = c_idx
                temp ^= lsb
                
        # Sort by color descending, then by index (or degree)
        # Vertices in C
        verts_in_C = []
        temp = C
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts_in_C.append(v)
            temp ^= lsb
            
        # Sort: higher color first
        verts_in_C.sort(key=lambda v: -vert_color[v])
        
        # Branch on the first vertex (highest color)
        # Actually, Tomita's algorithm branches on vertices in the order of the coloring.
        # It picks a vertex v, adds it to S, and removes N(v) and v from C.
        
        # We iterate through the ordered vertices.
        # For each vertex v in the order:
        # 1. If v is still in C (it might have been removed by previous branches? No, we branch recursively)
        # Standard B&B:
        # Pick v.
        # Recurse with C \ (N(v) U {v})
        # Then remove v from C and continue loop with next v.
        
        # Let's implement the loop carefully.
        # We need to maintain the set of candidates for the current level.
        # However, the standard Tomita implementation often picks the vertex with the highest color,
        # branches on it, and then in the "exclude" branch, removes it from C and continues.
        
        # To keep it simple and effective:
        # We pick the first vertex in the sorted list (highest color).
        # Let v = verts_in_C[0].
        
        # But we must ensure we explore all possibilities.
        # The standard pattern:
        # while C is not empty:
        #   v = select_vertex(C)
        #   C = C \ {v}
        #   search(C \ N(v), S U {v})
        #   # The loop continues with C not containing v, effectively excluding v
        
        # Let's use this pattern.
        
        # We need to re-color or use the existing bound for the remaining C?
        # The bound `c_bound` was for the original C.
        # As we remove vertices, the bound decreases.
        # We can just use `len(S) + c_bound` as a loose bound initially,
        # and re-calculate if we want tighter pruning, but that's expensive.
        # Given N=256, the initial bound is likely tight enough for the first few levels.
        
        # Let's stick to the simple loop.
        
        # To optimize, we can pass the color classes down?
        # For now, let's just use the simple recursive structure.
        
        # We need to be careful with the `bound` argument.
        # The `bound` passed in is the bound for the current C.
        # When we remove v, the new C is smaller, so the bound is <= old bound.
        # We can use `bound` for pruning in the recursive call?
        # Actually, the recursive call will calculate its own bound.
        
        # Let's refine the search function to use the ordered list properly.
        
        # We'll iterate through the vertices in `verts_in_C`.
        # We need to maintain the current set of candidates `C_curr`.
        # Initially C_curr = C.
        
        # However, the standard Tomita algorithm works on the set C.
        
        # Let's try a slightly different structure:
        # 1. Color C.
        # 2. Order vertices by color.
        # 3. Iterate through vertices in that order.
        # 4. If vertex is not in current C, skip.
        # 5. Branch: include v.
        # 6. Exclude v: remove v from C.
        
        # This is a bit complex to manage with bitmasks in a single loop without recursion for the "exclude" part?
        # No, the "exclude" part is just the next iteration of the loop.
        
        # Let's implement:
        
        # We need a mutable C for the loop?
        # Let's just pick the first vertex (highest color), branch, and then recurse on the remaining set with v removed?
        # No, that's not exhaustive.
        
        # Correct Tomita Loop:
        # def search(C, S):
        #   if C empty: update best
        #   bound = color_bound(C)
        #   if len(S) + bound <= best: return
        #   
        #   # Order vertices
        #   # Let's just pick v = argmax color
        #   # Actually, we can just pick the first vertex in the sorted list that is in C.
        
        #   # To be efficient, we can use the color classes to guide the order.
        #   # Let's use the simple approach:
        #   # Pick v from C (e.g., highest color).
        #   # Recurse(C & ~N(v) & ~(1<<v), S + [v])
        #   # Remove v from C: C = C & ~(1<<v)
        #   # Continue loop? 
        #   # If we just recurse and then return, we miss the case where v is excluded.
        #   # So we must loop.
        
        #   # Pseudocode:
        #   # while C:
        #   #   v = select(C)
        #   #   C = C \ {v}
        #   #   search(C \ N(v), S + [v])
        #   #   # Note: C is already updated to exclude v, so the next iteration handles the exclusion of v (and subsequent removals)
        
        #   # This works!
        
        # Let's implement this.
        
        # We need to select v.
        # We can use the coloring to select the vertex with the highest color.
        # If we have the color classes, we can just pick the first vertex from the last color class that is in C.
        
        # Let's re-calculate coloring inside the loop? No, that's expensive.
        # We calculate it once at the start of `search`.
        # But as we remove vertices, the coloring bound becomes loose.
        # However, for N=256, it might be fine.
        
        # Let's try to optimize `select`.
        # We have `verts_in_C` sorted by color descending.
        # We can just iterate through this list.
        
        # Wait, the `C` changes in the loop.
        # The `verts_in_C` list is static for the duration of the `search` call?
        # No, if we remove vertices, some in `verts_in_C` might not be in `C` anymore.
        # We can just check `if not (C & (1<<v)): continue`.
        
        # Let's rewrite `search` with this loop structure.
        
        def search_loop(C, S):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return
                
            if C == 0:
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                return
            
            # Calculate bound
            # To save time, we can use a simpler bound if N is small, but coloring is better.
            # Let's use the coloring bound.
            c_bound, color_classes = greedy_color_bound(C)
            
            if len(S) + c_bound <= best_size:
                return
            
            # Prepare ordered vertices
            # We need to map vertices to colors again?
            # The `greedy_color_bound` returned `color_classes`.
            # Let's extract vertices and their colors.
            
            vert_color = {}
            for c_idx, mask in enumerate(color_classes):
                temp = mask
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    vert_color[v] = c_idx
                    temp ^= lsb
                    
            # Get list of vertices in C
            verts_in_C = []
            temp = C
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                verts_in_C.append(v)
                temp ^= lsb
                
            # Sort by color descending
            verts_in_C.sort(key=lambda v: -vert_color[v])
            
            # Loop
            for v in verts_in_C:
                if time.time() > deadline:
                    return
                
                # Check if v is still in C
                if not (C & (1 << v)):
                    continue
                    
                # Exclude v from C for subsequent iterations
                C = C & ~(1 << v)
                
                # Include v
                new_S = S + [v]
                # New C: remove v and its neighbors
                # N(v) is adj[v]
                # We want C \ (N(v) U {v})
                # C is already excluding v.
                # So new_C = C & ~adj[v]
                new_C = C & ~adj[v]
                
                search_loop(new_C, new_S)
                
            # End for loop
            
        # Start search
        # Initial C is all vertices
        search_loop(all_vertices, [])
        
        # If no set found (should not happen), return empty or seed
        if not best_set:
            return list(range(1, 257, 2))
            
        # Convert indices back to values (0-based index -> 1-based value)
        return [x + 1 for x in best_set]

# EVOLVE-BLOCK-END
