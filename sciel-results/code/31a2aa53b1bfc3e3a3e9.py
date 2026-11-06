import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    N = 512
    limit = 512
    
    # Precompute adjacency bitsets
    # Bit i represents node i+1
    adj = [0] * (N + 1)
    for u in range(1, N + 1):
        mask = 0
        for v in range(1, N + 1):
            if u == v:
                continue
            g = math.gcd(u, v)
            l = u * v // g
            if l <= limit:
                mask |= (1 << (v - 1))
        adj[u] = mask
        
    # Tomita's Maximum Clique Algorithm
    # We use a greedy coloring heuristic to bound the search.
    
    best_clique = []
    best_size = 0
    
    # Order vertices by degree (descending) to help coloring and search
    # Degrees: popcount of adj[u]
    degrees = [0] * (N + 1)
    for u in range(1, N + 1):
        degrees[u] = adj[u].bit_count()
    
    # Sort vertices 1..N by degree descending
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    # We will operate on the ordered list of vertices.
    # But the bitsets are based on original indices.
    # It is easier to just use the original indices and a permutation array if needed,
    # or just iterate through the sorted list in the coloring step.
    
    # Let's implement the expand function.
    # P is a bitmask of candidates.
    # C is the current clique (list of vertices).
    
    def color_sort(P):
        """
        Greedy coloring of the subgraph induced by P.
        Returns a list of (vertex, color) sorted by color descending.
        The color of a vertex is an upper bound on the clique size
        that can be formed including that vertex and vertices of higher colors.
        """
        U = P
        colors = {}
        color_count = 0
        # Greedy coloring: assign smallest possible color
        # We want to order vertices so that vertices with higher color bounds are considered first.
        # Standard Tomita: color vertices one by one.
        
        # To make it fast, we can just iterate through bits in P.
        # However, greedy coloring requires checking neighbors in the *remaining* set U.
        
        # A simpler bound: just use the degree in P? No, coloring is tighter.
        # Let's do a simple greedy coloring.
        
        # We need to map bit positions to vertex indices.
        # It's easier to just get the list of vertices in P.
        verts_in_P = []
        temp_P = P
        while temp_P:
            # Get lowest set bit
            lsb = temp_P & -temp_P
            idx = lsb.bit_length() - 1
            v = idx + 1
            verts_in_P.append(v)
            temp_P ^= lsb
            
        # Greedy coloring
        # We assign colors 1, 2, ...
        # For each vertex, assign the smallest color not used by its neighbors in U.
        # This is O(N^2) per call, which might be slow if called many times.
        # But N=512 is small.
        
        # Optimization: We can just use the number of uncolored neighbors as a loose bound?
        # No, let's stick to proper greedy coloring for better pruning.
        
        # Actually, a very fast bound is just the size of P, but that's weak.
        # Let's try a simple sequential coloring.
        
        color_of = {}
        U_set = set(verts_in_P)
        color_seq = 0
        colored = []
        
        # To speed up, we can just sort by degree in P and color?
        # Let's just iterate.
        
        # Better approach for Tomita:
        # 1. Initialize U = P
        # 2. While U is not empty:
        #    - Pick a vertex v from U (e.g., highest degree in U)
        #    - Assign current color to v
        #    - Remove v and its neighbors from U
        #    - Increment color
        # This produces a valid coloring where each color class is an independent set.
        # The number of colors is an upper bound on the clique size.
        
        U_bits = P
        colors_list = [] # list of (vertex, color)
        current_color = 0
        
        while U_bits:
            current_color += 1
            # Pick a vertex from U_bits
            # Let's pick the one with highest degree in U_bits to make colors tighter?
            # Or just the first one? First one is faster.
            lsb = U_bits & -U_bits
            v = lsb.bit_length()
            # v is 1-based index?
            # bit 0 is node 1. lsb.bit_length() gives position+1.
            # So if lsb is 1 (bit 0), bit_length is 1. Node 1.
            # If lsb is 2 (bit 1), bit_length is 2. Node 2.
            # So node = lsb.bit_length()
            
            colors_list.append((v, current_color))
            
            # Remove v and its neighbors from U_bits
            # Neighbors of v in U_bits: adj[v] & U_bits
            # We remove v itself and all neighbors.
            # So we keep U_bits & ~(adj[v] | (1 << (v-1)))
            # Wait, adj[v] contains neighbors. v is not in adj[v].
            # So remove v: U_bits ^= (1 << (v-1))
            # Then remove neighbors: U_bits &= ~adj[v]
            
            U_bits &= ~(adj[v] | (1 << (v-1)))
            
            # Check time
            if time.time() > deadline:
                return None, None # Signal timeout
            
        # Sort by color descending
        colors_list.sort(key=lambda x: x[1], reverse=True)
        return colors_list

    def expand(C, P):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(C) > best_size:
                best_size = len(C)
                best_clique = C.copy()
            return
            
        # Get colored vertices
        res = color_sort(P)
        if res is None:
            return
        colors_list = res
        
        # Iterate in reverse order of colors
        # We need to remove vertices from P as we go
        # The colors_list gives us an order.
        # But we must ensure we only consider vertices still in P.
        # Since color_sort was called on P, all vertices in colors_list are in P.
        
        # We will iterate through the list.
        # For each vertex v, if len(C) + color[v] <= best_size, we can stop.
        # Because color[v] is the max clique size possible in the subgraph of vertices with color >= color[v].
        # Wait, the bound is: len(C) + (max color in remaining set)
        # In Tomita, the colors are assigned such that vertices with higher colors are processed first.
        # The color of a vertex is an upper bound on the clique size that can be formed
        # from that vertex and any vertices with higher colors.
        # So if we process vertices in descending color order, and we find that
        # len(C) + color[current_vertex] <= best_size, then no subsequent vertex
        # (which has lower or equal color) can lead to a better solution.
        
        for v, col in colors_list:
            # Check if v is still in P (it should be, unless we modified P in a way that affects this list? 
            # No, we are just iterating the list. But we need to remove v from P before recursing.)
            
            if len(C) + col <= best_size:
                break
                
            # Recurse
            C.append(v)
            # New P is intersection of current P and neighbors of v
            # Note: P contains v. Neighbors of v do not contain v.
            # So P & adj[v] automatically excludes v.
            new_P = P & adj[v]
            expand(C, new_P)
            C.pop()
            
            # Remove v from P for the next iteration
            P &= ~(1 << (v-1))
            
            if time.time() > deadline:
                return

    # Start with empty clique and all vertices
    initial_P = (1 << N) - 1
    expand([], initial_P)
    
    # Fallback if timeout or empty (should not be empty)
    if not best_clique:
        # Return seed
        return list(range(1, 23))
        
    return best_clique
