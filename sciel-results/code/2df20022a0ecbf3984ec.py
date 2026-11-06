"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    N = 256
    # 1. Filter out vertices v where 2*v is a perfect square
    valid_vertices = []
    for v in range(1, N + 1):
        s = 2 * v
        root = int(math.isqrt(s))
        if root * root != s:
            valid_vertices.append(v)
    
    # Map original values to indices in valid_vertices for efficient graph representation
    # We want to find MIS in G, which is Max Clique in complement G_bar
    # Let's build the complement graph adjacency list.
    # Edge in G_bar (u, v) if u+v is NOT a perfect square.
    
    M = len(valid_vertices)
    if M == 0:
        return []
        
    # Precompute squares up to 2*256 = 512
    squares = set()
    for k in range(1, 23): # 15^2 = 225, 16^2=256, 17^2=289, 18^2=324, 19^2=361, 20^2=400, 21^2=441, 22^2=484, 23^2=529
        squares.add(k * k)
        
    # Build adjacency list for complement graph
    # adj[i] is a set/list of indices j such that valid_vertices[i] + valid_vertices[j] is NOT a square
    # Note: i != j
    adj = [[] for _ in range(M)]
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            if (vi + vj) not in squares:
                adj[i].append(j)
                adj[j].append(i)
                
    # We want to find the maximum clique in this graph.
    # Tomita's algorithm with coloring bound.
    
    # Order vertices by degree descending to improve pruning? 
    # Actually, standard Tomita works on the given order. 
    # Let's try to order by degree descending initially for better bounds.
    degrees = [len(adj[i]) for i in range(M)]
    order = sorted(range(M), key=lambda x: -degrees[x])
    
    # Remap graph according to order
    # new_index[old_index] = new_index
    new_index = [0] * M
    for new_i, old_i in enumerate(order):
        new_index[old_i] = new_i
        
    new_adj = [[] for _ in range(M)]
    for i in range(M):
        old_i = order[i]
        for j in adj[old_i]:
            new_j = new_index[j]
            new_adj[i].append(new_j)
            
    # Sort neighbors in new_adj to ensure consistency? Not strictly needed for set logic, but good for determinism.
    for i in range(M):
        new_adj[i].sort()
        
    # Convert to sets for faster intersection? Or just lists.
    # For M=256, lists are fine. Sets might be slower for small sizes.
    # Let's use a bitset approach if possible, or just list intersection.
    # Given M is small, list intersection is O(M^2) per node, which might be too slow if tree is deep.
    # However, coloring bound prunes heavily.
    
    # Let's use a recursive backtracking with coloring.
    # P: set of candidate vertices
    # X: set of already processed vertices (not used for bound, but for reconstruction)
    # current_clique: list of vertices in current clique
    
    best_clique = []
    
    # Coloring function: 
    # Given a set of vertices P, color them such that adjacent vertices have different colors.
    # The color of a vertex is an upper bound on the clique size it can contribute.
    # We want to find a greedy coloring of P.
    
    def greedy_coloring(P):
        # P is a list of vertices
        # Returns a list of colors for each vertex in P, and the max color
        # We process vertices in the order they appear in P (which is sorted by degree desc initially, but changes as we recurse)
        # Actually, standard Tomita reorders P based on coloring.
        
        # We will maintain a list 'U' of uncolored vertices.
        # And a list of color classes.
        
        # To make it efficient, we can use a simple greedy approach:
        # color[v] = c
        # We iterate through P.
        
        # Optimization: If we pass P as a list, we can just iterate.
        # The bound is the maximum color assigned.
        
        colors = {}
        max_color = 0
        # We need to check adjacency with previously colored vertices in the same color class?
        # No, greedy coloring: assign smallest possible color.
        
        # For efficiency, we can keep track of which colors are "blocked" for a vertex.
        # But M is small.
        
        # Let's use a simpler bound: 
        # The size of the maximum clique in P is at most the number of colors in a valid coloring.
        
        # We will implement a specific heuristic: 
        # Order P by degree within P descending.
        # Then assign colors.
        
        # Actually, the standard Tomita implementation uses a specific structure.
        # Let's stick to a clean implementation.
        
        # P is a list of vertices.
        # We want to color them.
        # color_class[c] = list of vertices with color c
        
        # We iterate through P.
        # For each vertex v, find the smallest color c such that v is not adjacent to any vertex in color_class[c].
        
        # This is O(M^2) per coloring step.
        # Number of nodes in search tree is the bottleneck.
        
        # Let's optimize by using bitsets if M <= 256 (fits in 4 ints or 1 long).
        # Python ints are arbitrary precision, so we can use a single int as a bitset.
        
        # Let's switch to bitset representation for P and adjacency.
        
        pass # Placeholder, we will implement below

    # Bitset approach
    # adj_bits[i] is an integer with bits set for neighbors of i.
    adj_bits = [0] * M
    for i in range(M):
        mask = 0
        for j in new_adj[i]:
            mask |= (1 << j)
        adj_bits[i] = mask
        
    # Recursive function
    # P: bitset of available vertices
    # current_clique: list of vertices
    # depth: current size
    
    def expand(P, current_clique, depth):
        nonlocal best_clique, deadline
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if depth > len(best_clique):
                best_clique = current_clique[:]
            return
            
        # Greedy coloring of P to get an upper bound
        # We extract vertices from P
        # We want to order them by degree in P descending to get a tighter bound?
        # Or just process them.
        
        # Let's get the list of vertices in P
        # For bitsets, iterating bits is efficient.
        
        # We will perform a greedy coloring.
        # We maintain a list of color classes (bitsets).
        # We iterate through vertices in P.
        
        # To get a good bound, we should order vertices in P by their degree in P descending.
        # Let's calculate degrees in P for all v in P.
        
        # This might be expensive. 
        # Alternative: Just use the global degree order? 
        # Tomita's algorithm typically reorders P based on the coloring process.
        
        # Let's try a simpler bound first: 
        # The number of vertices in P is an upper bound.
        # If depth + popcount(P) <= len(best_clique): return
        
        # Popcount in Python: bin(P).count('1') is slow. 
        # Use int.bit_count() (Python 3.10+) or fallback.
        
        if hasattr(int, 'bit_count'):
             pc = P.bit_count()
        else:
             pc = bin(P).count('1')
             
        if depth + pc <= len(best_clique):
            return
            
        # Coloring
        # We will assign colors to vertices in P.
        # We want to find a coloring c: P -> {1..k} such that adjacent vertices have different colors.
        # The max color k is an upper bound on the clique size in P.
        
        # We process vertices one by one.
        # To improve the bound, we process vertices with high degree in P first.
        
        # Let's get the list of vertices in P
        verts_in_P = []
        temp_P = P
        while temp_P:
            # Get lowest set bit
            lsb = temp_P & -temp_P
            idx = lsb.bit_length() - 1
            verts_in_P.append(idx)
            temp_P ^= lsb
            
        # Sort by degree in P descending
        # deg_in_P(v) = popcount(adj_bits[v] & P)
        def deg_in_P(v):
            return (adj_bits[v] & P).bit_count()
            
        verts_in_P.sort(key=deg_in_P, reverse=True)
        
        # Now greedy color
        # color_class is a list of bitsets
        color_classes = []
        # We also need to map vertex -> color for the bound?
        # The bound is just the max color index + 1.
        # But we also need to prune branches based on colors.
        # In Tomita, we process vertices in order of increasing color.
        
        # Let's assign colors
        vertex_color = {}
        for v in verts_in_P:
            # Find smallest color c such that v is not adjacent to any vertex in color_classes[c]
            assigned = False
            for c in range(len(color_classes)):
                # Check if v has any neighbor in color_classes[c]
                if (adj_bits[v] & color_classes[c]) == 0:
                    vertex_color[v] = c
                    color_classes[c] |= (1 << v)
                    assigned = True
                    break
            if not assigned:
                c = len(color_classes)
                vertex_color[v] = c
                color_classes.append(1 << v)
                
        max_color = len(color_classes)
        
        if depth + max_color <= len(best_clique):
            return
            
        # Now recurse
        # We process vertices in the order of their colors.
        # Actually, standard Tomita:
        # For each color class from max_color down to 1:
        #   Take a vertex v from that class (usually the first one or one with high degree)
        #   Recurse with P' = (P \ {v}) & adj_bits[v]
        #   Remove v from P
        
        # To be efficient, we should process vertices in a specific order.
        # The bound for a vertex v with color c is depth + c.
        # If depth + c <= len(best_clique), we can skip v and all remaining vertices with color <= c?
        # No, the coloring is for the set P.
        
        # Let's iterate through vertices in P in the order defined by the coloring (descending color).
        # We maintain a working set P_work.
        
        P_work = P
        # We need to iterate through the vertices in the order they were colored?
        # Actually, we can just iterate through verts_in_P (which is sorted by degree).
        # But the bound is tighter if we use the color of the vertex.
        
        # Let's create a list of (color, vertex) and sort by color descending.
        colored_verts = [(vertex_color[v], v) for v in verts_in_P]
        colored_verts.sort(key=lambda x: -x[0])
        
        for c, v in colored_verts:
            if time.time() > deadline:
                return
                
            # Bound check
            if depth + 1 + c <= len(best_clique): 
                # Actually, the bound for the remaining vertices in P_work that have color <= c is c.
                # But we are removing v.
                # The standard check is: if depth + c <= len(best_clique), break?
                # Let's be safe: if depth + c <= len(best_clique), we can stop processing this color class and lower?
                # Since colors are descending, if depth + c <= best, then for any subsequent vertex with color <= c, depth + color <= best.
                # So we can break the loop.
                break
                
            # Include v in clique
            # New P = P_work & adj_bits[v]
            # But we must remove v from P_work for subsequent iterations
            
            P_next = P_work & adj_bits[v]
            
            expand(P_next, current_clique + [v], depth + 1)
            
            # Remove v from P_work
            P_work &= ~(1 << v)
            
            # Update bound? 
            # The coloring of the remaining P_work might be different, but we used the initial coloring.
            # This is valid for pruning.
            
            if P_work == 0:
                break

    expand((1 << M) - 1, [], 0)
    
    if not best_clique:
        # Fallback if something went wrong or no clique found (should not happen if M>0)
        # Return a valid single vertex if any
        if valid_vertices:
            return [valid_vertices[0]]
        return []
        
    # Convert indices back to values
    result = [valid_vertices[i] for i in best_clique]
    return result

# EVOLVE-BLOCK-END
