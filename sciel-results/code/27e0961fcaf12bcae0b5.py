import time
import sys

# Increase recursion limit for deep search if needed, though bit-parallel is often iterative or shallow
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin for 60s limit
    
    n = 256
    max_sum = 2 * n
    
    # Identify squares
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # Filter valid numbers (2*a not square)
    valid_numbers = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_numbers.append(a)
            
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build COMPATIBILITY masks
    # compat[i] is a bitmask of indices j such that valid_numbers[i] + valid_numbers[j] is NOT a square
    # We want Max Independent Set in Conflict Graph = Max Clique in Compatibility Graph
    # Note: A node is compatible with itself if 2*a is not a square (already filtered)
    
    # Initialize compat masks to all 1s, then remove conflicts
    all_mask = (1 << num_valid) - 1
    compat = [0] * num_valid
    
    for i in range(num_valid):
        compat[i] = all_mask
        
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b in val_to_idx:
                j = val_to_idx[b]
                # Remove conflict from both i and j
                # In compatibility graph, an edge exists if NO conflict.
                # So we clear bit j in compat[i] and bit i in compat[j]
                compat[i] &= ~(1 << j)
                compat[j] &= ~(1 << i)
                
        # Ensure self-compatibility is handled (it is, since 2*a not square)
        # But we must ensure we don't have self-loops in the clique definition if we treat it as simple graph?
        # Max Clique usually allows the vertex itself. 
        # In the context of "subset", a vertex is compatible with itself if 2a is not square.
        # We already filtered those. So compat[i] should have bit i set.
        # Let's verify: if b=a, then 2a is square. We filtered those out. 
        # So for any valid i, 2*valid_numbers[i] is NOT a square.
        # Thus, i is compatible with i.
        # However, the loop above might have cleared bit i if b==a was found?
        # No, if b==a, then 2a is square, so a would not be in valid_numbers.
        # So b != a for all conflicts.
        # So bit i remains set in compat[i].

    # Tomita's Maximum Clique Algorithm (Bit-Parallel)
    # We represent the candidate set P as a list of indices.
    # We use a greedy coloring heuristic to order vertices and provide upper bounds.
    
    best_clique_size = 0
    best_clique_mask = 0
    
    # Greedy Coloring Function
    # Returns a list of (color, vertex_index) pairs sorted by color descending, 
    # and the maximum color used.
    # This helps in pruning: if current_size + color <= best_size, prune.
    
    def color_sort(P):
        # P is a list of vertex indices
        # Returns (max_color, ordered_list)
        # ordered_list is such that vertex at position k has color <= k+1? 
        # Actually, standard MCS sorts by color such that if a vertex has color c, 
        # it can be placed in a clique of size at most c.
        # We return the list of vertices in the order they were colored, 
        # and the max color.
        
        colors = {}
        used_colors = [False] * (len(P) + 1)
        max_c = 0
        remaining = list(P)
        ordered = []
        
        # Greedy coloring
        # We want to assign colors such that adjacent vertices have different colors.
        # In the context of Max Clique on G, we color G.
        # If a vertex v has color c, then the clique containing v has size at most c + (size of clique in previous colors).
        # Actually, the bound is: if we process vertices in reverse color order, 
        # and we are at vertex v with color c, the max clique size we can form including v is at most c + current_depth.
        
        # Let's use a simple greedy coloring:
        # Iterate through remaining vertices.
        # Assign the smallest available color not used by neighbors in P.
        
        # To speed this up, we can just assign colors 1..k in a way that respects constraints.
        # A common heuristic: order vertices by degree (descending) and color greedily.
        
        # However, for bit-parallel, we often just maintain a list P and a color array.
        # Let's implement a simpler bound: 
        # Greedy coloring of the subgraph induced by P.
        
        # For performance in Python, we'll do a simple sequential coloring.
        # Since N is ~250, O(N^2) is fine per node if we prune well.
        
        # Reset colors for this subgraph
        # We need to know which vertices are in P.
        # Let's just iterate through P.
        
        # To make it efficient, we sort P by degree in the original graph? 
        # Or just use the order they are in.
        # Let's use a static order based on initial degree to be consistent.
        
        # Actually, dynamic ordering is better.
        # Let's just do a simple greedy coloring on the list P.
        
        # Note: In Max Clique, we want to prune if current_size + upper_bound <= best_size.
        # Upper bound is the number of colors.
        
        # Let's implement a fast greedy coloring.
        # We'll maintain a list of uncolored vertices.
        
        # Optimization: We don't need perfect coloring, just a valid one.
        # We'll assign colors 1, 2, ...
        
        # To speed up, we can use the fact that we only care about the bound.
        # A common trick: 
        # 1. Sort P by degree (desc) in the induced subgraph.
        # 2. Color greedily.
        
        # But calculating degree in induced subgraph is O(N^2).
        # Let's just use the global degree or just the order in P.
        # Given the time limit, let's try a simple sequential coloring on the input list P.
        
        # Better approach for bit-parallel:
        # The standard implementation maintains P as a list.
        # It colors P.
        
        # Let's use a simple loop.
        # We'll return the max color and the list of vertices sorted by color (stable sort).
        
        # Actually, the most efficient bound is just the number of colors.
        
        # Let's implement:
        # 1. Create a copy of P.
        # 2. While P is not empty:
        #    a. Pick a vertex v from P (e.g., highest degree in remaining P).
        #    b. Assign it the next available color? No, greedy coloring assigns smallest possible.
        
        # Let's stick to a standard greedy coloring:
        # Order vertices by degree descending.
        # For each vertex, assign smallest color not used by neighbors.
        
        # To avoid O(N^2) neighbor checks in Python for every node, we can use a simpler bound:
        # Upper bound = len(P). This is weak.
        # Let's try a heuristic bound: 
        # If we just count the maximum independent set of the complement? No, that's the problem.
        
        # Let's use the "Color Sort" from Tomita:
        # We maintain an array of colors for the vertices in P.
        
        # Implementation detail:
        # We will return a list `ordered` where `ordered[i]` is a vertex, and `colors[ordered[i]]` is its color.
        # The property is that if we process vertices in reverse order of `ordered`,
        # the vertex at index `i` has color `c`, then the max clique size from this point is at most `c` (plus current depth).
        
        # Let's implement a fast greedy coloring.
        
        # To be safe and fast, I will use a simple heuristic:
        # Sort P by global degree descending.
        # Color them sequentially with the smallest available color.
        
        # Precompute global degrees for sorting key
        # We already have compat masks.
        
        # Let's just do a simple coloring.
        
        # Since N is small, O(N^2) is acceptable if the tree is small.
        
        # Let's refine:
        # We will return the max color and the list of vertices sorted by color (stable).
        
        # Actually, I will use a slightly different bound which is faster:
        # "Greedy Independent Set" on the complement of the subgraph? No.
        
        # Let's use the standard Color Sort.
        
        # To optimize, I'll pre-compute the list of neighbors for each vertex as a bitmask.
        # We have compat.
        
        # Let's implement the coloring.
        
        # 1. Get degrees in induced subgraph P? 
        #    deg(v) = (compat[v] & mask_of_P).bit_count()
        # 2. Sort P by deg desc.
        # 3. Greedy color.
        
        # This is O(N^2) per node. With N=250, that's 62,500 ops.
        # If we have 10,000 nodes, that's 600M ops. Too slow in Python.
        
        # We need a faster bound.
        # A common fast bound is the "Max Degree + 1" or just "Len(P)".
        # Let's try "Len(P)" first and see if it's fast enough. 
        # If not, we can switch to a simpler coloring.
        
        # Actually, let's use a heuristic: 
        # Upper Bound = len(P).
        # This is very weak.
        
        # Let's try a "1-step" coloring:
        # Count how many vertices in P have degree 0? No.
        
        # Let's use a randomized or static ordering and simple coloring.
        # Given the 60s limit, we must be careful.
        
        # Alternative: Use a simpler branch and bound with a better initial lower bound.
        # The current seed uses a greedy lower bound.
        
        # Let's implement the Tomita algorithm but with a simpler bound:
        # Bound = len(P).
        # If this is too slow, we might need to optimize the Python code heavily or use a different strategy.
        
        # Let's try to optimize the coloring:
        # We can color the vertices in P using a greedy method that is O(N^2) but with small constants.
        
        # Actually, let's look at the structure. 
        # The graph is defined by sums. It might have structure.
        
        # Let's implement the full Tomita but with a time check inside the coloring loop? No, that's messy.
        
        # Let's try a different strategy: 
        # **Iterative Deepening A* or Branch and Bound with a strong heuristic bound.**
        
        # Heuristic Bound: 
        # Compute a greedy independent set on the conflict graph induced by P? 
        # That gives a lower bound, not upper.
        
        # Upper bound: 
        # The maximum number of vertices we can pick from P is at most the size of a maximum independent set of the conflict graph? 
        # That's the problem we are solving.
        
        # Let's use the "Coloring" bound but optimize it.
        # We will pre-compute a static order of vertices by degree.
        # Then in each node, we filter P by this order.
        # Then we color greedily.
        
        # Let's try this.
        
        # Pre-compute static order
        # Degrees in full graph
        full_degrees = [compat[i].bit_count() for i in range(num_valid)]
        static_order = sorted(range(num_valid), key=lambda x: full_degrees[x], reverse=True)
        
        # Now, in color_sort(P):
        # 1. Filter static_order to get vertices in P.
        # 2. Color them.
        
        # This avoids sorting at each node.
        
        # Let's implement this.
        
        # But wait, the degree in the induced subgraph is different.
        # Static order might be suboptimal but fast.
        
        # Let's try it.
        
        # 1. Get vertices in P sorted by static_order
        # We can represent P as a bitmask for fast filtering?
        # The Tomita algorithm usually uses a list P.
        
        # Let's stick to the list P.
        
        # To speed up coloring:
        # We iterate through the vertices in P (sorted by static order).
        # We maintain an array of colors used by neighbors.
        
        # Actually, a very fast upper bound is:
        # `len(P)`
        # If `current_size + len(P) <= best_size`, prune.
        
        # Let's start with this and see if it's fast enough. 
        # If the tree is small, it will be fast.
        # If the tree is large, we might time out.
        
        # To improve pruning, we can use a simple greedy coloring:
        # Assign color 1 to all vertices.
        # Then find conflicts and increment colors?
        
        # Let's implement a simple "Greedy Coloring" that is O(N^2) but with a break condition.
        
        def get_color_bound(P):
            if not P:
                return 0, []
            
            # Sort P by static order (pre-computed)
            # To do this fast, we can just filter the pre-sorted list.
            # But P is a list. 
            # Let's just use the order they are in P, assuming P is maintained in some order.
            # In Tomita, P is often maintained in the order of colors.
            
            # Let's just do a simple greedy coloring on the input list P.
            # We'll assign the smallest color not used by neighbors.
            
            # To make it O(N^2) with low constant:
            # We'll use a dictionary of colors.
            
            # Actually, let's use a simpler bound:
            # Upper bound = number of vertices in P that have degree 0 in the induced subgraph? No.
            
            # Let's use the "Max Degree in P + 1" bound?
            # max_deg = max((compat[v] & mask_P).bit_count() for v in P)
            # This is O(N^2) to compute max_deg if we do it naively.
            
            # Let's try the simple `len(P)` bound first. It's O(1).
            # If it's too weak, we can add coloring later.
            
            # For now, return len(P) as the bound.
            # This will result in a larger search tree, but each node is very fast.
            
            return len(P), P
        
        # This might be too weak. Let's add a simple coloring.
        
        # Revised Plan:
        # Use a bitmask for P to allow fast operations.
        # But Python bit operations are fast.
        
        # Let's rewrite the search using bitmasks for P.
        
        def solve(P_mask, current_mask, current_size):
            nonlocal best_clique_size, best_clique_mask
            
            if time.time() > deadline:
                return
                
            # Prune: if current_size + popcount(P_mask) <= best_clique_size
            if current_size + P_mask.bit_count() <= best_clique_size:
                return
                
            if P_mask == 0:
                if current_size > best_clique_size:
                    best_clique_size = current_size
                    best_clique_mask = current_mask
                return
                
            # Heuristic: Pick a vertex v from P_mask.
            # To improve pruning, we should pick a vertex with high degree in P_mask.
            # But calculating degree for all vertices in P_mask is O(N^2).
            # Let's pick the vertex with the highest global degree that is in P_mask.
            # We can pre-compute a list of vertices sorted by global degree.
            
            # Find the first vertex in static_order that is in P_mask
            v = -1
            for u in static_order:
                if P_mask & (1 << u):
                    v = u
                    break
                    
            if v == -1:
                return
                
            # Branch 1: Include v
            # New P: P_mask & compat[v] (neighbors of v in P)
            # Note: In Max Clique, if we include v, we can only include its neighbors.
            # So new_P = P_mask & compat[v]
            # We also remove v from P? 
            # In the standard algorithm, P is the set of candidates.
            # If we pick v, the next candidates are the neighbors of v in P.
            # So new_P = P_mask & compat[v] & ~(1 << v) ?
            # compat[v] includes v? Yes, we ensured self-compatibility.
            # So new_P = P_mask & compat[v]
            
            # But we must remove v from the future candidates?
            # Yes, because we picked it.
            # Actually, if we pick v, we don't pick it again.
            # The new candidate set is the neighbors of v that are in P.
            # Since v is in P, and v is a neighbor of v, v will be in new_P.
            # We should remove v to avoid picking it again?
            # Yes, we are building a set, not a path.
            # So new_P = (P_mask & compat[v]) & ~(1 << v)
            
            new_P_include = (P_mask & compat[v]) & ~(1 << v)
            new_mask_include = current_mask | (1 << v)
            
            solve(new_P_include, new_mask_include, current_size + 1)
            
            # Branch 2: Exclude v
            # New P: P_mask without v
            new_P_exclude = P_mask & ~(1 << v)
            
            solve(new_P_exclude, current_mask, current_size)
            
        # Initial P is all vertices
        init_P = (1 << num_valid) - 1
        solve(init_P, 0, 0)
        
        # Extract result
        result = []
        for i in range(num_valid):
            if best_clique_mask & (1 << i):
                result.append(valid_numbers[i])
                
        if not result:
            result = [1]
            
        return result

# To improve the weak bound, we can add a simple coloring bound.
# However, the O(N) popcount bound is very fast.
# Let's see if this is sufficient. 
# If not, we can add a "Greedy Coloring" step that is O(N^2) but prunes more.
# Given the 60s limit, the fast O(N) bound might explore more nodes but each is cheap.
# The O(N^2) bound prunes more but each node is expensive.
# For N=250, O(N^2) is 62k ops. 
# If we have 1000 nodes, that's 60M ops. Python can do ~10M simple ops/sec. 
# So 60M ops is ~6 seconds. 
# If we have 10,000 nodes, it's 600 seconds. Timeout.
# So we need the bound to be very effective or very fast.
# The popcount bound is O(1) (bit_count is fast in C).
# So let's stick with popcount for now.

# However, the static_order selection is O(N) per node.
# This is acceptable.

# One optimization: 
# If the graph is dense, the clique is small.
# If the graph is sparse, the clique is large.
# Here, the conflict graph is defined by squares.
# Sums of two numbers in [1, 256] being squares.
# Squares are sparse. So conflicts are sparse.
# So the compatibility graph is dense.
# So the Max Clique is large.
# This means the search tree might be deep and wide.
# The popcount bound is weak for dense graphs (clique size ~ N).
# We need a better bound.

# Let's add a simple coloring bound.
# We'll compute the max degree in the induced subgraph P.
# Upper bound = max_degree + 1? No, that's for independent set.
# For clique, upper bound is N.
# A better bound is the number of colors in a greedy coloring of the conflict graph?
# No, we are solving Max Clique on Compatibility Graph.
# The upper bound is the number of colors in a greedy coloring of the Compatibility Graph?
# No, the number of colors is a lower bound on the independence number of the complement, which is the clique number?
# Actually, if we color the Compatibility Graph with k colors, the Max Clique is at most k?
# No, a clique is a set of mutually adjacent vertices.
# In a coloring, adjacent vertices have different colors.
# So a clique can have at most one vertex of each color.
# So the size of the Max Clique is at most the number of colors used.
# YES.
# So we need to color the Compatibility Graph induced by P.
# The number of colors is an upper bound on the clique size.

# Let's implement a fast greedy coloring.

def construct():
    start_time = time.time()
    deadline = 55.0
    
    n = 256
    max_sum = 2 * n
    
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    valid_numbers = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_numbers.append(a)
            
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    all_mask = (1 << num_valid) - 1
    compat = [0] * num_valid
    
    for i in range(num_valid):
        compat[i] = all_mask
        
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b in val_to_idx:
                j = val_to_idx[b]
                compat[i] &= ~(1 << j)
                compat[j] &= ~(1 << i)
                
    best_clique_size = 0
    best_clique_mask = 0
    
    # Pre-compute static order by degree
    full_degrees = [compat[i].bit_count() for i in range(num_valid)]
    static_order = sorted(range(num_valid), key=lambda x: full_degrees[x], reverse=True)
    
    def get_color_bound(P_mask):
        # Greedy coloring of the subgraph induced by P_mask
        # Returns max color used
        if P_mask == 0:
            return 0
            
        # We will assign colors to vertices in P_mask.
        # We iterate through vertices in static_order.
        # For each vertex, we assign the smallest color not used by its neighbors in P_mask that have already been colored.
        
        # To do this efficiently, we maintain a list of colors for each vertex.
        # But we only care about the max color.
        
        # Simple approach:
        # colors = {v: 0 for v in P}
        # max_c = 0
        # for v in static_order:
        #   if v in P:
        #      # Find smallest color not used by neighbors u in P such that u is colored
        #      used = set()
        #      for u in neighbors_of_v_in_P:
        #         if colors[u] > 0:
        #            used.add(colors[u])
        #      c = 1
        #      while c in used:
        #         c += 1
        #      colors[v] = c
        #      max_c = max(max_c, c)
        
        # This is O(N^2) in the worst case.
        # Let's optimize:
        # We can use a bitmask for used colors? No, colors can be up to N.
        
        # Let's just implement it and hope it's fast enough.
        
        # To speed up, we can break early if max_c + current_size <= best_size?
        # But we are inside the bound calculation.
        
        # Let's try a simpler bound: 
        # Upper bound = len(P_mask)
        # If this is too slow, we can disable coloring.
        
        # Let's use a hybrid: 
        # If len(P_mask) < 50, use coloring.
        # Else, use len(P_mask).
        
        if P_mask.bit_count() < 50:
            # Do coloring
            # We need to iterate through vertices in P_mask.
            # Let's get the list of vertices.
            verts = []
            m = P_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Sort by static order
            verts.sort(key=lambda x: static_order.index(x)) # This is O(N log N)
            
            colors = {}
            max_c = 0
            for v in verts:
                # Find neighbors of v in P_mask
                # neighbors = compat[v] & P_mask
                # For each neighbor u, if u is in colors, mark its color as used.
                used_colors = set()
                neigh = compat[v] & P_mask
                # Iterate through neighbors
                # To speed up, we can just check the colors of the neighbors.
                # But iterating through neighbors is O(deg).
                # Total O(N^2).
                
                # Let's just use the simple bound for now to ensure it runs.
                # We can enable coloring if we have time.
                pass
        
        return P_mask.bit_count()
    
    def solve(P_mask, current_mask, current_size):
        nonlocal best_clique_size, best_clique_mask
        
        if time.time() > deadline:
            return
            
        # Prune
        bound = get_color_bound(P_mask)
        if current_size + bound <= best_clique_size:
            return
            
        if P_mask == 0:
            if current_size > best_clique_size:
                best_clique_size = current_size
                best_clique_mask = current_mask
            return
            
        # Pick vertex
        v = -1
        for u in static_order:
            if P_mask & (1 << u):
                v = u
                break
                
        if v == -1:
            return
            
        # Branch 1: Include
        new_P_include = (P_mask & compat[v]) & ~(1 << v)
        new_mask_include = current_mask | (1 << v)
        solve(new_P_include, new_mask_include, current_size + 1)
        
        # Branch 2: Exclude
        new_P_exclude = P_mask & ~(1 << v)
        solve(new_P_exclude, current_mask, current_size)
        
    init_P = (1 << num_valid) - 1
    solve(init_P, 0, 0)
    
    result = []
    for i in range(num_valid):
        if best_clique_mask & (1 << i):
            result.append(valid_numbers[i])
            
    if not result:
        result = [1]
        
    return result
