# EVOLVE-BLOCK-START
import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a maximum subset of Z/205Z such that no two elements differ by a 
    non-zero quadratic residue modulo 205.
    
    Uses a Tomita-style Branch and Bound algorithm on the complement graph 
    (Maximum Clique problem) with bitset representations.
    """
    m = 205
    k = 2
    
    # Deadline guard: 630s limit, use 600s for safety
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # 2. Build the complement graph (compatibility graph) as bitmasks
    # In this graph, an edge exists between i and j if they are NOT in conflict.
    # We want to find a Maximum Clique in this graph, which corresponds to 
    # an Independent Set in the original conflict graph.
    
    # neighbors_comp[i] is a bitmask of vertices compatible with i
    neighbors_comp = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j:
                d = (i - j) % m
                if d not in forbidden:
                    mask |= (1 << j)
        neighbors_comp[i] = mask

    # Helper to get the list of indices set in a bitmask
    def bits_to_list(mask):
        return [i for i in range(m) if (mask >> i) & 1]

    # 3. Tomita Algorithm (Maximum Clique)
    # State:
    # R: current clique (list of vertices)
    # P: candidate set (bitset of vertices that can extend R)
    
    best_size = 0
    best_clique = []
    
    # Greedy coloring to get upper bounds
    # We need a function that colors the subgraph induced by P.
    # Returns: colors (list of color indices for each vertex in P), and max_color
    # We implement a simple greedy coloring:
    # Sort vertices in P by some heuristic (e.g., degree in P) or just iterate.
    # Assign the smallest available color.
    
    def color_bound(P, neighbors):
        """
        Greedy coloring of the subgraph induced by P.
        Returns max_color used.
        """
        if P == 0:
            return 0
        
        # Extract list of vertices in P
        # For efficiency, we can process vertices in a specific order.
        # A common heuristic is to order by degree in the induced subgraph, 
        # but extracting and sorting is expensive. 
        # Let's use a simple approach: iterate through bits.
        
        # To speed up, we can pre-compute degrees or just use a fixed order.
        # For Tomita, the ordering of coloring affects the tightness of the bound.
        # Let's try ordering by degree in P (descending) for better bounds.
        
        # Get vertices in P
        verts = []
        temp = P
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts.append(v)
            temp &= temp - 1
            
        if not verts:
            return 0
            
        # Calculate degree in P for each vertex
        # deg[v] = popcount(neighbors[v] & P)
        # To avoid sorting overhead if P is small, we might just iterate.
        # But sorting by degree descending usually gives tighter bounds.
        
        # Optimization: If P is small, sorting is fine.
        # Let's calculate degrees and sort.
        degs = []
        for v in verts:
            deg = bin(neighbors[v] & P).count('1')
            degs.append((deg, v))
            
        # Sort by degree descending
        degs.sort(reverse=True)
        
        # Assign colors
        # color_of[v] = c
        # colors_used[c] = set of vertices with color c
        # We want to assign the smallest possible color.
        
        # Simple greedy:
        # For each vertex v in sorted order:
        #   Find smallest color c such that no neighbor of v in P has color c.
        
        # To make this fast, we can keep track of colors assigned to neighbors.
        # However, for N=205, a simple check is likely fast enough if we do it well.
        
        # Let's use an array to store the color of each vertex.
        color = [0] * m
        max_c = 0
        
        # We process vertices in the order of degs (highest degree first).
        # This is a heuristic for coloring.
        
        # To check if a color is available, we need to know which neighbors of v
        # have that color.
        # We can maintain a list of vertices per color.
        # color_classes = [set() for _ in range(max_c+1)]
        
        # Alternative: Just check neighbors.
        # For v in sorted order:
        #   used_colors = set(color[u] for u in neighbors[v] & P if color[u] != 0)
        #   c = 1
        #   while c in used_colors: c += 1
        #   color[v] = c
        #   max_c = max(max_c, c)
        
        # This is O(N^2) in the worst case per bound calculation.
        # With N=205, N^2 ~ 40k ops. If called many times, it might be slow.
        # But Tomita's pruning is very effective, so the number of calls might be low.
        
        # Let's implement this.
        
        # We need to be careful: only consider neighbors that are in P.
        # And only consider neighbors that have already been colored (i.e., processed earlier).
        # Wait, the standard greedy coloring for Tomita processes vertices in a specific order.
        # The order determines the bound.
        
        # Let's stick to the sorted order by degree.
        
        for deg, v in degs:
            # Find smallest color not used by neighbors in P that are already colored?
            # No, in greedy coloring for bound, we assign colors to vertices in P.
            # The condition is: v and u cannot share color if u is a neighbor of v.
            # Since we are coloring the induced subgraph, we check all neighbors in P.
            # But we process one by one. If we check neighbors that are NOT yet colored, 
            # their color is 0.
            
            # So, we check neighbors u in P such that u has been processed (colored).
            # How do we know which are processed? 
            # The `degs` list defines the processing order.
            # We can just check all neighbors in P. If a neighbor hasn't been colored yet, 
            # its color is 0, which doesn't conflict.
            
            used = set()
            # Iterate neighbors of v in P
            nb_mask = neighbors[v] & P
            temp_nb = nb_mask
            while temp_nb:
                lsb = temp_nb & -temp_nb
                u = lsb.bit_length() - 1
                if color[u] > 0: # If u is already colored
                    used.add(color[u])
                temp_nb &= temp_nb - 1
                
            c = 1
            while c in used:
                c += 1
                
            color[v] = c
            if c > max_c:
                max_c = c
                
        return max_c

    # Recursive B&B function
    # To avoid recursion depth issues and overhead, we can use an explicit stack, 
    # but recursion is cleaner for Tomita if depth is < 205.
    # Python recursion limit is usually 1000, so 205 is safe.
    
    def expand(R, P):
        nonlocal best_size, best_clique
        
        # If P is empty, update best
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        
        # Calculate upper bound
        # The size of the clique cannot exceed len(R) + max_color
        # We compute max_color using greedy coloring on P
        
        # Optimization: If len(R) + popcount(P) <= best_size, prune
        # This is a weaker bound than coloring, but faster.
        if len(R) + bin(P).count('1') <= best_size:
            return
            
        max_c = color_bound(P, neighbors_comp)
        
        if len(R) + max_c <= best_size:
            return
            
        # Select a vertex to branch on
        # Tomita's strategy: Pick a vertex v from P that has the highest color number?
        # Or just pick the first one?
        # The standard Tomita algorithm picks vertices in the order of the coloring.
        # Specifically, it iterates through vertices in P, and for each, it branches.
        # To prune effectively, we should process vertices with higher color numbers first?
        # Actually, the bound `max_c` is the max color.
        # We can iterate through vertices in P.
        
        # Let's pick a vertex v from P.
        # A good heuristic is to pick the vertex with the highest degree in P, 
        # or the one with the highest color assigned in the coloring step.
        # Since we computed colors in color_bound, we don't have them here easily unless we return them.
        # Let's just pick the first vertex in P? 
        # Or better: Pick a vertex v such that its color is max_c?
        
        # To keep it simple and robust, let's pick the vertex with the highest degree in P.
        # We can reuse the logic from color_bound to find the best v.
        
        # Actually, let's modify color_bound to return the order of vertices and their colors.
        # Or simpler: Just pick any vertex? No, that's inefficient.
        
        # Let's pick the vertex v in P with the maximum color assigned in the greedy coloring.
        # We need to re-calculate or store it.
        # Let's modify the approach:
        # 1. Compute colors and order.
        # 2. Iterate through vertices in reverse order of coloring (highest color first).
        
        # Let's refactor: 
        # We will compute the coloring and store the colors.
        
        # Re-implementing the bound calculation to also give us the vertex order.
        
        # For now, let's just pick the vertex with the highest degree in P.
        # This is a common heuristic.
        
        best_v = -1
        max_deg = -1
        temp = P
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = bin(neighbors_comp[v] & P).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp &= temp - 1
            
        if best_v == -1:
            return
            
        # Branch on best_v
        # Include best_v
        new_P = P & neighbors_comp[best_v]
        R.append(best_v)
        
        # Check time
        if time.time() > deadline:
            return
            
        expand(R, new_P)
        
        R.pop()
        
        # Exclude best_v
        P &= ~(1 << best_v)
        
        # Check time
        if time.time() > deadline:
            return
            
        expand(R, P)

    # Initial call
    initial_P = (1 << m) - 1
    # We can sort initial vertices? The algorithm handles it.
    # To improve, we can order the initial P by degree?
    # The expand function picks the best vertex dynamically.
    
    # One issue: `color_bound` is called frequently and is O(N^2).
    # For N=205, this is ~40k operations. 
    # If the tree is deep, this might be slow.
    # But Tomita's pruning is strong.
    
    # Let's run it.
    expand([], initial_P)
    
    # Extract result
    result_list = best_clique
    
    # Fallback if something went wrong
    if not result_list:
        result_list = [0]
        
    return np.array(sorted(result_list), dtype=np.int64)


# EVOLVE-BLOCK-END
