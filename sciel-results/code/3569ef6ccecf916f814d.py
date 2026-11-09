import time
from math import gcd

def construct():
    N = 512
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    # 1. Build Graph
    # adj[i] is a bitmask where bit j is 1 if i and j are compatible
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(N):
            b = j + 1
            # lcm(a,b) <= 512
            if (a * b) // gcd(a, b) <= 512:
                adj[i] |= (1 << j)
    
    # 2. Bron-Kerbosch with Greedy Coloring
    best_clique = []
    
    def color_sort(P):
        # Greedy coloring of the subgraph induced by P
        # Returns list of (vertex, color) sorted by color
        # We want to process high colors first to get better bounds?
        # Actually, we sort by color ascending, and process in reverse?
        # Tomita: order by color. The color of a vertex is the max clique size bound for the subgraph containing it and its successors in the ordering.
        
        uncolored = P
        colors = []
        c = 0
        while uncolored:
            c += 1
            # Find a maximal independent set in the remaining uncolored vertices
            # Greedy: pick a vertex, remove it and its neighbors from 'available' for this color class
            # Actually, standard greedy coloring for clique bound:
            # We assign colors such that if u and v are adjacent, color[u] != color[v] is NOT required.
            # Wait, the coloring is for the *complement* graph? No.
            # The bound is based on the fact that a clique in G is an independent set in G_bar.
            # No, the standard Tomita coloring is:
            # Color the vertices in P such that no two *adjacent* vertices in G have the same color?
            # No, that's graph coloring.
            # The clique bound using coloring:
            # We color the vertices of P. If we use $k$ colors, the max clique size is $\le k$?
            # This is true if the coloring is proper for the *complement* graph?
            # Let's stick to the standard Tomita description:
            # "We order the vertices in P by a greedy coloring heuristic. The color of a vertex v is an upper bound on the size of a clique containing v and only vertices that appear after v in the ordering."
            # To achieve this, we color the graph induced by P.
            # We assign color 1 to a maximal independent set? No.
            # We assign colors such that adjacent vertices have different colors?
            # If adjacent vertices have different colors, a clique can have at most one vertex of each color.
            # So max clique size <= number of colors used.
            # Yes. So we perform a greedy graph coloring on the subgraph induced by P.
            
            available = uncolored
            color_class = 0
            while available:
                # Pick a vertex
                # Heuristic: pick the one with highest degree in 'available'? Or just first?
                # For speed, just pick the first available bit.
                v = (available & -available).bit_length() - 1
                color_class |= (1 << v)
                # Remove v and its neighbors from 'available'
                available &= ~adj[v]
                available &= uncolored # ensure we stay in uncolored
            # Now color_class is an independent set in G (clique in G_bar)
            # Assign color c to all vertices in color_class
            for v_bit in color_class:
                 # need to map bit to index
                 pass
            # This is getting complex to implement efficiently with just bitmasks without storing color arrays.
            
            # Alternative:
            # Store colors in an array.
            pass
            
        # Let's use a simpler, slightly slower but correct approach for coloring:
        # Recursive coloring or iterative.
        
        # Since N=512, we can use an array `color` of size N.
        # But we need to color only vertices in P.
        
        # Let's implement a helper that returns a list of (v, color) sorted by color.
        
        colors_arr = [0] * N
        uncolored = P
        c = 0
        ordered = []
        while uncolored:
            c += 1
            # Greedy independent set selection
            # To make the bound tight, we should pick vertices that constrain the rest most?
            # Simple greedy: pick first available.
            # But to get a *good* bound (low max color), we want to color the graph densely.
            # Standard greedy: process vertices in some order (e.g., degree order) and assign smallest possible color.
            
            # Let's try: 
            # 1. Extract list of vertices in P.
            # 2. Sort by degree in P (descending).
            # 3. Greedy assign colors.
            
            verts = []
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                verts.append(v)
                temp ^= lsb
            
            # Sort by degree in P
            # Degree of v in P is popcount(adj[v] & P)
            verts.sort(key=lambda v: (adj[v] & P).bit_count(), reverse=True)
            
            # Greedy Coloring
            # color[v] = smallest color not used by neighbors of v in P that are already colored
            # Since we process in order, we only look at neighbors that come before v in this order?
            # No, greedy coloring usually requires looking at all already colored neighbors.
            
            used_colors = [0] * N # used_colors[v] is bitmask of colors used by neighbors of v?
            # Or just a set of colors used by each vertex's colored neighbors.
            
            # Simpler:
            # color[v]
            # for v in verts:
            #    forbidden = set()
            #    for u in neighbors of v in P:
            #       if u is colored:
            #          forbidden.add(color[u])
            #    color[v] = smallest positive int not in forbidden
            
            # This is O(N^2) per coloring step. With N=512, N^2 = 260k.
            # Bron-Kerbosch might call this many times.
            # But the tree is pruned heavily.
            
            # Let's optimize the neighbor check.
            # We can precompute adjacency.
            
            # Implementation of coloring:
            color = {}
            # We need to check neighbors in P.
            # adj[v] & P gives neighbors in P.
            
            # To speed up, we can maintain a bitmask of colors used by neighbors?
            # Colors are small (<= N).
            
            # Let's just do the simple version first.
            
            # Reset colors for vertices in P
            for v in verts:
                color[v] = 0
            
            for v in verts:
                # Find neighbors of v in P that are already colored
                # Neighbors in P: adj[v] & P
                # But we only care about those in `verts` that are processed (colored).
                # Since we process in order, we can just check all u in verts such that u is neighbor of v and color[u] > 0.
                
                forbidden = 0 # bitmask of colors
                # Iterate neighbors in P
                nb_mask = adj[v] & P
                # We need to check which of these are colored.
                # Instead of iterating bits of nb_mask, we can iterate `verts`? No, that's O(N).
                # Iterating bits of nb_mask is better if sparse.
                
                # Actually, we can just iterate through the list of vertices we've colored so far?
                # No, we need to check adjacency.
                
                # Let's iterate through bits of nb_mask.
                m = nb_mask
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    if color[u] > 0:
                        forbidden |= (1 << color[u])
                    m ^= lsb
                
                # Find smallest color not in forbidden
                c_val = 1
                while (forbidden >> c_val) & 1:
                    c_val += 1
                color[v] = c_val
                
            # Now we have colors for all v in P.
            # Sort vertices by color ascending.
            # Tomita processes in reverse order (highest color first) to get the bound?
            # The bound for a vertex v is its color.
            # If we process v, the max clique size extending from v is <= color[v].
            # So we should process vertices with higher colors first?
            # If we process a low color vertex, the bound is low, so we prune early.
            # Yes, process in increasing color order?
            # Wait. If color[v] is small, the bound is small.
            # If current_size + color[v] <= best, we prune.
            # So we want to prune early.
            # So we should check vertices with small colors first?
            # But if we pick a small color vertex, we might miss a large clique that requires a high color vertex?
            # No, the bound is valid for ANY clique containing v.
            # If we process v and prune, we don't explore it.
            # If we don't prune, we explore.
            # To find the MAX clique, we need to explore all valid branches.
            # The order affects pruning efficiency.
            # Tomita's algorithm sorts by color and processes in *reverse* order (highest to lowest)?
            # Let's check: "The vertices are ordered by color. The algorithm processes them in reverse order."
            # Why? Because the color of a vertex is an upper bound on the clique size in the subgraph induced by that vertex and all vertices with *higher* or *equal* color?
            # No, the coloring is done on the whole P.
            # The property is: The max clique in the subgraph induced by $\{v_i, \dots, v_k\}$ is at most $color(v_k)$?
            # Actually, the standard bound is: If we order $v_1, \dots, v_k$ such that $color(v_1) \le \dots \le color(v_k)$, then the max clique in the subgraph induced by $\{v_i, \dots, v_k\}$ has size at most $color(v_k)$.
            # So if we are at $v_i$, the remaining candidates are $\{v_i, \dots, v_k\}$ (if we process in order).
            # But in BK, we pick a vertex $v$, and the new P is $P \cap N(v)$.
            # The bound is applied before picking.
            
            # Let's follow Tomita exactly:
            # 1. Color P.
            # 2. Sort P by color ascending.
            # 3. Iterate from last to first (highest color to lowest).
            # 4. For vertex v at index i (0-indexed from start, so we start at k-1), the bound is color[v].
            # 5. If current_size + color[v] <= best, break (since subsequent vertices have lower/equal color, their bounds are also <= best).
            # 6. Recurse with P' = P \ {v_0, ..., v_{i-1}}? No.
            #    P' = (P \ {v}) \cap N(v).
            #    And we must remove v from P for subsequent iterations in this loop.
            #    Also, we should remove vertices with color < color[v] from consideration for this branch?
            #    No, the standard BK with coloring just uses the bound to prune.
            #    The set of candidates for the next level is $P \cap N(v)$.
            #    The set of candidates for the *next iteration of the loop* is $P \setminus \{v\}$.
            
            # Let's implement this.
            
            # Return the sorted list of vertices
            verts.sort(key=lambda v: color[v])
            return verts, color

    def bk(P, X, clique_size):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if not P:
            if clique_size > len(best_clique):
                best_clique = clique_size # We need to store the actual set, not just size
                # This is a problem. We need to track the set.
                # Passing a list and appending is O(N) per level.
                # Better to pass a tuple or use a global list and backtrack.
            return
        
        # Color and sort
        verts, colors = color_sort(P)
        
        # Process in reverse order (highest color first)
        # We need to maintain the set of available vertices for the loop.
        # In Tomita, we iterate through the sorted list.
        # For each v, we check bound.
        # Then we recurse with P_new = P & adj[v] & ~ (bits of vertices processed so far in this loop?)
        # Actually, P_new = P & adj[v].
        # But we must ensure we don't pick vertices that were "excluded" by previous choices in this loop?
        # In BK, the loop variable effectively removes v from P for subsequent iterations.
        # So P_loop starts as P.
        # For each v in reversed(verts):
        #    if clique_size + colors[v] <= best_size: break
        #    P_new = P_loop & adj[v]
        #    X_new = X | P_loop_without_v? No.
        #    Standard BK:
        #    for v in P:
        #       P' = P \ {v}
        #       X' = X \ {v}
        #       recurse(P & N(v), X & N(v), C U {v})
        #       P = P'
        #       X = X'
        
        # With coloring, we sort P.
        # We iterate v in sorted_P (reverse).
        # P_current is the set of vertices in sorted_P from index i to end?
        # No, P_current is the set of vertices not yet processed in this loop.
        
        P_loop = P
        # We need to iterate through verts in reverse.
        # But verts contains all vertices in P.
        # As we process v, we remove it from P_loop.
        
        for v in reversed(verts):
            # Bound check
            if clique_size + colors[v] <= len(best_clique):
                break
                
            # New P for recursion: neighbors of v in P_loop
            P_new = P_loop & adj[v]
            
            # New X for recursion: neighbors of v in X
            X_new = X & adj[v]
            
            # Recurse
            # We need to add v to clique
            # To avoid copying list, we can use a global stack or pass a tuple.
            # Given N=512, passing a list and slicing/appending is probably fine if depth is small.
            # But let's use a global `current_clique` list and backtrack.
            
            current_clique.append(v)
            bk(P_new, X_new, clique_size + 1)
            current_clique.pop()
            
            # Update P_loop and X for next iteration
            P_loop &= ~(1 << v)
            # X doesn't need to be updated for the loop itself, only passed to recursion.
            # Wait, in standard BK, X is updated to exclude v for subsequent siblings.
            # But here, the "siblings" are the next v in the loop.
            # The X passed to recursion is `X & adj[v]`.
            # The X for the *next* iteration of the loop should be `X | {v}`?
            # No, X represents vertices that were removed from P in previous steps and are not in N(v).
            # Actually, the standard BK loop:
            # while P:
            #   v = pop(P)
            #   ...
            #   X.add(v)
            
            # So yes, we should add v to X for the purpose of the next sibling?
            # No, the X passed to the *next* sibling's recursion is not used directly.
            # The X passed to the *next* sibling's recursion is `X_current & adj[v_next]`.
            # But `X_current` should have v in it?
            # Let's check the logic.
            # X contains vertices that are not in P but are compatible with C.
            # When we pick v, we move v from P to X?
            # Yes, effectively.
            # So for the next iteration, X should include v.
            X |= (1 << v)
            
            if not P_loop:
                break

    current_clique = []
    best_size = 0
    
    # Initial call
    # P is all vertices
    P_init = (1 << N) - 1
    X_init = 0
    
    # We need to update best_clique when we find a larger one.
    # My `bk` function above updates `best_clique` size but not the list.
    # I need to fix that.
    
    # Revised BK
    def bk2(P, X, size):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P:
            if size > best_size:
                best_size = size
                best_clique = list(current_clique)
            return
        
        verts, colors = color_sort(P)
        
        P_loop = P
        X_loop = X
        
        for v in reversed(verts):
            if size + colors[v] <= best_size:
                break
            
            P_new = P_loop & adj[v]
            X_new = X_loop & adj[v]
            
            current_clique.append(v)
            bk2(P_new, X_new, size + 1)
            current_clique.pop()
            
            P_loop &= ~(1 << v)
            X_loop |= (1 << v)
            
            if not P_loop:
                break

    bk2(P_init, X_init, 0)
    
    # Convert indices to values (1-based)
    result = [v + 1 for v in best_clique]
    
    # Fallback if empty (should not happen)
    if not result:
        result = [1]
        
    return result
