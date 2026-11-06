import time
import sys

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before the 60s evaluator limit (actually 90s limit mentioned in prompt, but 60s in contract? "Each evaluation must finish within 60 s" vs "KILLS ... after 90s". I will use 55s to be safe for both interpretations, assuming 60s is the hard constraint for scoring validity or similar).
    
    n = 256
    # We want MIS in G. This is Max Clique in G_complement.
    # Edge in G_complement (a,b) if (a*b) % (a+b) != 0
    
    # adj_comp[i] is a bitmask of vertices that are NOT adjacent to i in G (i.e., adjacent in G_comp)
    # Actually, let's just build adj_comp directly.
    adj_comp = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) != 0:
                adj_comp[a] |= (1 << b)
                adj_comp[b] |= (1 << a)
    
    # Tomita's Algorithm for Maximum Clique
    # We work with the complement graph G_comp.
    # A clique in G_comp is an independent set in G.
    
    best_clique_size = 0
    best_clique_mask = 0
    
    # Greedy Coloring of a candidate set P
    # Returns a list of (vertex, color) sorted by color descending
    # and the max color
    def greedy_coloring(P):
        # P is a bitmask
        if P == 0:
            return [], 0
        
        colors = {}
        uncolored = P
        color = 0
        ordered = []
        
        while uncolored:
            color += 1
            # Find a maximal independent set in the subgraph induced by uncolored
            # In G_comp, we want to color such that adjacent vertices have different colors.
            # Wait, greedy coloring for upper bound in Max Clique:
            # We want to partition P into independent sets in G_comp.
            # An independent set in G_comp is a clique in G.
            # No, the standard Tomita coloring:
            # We assign colors to vertices in P such that if u and v are adjacent in G_comp, they have different colors.
            # This is a proper coloring of G_comp[P].
            # The number of colors is an upper bound on the clique size.
            
            # To get a tight bound, we should order vertices by degree or similar.
            # Standard greedy:
            # 1. Order vertices in P (e.g., by degree in P descending).
            # 2. Assign smallest available color.
            
            # For efficiency in Python with bitmasks, we can just iterate.
            # Let's use a simple greedy: pick a vertex, remove it and its neighbors from the "current color class" candidates?
            # No, that's for independent set.
            # For coloring G_comp[P]:
            # We want to find a set of vertices that are pairwise non-adjacent in G_comp (i.e., adjacent in G) to form a color class?
            # No. A color class is an independent set in G_comp.
            # So we want to find a maximal independent set in G_comp[uncolored].
            # An independent set in G_comp is a set of vertices with no edges between them in G_comp.
            # This is equivalent to a clique in G.
            
            # Let's just do a simple greedy coloring:
            # Iterate through vertices in uncolored.
            # Assign the smallest color not used by its neighbors in G_comp that are already colored.
            
            # To speed this up, we can pick vertices in a specific order.
            # Let's just pick the lowest set bit to keep it deterministic and fast.
            # Or better, pick the vertex with highest degree in G_comp[uncolored] to color first?
            # Actually, the order of coloring affects the bound.
            # Tomita suggests ordering by degree.
            
            # Let's try a simple version first:
            # Pick a vertex v from uncolored.
            # Assign it color `color`.
            # Remove v from uncolored.
            # Remove neighbors of v in G_comp from the "available for this color" set?
            # No, standard greedy coloring:
            # For each vertex v in P (in some order):
            #   find smallest color c such that no neighbor of v in G_comp has color c.
            #   assign c to v.
            
            # This is O(V^2) per coloring call if done naively.
            # V is at most 256. 256^2 = 65k ops.
            # Number of coloring calls is the number of nodes in the search tree.
            
            # Let's implement a fast greedy coloring.
            # We maintain a list of vertices to process.
            # We assign colors 1, 2, ...
            # For each color, we find a maximal independent set in the remaining graph.
            # This is equivalent to finding a maximal clique in G[remaining].
            
            # Let's stick to the vertex-by-vertex assignment, but optimize the neighbor check.
            
            # Actually, there is a simpler way to get a bound:
            # Just count the number of vertices if we don't care about tightness? No, that's too loose.
            
            # Let's use the "Maximal Independent Set in G_comp" approach for each color.
            # This is fast if we use bitmasks.
            
            # Current uncolored set: U
            # Color class C = empty
            # While U is not empty:
            #   Pick a vertex v from U (e.g., lowest bit)
            #   Add v to C
            #   Remove v and its neighbors in G_comp from U?
            #   No, neighbors in G_comp cannot be in the same color class.
            #   So remove v and N_G_comp(v) from U.
            
            # This finds a maximal independent set in G_comp, which is a color class.
            
            pass # Placeholder for logic
            
        # This function needs to return the ordered list and max color.
        # Let's rewrite greedy_coloring properly.
        
        # We will return a list of vertices sorted by color descending.
        # And the max color.
        
        pass

    # Let's implement a cleaner version of Tomita with bitmasks.
    
    def expand(R, P):
        nonlocal best_clique_size, best_clique_mask
        
        # If P is empty, we have a complete clique
        if P == 0:
            if R.bit_count() > best_clique_size:
                best_clique_size = R.bit_count()
                best_clique_mask = R
            return
        
        # If time is up, stop
        if time.time() - start_time > timeout:
            return
            
        # Greedy Coloring of P
        # We want to order vertices in P by color.
        # Color i is an independent set in G_comp (clique in G).
        
        U = P
        color = 0
        # We store vertices by color to sort them later
        # Actually, we just need the sequence of vertices in the order they were colored,
        # and for each, its color.
        # Then we sort by color descending.
        
        verts_by_color = []
        
        while U:
            color += 1
            # Find a maximal independent set in G_comp[U]
            # This is a maximal clique in G[U]
            C = 0
            U_rem = U
            while U_rem:
                # Pick a vertex, e.g., the one with highest degree in U_rem?
                # Or just the lowest bit for speed?
                # Let's pick the lowest bit to be fast.
                v_bit = U_rem & -U_rem
                v = v_bit.bit_length() - 1
                
                C |= v_bit
                # Remove v and its neighbors in G_comp from U_rem
                # Neighbors in G_comp are adj_comp[v]
                U_rem &= ~(adj_comp[v] | v_bit)
            
            # All vertices in C have color `color`
            # Add them to the list
            # We need to iterate through bits of C
            c = C
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                verts_by_color.append((v, color))
                c ^= lsb
                
            U &= ~C
            
        # Sort by color descending
        verts_by_color.sort(key=lambda x: x[1], reverse=True)
        
        max_color = color if color > 0 else 0
        
        # Branch and Bound
        # Iterate through vertices in sorted order
        for v, c_val in verts_by_color:
            if time.time() - start_time > timeout:
                return
                
            # Prune: if current size + max possible (which is c_val) <= best, skip
            if R.bit_count() + c_val <= best_clique_size:
                # Since colors are descending, all subsequent vertices have <= c_val
                # So we can break
                break
                
            # Include v
            # New P is P intersect N_G_comp(v)
            # But we must only consider vertices that come AFTER v in the sorted order?
            # In Tomita, we maintain the set of candidates.
            # The standard implementation passes the subset of P that is "after" v in the ordering.
            # Let's reconstruct P to be the subset of vertices in P that are adjacent to v in G_comp
            # AND have index > v in the sorted order?
            # Actually, the standard recursive step:
            # expand(R U {v}, P U {v} intersect N(v))
            # But to avoid duplicates and ensure we cover all cliques,
            # we should restrict the new P to vertices that were "after" v in the coloring order.
            
            # Let's create a mask of vertices in P that are adjacent to v in G_comp
            # AND are in the "remaining" part of the list.
            
            # It's easier to just pass the whole P intersection N(v), but we must ensure we don't revisit.
            # The coloring order provides a permutation.
            # Let's keep track of the position.
            
            # Alternative: Just pass P_new = P & adj_comp[v]
            # And remove v from P for the next iteration?
            # Yes.
            
            P_new = P & adj_comp[v]
            R_new = R | (1 << v)
            
            expand(R_new, P_new)
            
            # Remove v from P for the next iteration (exclude v)
            P &= ~(1 << v)
            
            # Update the coloring?
            # In Tomita, the coloring is updated dynamically.
            # However, using the initial coloring of P for the entire loop is a valid
            # (though potentially looser) bound if we are careful.
            # Actually, the bound `c_val` is specific to the vertex v in the *original* coloring of P.
            # As we remove vertices from P, the coloring might change, but the bound remains valid
            # (it's an upper bound).
            # However, if we remove vertices, `c_val` for remaining vertices might be too high?
            # No, the color of a vertex in the subgraph is <= its color in the supergraph.
            # So using the original colors is safe for pruning (it might prune less effectively, but it's correct).
            
            # Wait, if I remove a vertex, the max color of the remaining set might be lower.
            # But `c_val` is the color of the *current* vertex v.
            # The condition `R.bit_count() + c_val <= best` uses the color of v.
            # If v has a high color, it means it was hard to color, implying it's in a dense part?
            # Actually, high color means it was assigned late, so it has many neighbors in previous color classes?
            # No, it means it couldn't fit in earlier classes.
            # Regardless, it is a valid upper bound for the clique size extending from v?
            # Yes, the clique containing v can have at most `c_val` vertices from the set of vertices with color >= c_val?
            # No, the clique size is bounded by the number of colors.
            # The standard Tomita bound is:
            # If |R| + color(v) <= best, prune.
            # This is valid because any clique extending R and containing v can include at most
            # one vertex from each color class 1..color(v).
            
            # So the logic holds.

    # Start with P = all vertices
    P_init = (1 << (n + 1)) - 2 # bits 1..n
    R_init = 0
    
    # Initial heuristic to set a good lower bound
    # Greedy MIS on G (which is Greedy Clique on G_comp)
    # Let's just run expand.
    
    # To improve the initial bound, we can run a greedy clique finder on G_comp.
    def greedy_clique(P):
        C = 0
        U = P
        while U:
            # Pick vertex with max degree in U
            # Or just pick one
            v_bit = U & -U
            v = v_bit.bit_length() - 1
            C |= v_bit
            U &= adj_comp[v] # Intersect with neighbors in G_comp
            U &= ~v_bit
        return C

    # Get initial solution
    init_clique = greedy_clique(P_init)
    best_clique_size = init_clique.bit_count()
    best_clique_mask = init_clique
    
    # Run Tomita
    expand(R_init, P_init)
    
    # Extract the list of vertices from the best clique mask
    result = []
    m = best_clique_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
        
    return result

# Quick sanity check
if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a = res[i]
            b = res[j]
            if (a*b) % (a+b) == 0:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
