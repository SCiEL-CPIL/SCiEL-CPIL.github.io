import time
import sys

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    n = 256
    
    # 1. Build Graph
    # Edge (a,b) if (a*b) % (a+b) == 0
    # We want to find a set S such that for all distinct a,b in S, (a*b) % (a+b) != 0.
    # This is equivalent to finding a Maximum Independent Set in the graph G where edges represent conflicts.
    # Or a Maximum Clique in the complement graph G'.
    
    # adj_mask[v] will have bit u set if (v, u) is an edge in G (conflict).
    adj_mask = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj_mask[a] |= (1 << (b - 1))
                adj_mask[b] |= (1 << (a - 1))
                
    # Complement graph for Max Clique
    # comp_adj_mask[v] has bit u set if (v, u) is NOT an edge in G (and u != v).
    full_mask = (1 << n) - 1
    comp_adj_mask = [0] * (n + 1)
    for v in range(1, n + 1):
        # full_mask ^ adj_mask[v] gives bits that are NOT in adj_mask[v].
        # We must exclude v itself from its own adjacency in the complement (self-loops are not allowed in simple graphs for clique).
        # adj_mask[v] does not have bit v-1 set (since we only added pairs a<b).
        # So full_mask ^ adj_mask[v] has bit v-1 set. We must clear it.
        comp_adj_mask[v] = (full_mask ^ adj_mask[v]) & ~(1 << (v - 1))
    
    # 2. Initial greedy solution to get a good lower bound
    # Greedy: sort by degree in G (ascending) to pick vertices with fewest conflicts first.
    # This is a heuristic for Maximum Independent Set.
    
    conflict_deg = [0] * (n + 1)
    for v in range(1, n + 1):
        conflict_deg[v] = adj_mask[v].bit_count()
    
    init_set = []
    init_mask = 0
    # Sort by conflict degree ascending
    vertices_sorted = sorted(range(1, n + 1), key=lambda x: conflict_deg[x])
    
    for v in vertices_sorted:
        # Check if v can be added to init_set
        # v can be added if it has no conflict with any vertex in init_set
        # i.e., adj_mask[v] & init_mask == 0
        if (adj_mask[v] & init_mask) == 0:
            init_set.append(v)
            init_mask |= (1 << (v - 1))
    
    best_size = len(init_set)
    best_set = init_set[:]
    
    # Local search to improve initial solution: Try swapping one out, two in? 
    # Or just try adding more? The greedy above is maximal.
    # Let's try a simple 1-move local search: remove v, see if we can add 2 new ones.
    # Given time constraints, we'll rely on the B&B to find the optimum, but a better LB helps.
    
    # 3. Tomita's Branch and Bound for Maximum Clique on complement graph
    
    # We will use a recursive approach with explicit stack or recursion.
    # Recursion depth is at most n=256, which is fine if we increase recursion limit.
    sys.setrecursionlimit(10000)
    
    # Helper to extract vertices from a mask, optionally filtered by a condition or sorted.
    # For Tomita, we need to order vertices by color.
    
    def expand(P_mask, clique_size, clique):
        nonlocal best_size, best_set
        
        if time.time() - start_time > timeout:
            return
            
        if P_mask == 0:
            if clique_size > best_size:
                best_size = clique_size
                best_set = clique[:]
            return
            
        # Greedy Coloring of P_mask
        # We need to assign colors to vertices in P_mask such that adjacent vertices in G' have different colors.
        # This gives an upper bound on the clique size: max_color.
        # We also need to order vertices by color for branching.
        
        # To make coloring fast, we can process vertices in a specific order.
        # Tomita suggests ordering by degree in the induced subgraph, or just arbitrary.
        # Let's extract vertices from P_mask.
        
        verts = []
        temp_mask = P_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length()
            verts.append(v)
            temp_mask ^= lsb
            
        # Sort vertices by degree in G' within P_mask? 
        # Degree of v in P is popcount(comp_adj_mask[v] & P_mask).
        # This helps the greedy coloring produce tighter bounds (fewer colors).
        verts.sort(key=lambda v: (comp_adj_mask[v] & P_mask).bit_count(), reverse=True)
        
        color_of = [0] * (n + 1)
        # We store (vertex, color) in a list to iterate in sorted order
        v_colors = []
        colored_mask = 0
        max_color = 0
        
        for v in verts:
            # Find colors of neighbors of v in G' that are already colored
            neighbor_mask = comp_adj_mask[v] & colored_mask
            used_colors = set()
            temp = neighbor_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length()
                used_colors.add(color_of[u])
                temp ^= lsb
                
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
            v_colors.append((v, c))
            colored_mask |= (1 << (v - 1))
            
        # Sort by color ascending. 
        # Vertices with lower colors are "more constrained" or rather, the bound is tighter for higher indices?
        # In Tomita, we branch on vertices in order of increasing color? 
        # Actually, the standard is:
        # Order vertices v1, ..., vk such that color(v1) <= color(v2) <= ...
        # Then for each vi, if clique_size + 1 + (max_color - color(vi)) <= best_size, we can prune?
        # Wait, the bound is: The max clique in the remaining set (after vi) is at most max_color - color(vi) + 1?
        # No, the bound is simply: if we pick vi, we can pick at most (max_color - color(vi)) more vertices?
        # Let's verify.
        # If we have colored P with k colors.
        # A clique can contain at most one vertex of each color.
        # If we pick vi (color c), we cannot pick any other vertex of color c.
        # The remaining vertices have colors >= c (if we process in increasing color order? No, we process in the order of the list).
        # The standard Tomita algorithm orders vertices by color.
        # It iterates through the vertices.
        # If we are at vertex vi with color c, the maximum number of vertices we can add from the *remaining* vertices in the list (which have colors >= c if sorted) is bounded by the number of colors available?
        # Actually, the bound used is: `clique_size + 1 + (max_color - c) <= best_size` -> prune?
        # Or `clique_size + (max_color - c + 1) <= best_size`?
        
        # Let's use the standard:
        # Sort v_colors by color ascending.
        # Iterate i from 0 to len(v_colors)-1.
        # Let v = v_colors[i][0], c = v_colors[i][1].
        # The remaining candidates for the next step are neighbors of v in G' that are in P_mask and have index > i in this sorted list?
        # No, the set of candidates is simply P_mask intersected with N(v).
        # But to avoid re-exploring, we restrict to vertices that appear *after* v in the sorted order?
        # Yes, this is a standard optimization to ensure canonical ordering.
        
        # So, new_P_mask = comp_adj_mask[v] & (mask of vertices in v_colors[i+1:])
        
        # Precompute suffix masks for v_colors
        m = len(v_colors)
        suffix_masks = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            v_i = v_colors[i][0]
            suffix_masks[i] = suffix_masks[i+1] | (1 << (v_i - 1))
            
        for i in range(m):
            if time.time() - start_time > timeout:
                return
            
            v, c = v_colors[i]
            
            # Bound check
            # If we include v, we gain 1.
            # The remaining vertices we can potentially add are from suffix_masks[i+1] intersected with N(v).
            # The max clique size in any subset of P is bounded by max_color.
            # More specifically, if we have already "used" colors 1..c-1 (by excluding or including vertices with those colors? No, we are branching).
            # The bound is: `clique_size + 1 + (max_color - c) <= best_size`?
            # Let's check. If we pick v (color c), we cannot pick any other vertex of color c.
            # The remaining vertices have colors >= c (since we sorted by color).
            # So we can pick at most one vertex of each color from c to max_color.
            # That is max_color - c + 1 vertices total from the remaining set?
            # Wait, if we pick v, we use color c. So we can pick at most one vertex of each color in {c+1, ..., max_color}.
            # That is max_color - c vertices.
            # So total size <= clique_size + 1 + (max_color - c).
            
            if clique_size + 1 + (max_color - c) <= best_size:
                # Since colors are non-decreasing, for any j > i, c_j >= c.
                # So max_color - c_j <= max_color - c.
                # The bound will only get tighter (smaller).
                # So we can break the loop.
                break
            
            # Branch Include v
            new_P_mask = comp_adj_mask[v] & suffix_masks[i+1]
            
            if new_P_mask:
                expand(new_P_mask, clique_size + 1, clique + [v])
            else:
                if clique_size + 1 > best_size:
                    best_size = clique_size + 1
                    best_set = clique + [v]
                    
    # Initial P: all vertices
    initial_P_mask = full_mask
    expand(initial_P_mask, 0, [])
    
    if not best_set:
        best_set = [1]
        
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
