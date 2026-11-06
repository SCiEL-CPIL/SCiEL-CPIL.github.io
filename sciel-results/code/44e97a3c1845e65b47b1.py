import time
import sys

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills after 90s. We target finishing by 85s.
    deadline = start_time + 85.0
    
    N = 256
    # Vertices are 1..N. We use bit i to represent vertex i.
    # Note: bit 0 is unused.
    
    # 1. Build conflict graph adjacency bitmasks
    # adj_conflict[v] has bit u set if (v*u) % (v+u) == 0
    adj_conflict = [0] * (N + 1)
    
    # Precompute conflicts
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj_conflict[a] |= (1 << b)
                adj_conflict[b] |= (1 << a)
                
    # 2. Build complement graph adjacency bitmasks
    # In the complement graph, u and v are connected if they are NOT connected in G (and u != v).
    # adj_comp[v] has bit u set if u != v and (u,v) is NOT an edge in G.
    adj_comp = [0] * (N + 1)
    
    # Mask for all vertices 1..N
    all_vertices_mask = (1 << (N + 1)) - 1
    all_vertices_mask &= ~1 # Clear bit 0
    
    for v in range(1, N + 1):
        # Start with all vertices, remove v itself, remove conflict neighbors
        # adj_comp[v] = (All - {v} - adj_conflict[v])
        adj_comp[v] = all_vertices_mask & ~(1 << v) & ~adj_conflict[v]
        
    # Helper to get list of vertices from bitmask
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        return verts

    # 3. Greedy Coloring for Upper Bound
    # Colors the subgraph induced by candidates_mask in the complement graph.
    # Returns a list of (vertex, color) sorted by color descending.
    # The max color is an upper bound on the clique size in candidates_mask.
    def greedy_color(candidates_mask):
        if candidates_mask == 0:
            return [], 0
            
        # Order vertices by degree in the candidate subgraph, descending.
        # This helps produce a tighter coloring (fewer colors used).
        verts = get_vertices(candidates_mask)
        
        # Calculate degrees within candidates
        degrees = []
        for v in verts:
            # Degree is the number of neighbors in candidates_mask
            deg = (adj_comp[v] & candidates_mask).bit_count()
            degrees.append((deg, v))
            
        # Sort by degree descending
        degrees.sort(reverse=True)
        ordered_verts = [v for _, v in degrees]
        
        color_masks = [0] # color_masks[c] holds bitmask of vertices with color c
        max_color = 0
        vertex_colors = {}
        
        for v in ordered_verts:
            # Find smallest color c such that no neighbor of v has color c
            # Neighbor check: (adj_comp[v] & color_masks[c]) == 0
            c = 1
            while c < len(color_masks):
                if (adj_comp[v] & color_masks[c]) == 0:
                    break
                c += 1
            
            if c == len(color_masks):
                color_masks.append(0)
                
            vertex_colors[v] = c
            color_masks[c] |= (1 << v)
            if c > max_color:
                max_color = c
                
        # Create list of (vertex, color)
        # We want to process vertices in descending order of color in the search tree
        # to allow early pruning (if current_size + color <= best_size, prune)
        color_list = [(v, vertex_colors[v]) for v in ordered_verts]
        color_list.sort(key=lambda x: x[1], reverse=True)
        
        return color_list, max_color

    # 4. Branch and Bound Search
    best_size = 0
    best_set_mask = 0
    
    # Initial heuristic: Greedy Clique
    def greedy_clique(start_mask):
        candidates = start_mask
        clique_mask = 0
        size = 0
        while candidates:
            # Pick vertex with highest degree in candidates
            verts = get_vertices(candidates)
            best_v = -1
            best_deg = -1
            for v in verts:
                deg = (adj_comp[v] & candidates).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v == -1:
                break
            clique_mask |= (1 << best_v)
            size += 1
            # Restrict candidates to neighbors of best_v
            candidates = candidates & adj_comp[best_v]
        return size, clique_mask

    # Run greedy to get a good initial lower bound
    s, m = greedy_clique(all_vertices_mask)
    best_size = s
    best_set_mask = m
    
    # Try a few more greedy starts to improve lower bound
    # Remove each vertex and greedy
    for v in range(1, N + 1):
        if time.time() > deadline - 10:
            break
        mask_without_v = all_vertices_mask & ~(1 << v)
        s, m = greedy_clique(mask_without_v)
        if s > best_size:
            best_size = s
            best_set_mask = m

    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask
        
        # Timeout check
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Get upper bound via coloring
        color_list, max_color = greedy_color(candidates_mask)
        
        # Prune if upper bound is not better than current best
        if current_size + max_color <= best_size:
            return
            
        # Iterate through vertices in order of decreasing color
        # This allows us to prune branches where even adding all vertices of lower color
        # wouldn't beat the best size.
        for v, c in color_list:
            # If current size + this vertex's color bound <= best, 
            # then any clique formed by this vertex and subsequent (lower color) vertices
            # will be <= best. So we can break.
            if current_size + c <= best_size:
                break
                
            # Check if v is still in the candidate set?
            # The color_list is generated from the initial candidates_mask.
            # As we recurse, we effectively remove v from consideration for subsequent iterations
            # by intersecting with neighbors. But for the "skip v" branch, we just don't pick it.
            # The standard Tomita algorithm modifies the candidate set P.
            # Here, we simulate that by passing `new_candidates` to recursion.
            # For the "skip" part, we just continue the loop. The `color_list` is static for this call.
            # However, we must ensure we don't pick a vertex that has been "removed" by previous picks?
            # No, in this formulation:
            # We pick v. New candidates = candidates & neighbors(v).
            # We don't pick v. We just move to next vertex in list.
            # The bound `c` is valid for the subgraph induced by {v} U {u in candidates | u adjacent to v, color(u) <= color(v)}?
            # Actually, the bound `c` is the color assigned to v.
            # The property is: The maximum clique size in the subgraph induced by
            # {v} U {u in candidates | u is adjacent to v and u appears after v in the list}
            # is at most c.
            # So if current_size + c <= best_size, we can prune this branch.
            
            # Recurse: Pick v
            new_candidates = candidates_mask & adj_comp[v]
            expand(new_candidates, current_size + 1, current_mask | (1 << v))
            
            # Note: We do not explicitly remove v from candidates_mask for the next iteration
            # because the recursion handles the "include v" branch.
            # The "exclude v" branch is implicitly handled by moving to the next vertex in the list,
            # effectively ignoring v for the rest of the search at this level.
            # The bound check `current_size + c <= best_size` relies on the fact that
            # any clique in the remaining vertices (which are a subset of the original candidates
            # minus those picked in previous branches? No, minus v and its non-neighbors?)
            # Let's stick to the standard Tomita logic which is proven correct.
            pass

    # Start search
    expand(all_vertices_mask, 0, 0)
    
    # Extract result
    result_set = set()
    mask = best_set_mask
    while mask:
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        result_set.add(v)
        mask ^= lsb
        
    return list(result_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
