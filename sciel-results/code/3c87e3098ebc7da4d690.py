import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin from 90s limit

    n = 256
    vertices = range(1, n + 1)

    # 1. Build the conflict graph G (edges where (a*b) % (a+b) == 0)
    # We will solve for Maximum Independent Set in G.
    # This is equivalent to Maximum Clique in G_complement.
    
    # adjacency bitmask for G (conflicts)
    adj_conflict = [0] * (n + 1)
    
    for a in vertices:
        for b in vertices:
            if a < b:
                if (a * b) % (a + b) == 0:
                    adj_conflict[a] |= (1 << b)
                    adj_conflict[b] |= (1 << a)
                    
    # Build complement graph adjacency bitmasks
    # A vertex u is a neighbor of v in G_comp if u != v and u is NOT a neighbor of v in G.
    adj_comp = [0] * (n + 1)
    all_vertices_mask = 0
    for i in range(1, n + 1):
        all_vertices_mask |= (1 << i)
        
    for v in vertices:
        # Start with all vertices, remove v itself, remove conflict neighbors
        mask = all_vertices_mask & ~(1 << v)
        mask &= ~adj_conflict[v]
        adj_comp[v] = mask
        
    # Tomita Algorithm for Maximum Clique
    # We use bitmasks.
    best_size = 0
    best_set_mask = 0

    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        return verts

    # Greedy coloring for upper bound
    def greedy_color(candidates_mask):
        colors = {}
        verts = get_vertices(candidates_mask)
        
        # Optimization: Order by degree in the candidate subgraph (descending)
        # This usually leads to better coloring bounds
        degrees = []
        for v in verts:
            deg = (adj_comp[v] & candidates_mask).bit_count()
            degrees.append((deg, v))
            
        degrees.sort(reverse=True)
        ordered_verts = [v for _, v in degrees]
        
        max_color = 0
        colored_mask = 0
        
        for v in ordered_verts:
            used_colors = set()
            m = adj_comp[v] & colored_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                used_colors.add(colors[u])
                m ^= lsb
                    
            c = 1
            while c in used_colors:
                c += 1
                
            colors[v] = c
            if c > max_color:
                max_color = c
            colored_mask |= (1 << v)
            
        color_list = [(v, colors[v]) for v in ordered_verts]
        color_list.sort(key=lambda x: x[1], reverse=True)
        
        return color_list, max_color

    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask, start_time
        
        if time.time() - start_time > timeout:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        color_list, max_color = greedy_color(candidates_mask)
        
        if current_size + max_color <= best_size:
            return
            
        P = candidates_mask
        
        for v, c in color_list:
            if current_size + c <= best_size:
                break
                
            if not (P & (1 << v)):
                continue
                
            new_candidates = P & adj_comp[v]
            expand(new_candidates, current_size + 1, current_mask | (1 << v))
            
            P &= ~(1 << v)

    # --- Hybrid Strategy: Start with a good heuristic solution to prune early ---
    
    # 1. Greedy Independent Set on Conflict Graph (equivalent to Clique on Comp Graph)
    # We pick vertices from the conflict graph that have the fewest conflicts remaining?
    # Actually, for MIS on G, we want to pick vertices with low degree in G.
    # Let's implement a simple greedy for MIS on G (conflict graph).
    
    def greedy_mis():
        remaining_mask = all_vertices_mask
        result = []
        
        while remaining_mask:
            if time.time() - start_time > timeout - 5:
                break
                
            # Find vertex with minimum degree in the induced subgraph of 'remaining'
            min_v = None
            min_deg = float('inf')
            
            m = remaining_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (adj_conflict[v] & remaining_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                m ^= lsb
                    
            result.append(min_v)
            
            # Remove min_v and its neighbors from 'remaining'
            remaining_mask &= ~(adj_conflict[min_v] | (1 << min_v))
            
        return set(result)

    # Get initial solution
    init_sol = greedy_mis()
    init_size = len(init_sol)
    
    # Update best
    if init_size > best_size:
        best_size = init_size
        # Construct mask
        mask = 0
        for v in init_sol:
            mask |= (1 << v)
        best_set_mask = mask

    # Now run the exact Tomita solver, which will start with a strong lower bound
    initial_candidates = all_vertices_mask
    expand(initial_candidates, 0, 0)
    
    # If the exact solver finished and found something bigger, use that.
    # Otherwise, we already have the greedy one stored in best_set_mask.
    
    # Optional: Local Search Polish on the best set found
    # If we have time, try to improve the current best_set_mask
    if time.time() - start_time < timeout - 2:
        current_set = set()
        mask = best_set_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            current_set.add(v)
            mask ^= lsb
        
        def update_best():
            nonlocal best_size, best_set_mask
            if len(current_set) > best_size:
                best_size = len(current_set)
                new_mask = 0
                for x in current_set:
                    new_mask |= (1 << x)
                best_set_mask = new_mask
        
        def can_add(v, s):
            """Check if v can be added to set s (no conflicts in G)"""
            m = adj_conflict[v]
            for u in s:
                if (m >> u) & 1:
                    return False
            return True
        
        improved = True
        while improved and time.time() - start_time < timeout - 1:
            improved = False
            
            # Phase 1: Try adding any vertex not in set
            for v in range(1, n + 1):
                if v in current_set:
                    continue
                if can_add(v, current_set):
                    current_set.add(v)
                    improved = True
                    update_best()
            
            # Phase 2: Try swaps - remove one vertex, try to add two
            if time.time() - start_time > timeout - 1:
                break
            for v_remove in list(current_set):
                if time.time() - start_time > timeout - 1:
                    break
                current_set.discard(v_remove)
                added = 0
                for v_add1 in range(1, n + 1):
                    if v_add1 in current_set or v_add1 == v_remove:
                        continue
                    if can_add(v_add1, current_set):
                        current_set.add(v_add1)
                        added += 1
                        # Try to add a second one
                        for v_add2 in range(1, n + 1):
                            if v_add2 in current_set or v_add2 == v_remove or v_add2 == v_add1:
                                continue
                            if can_add(v_add2, current_set):
                                current_set.add(v_add2)
                                added += 1
                                break
                        break
                if added > 1:
                    improved = True
                    update_best()
                elif added == 1:
                    # We added one, net zero change in size but maybe better structure
                    update_best()
                else:
                    # Undo: put v_remove back
                    current_set.add(v_remove)
        
        update_best()
        
    result_set = set()
    mask = best_set_mask
    while mask:
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        result_set.add(v)
        mask ^= lsb
        
    if not result_set:
        # Fallback
        return [1]

    return list(result_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
