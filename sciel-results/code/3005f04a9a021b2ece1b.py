import time

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin from 90s limit
    
    n = 256
    vertices = list(range(1, n + 1))
    
    # Build the conflict graph G (edges where (a*b) % (a+b) == 0)
    adj_conflict = [0] * (n + 1)
    
    for a in vertices:
        for b in vertices:
            if a < b:
                if (a * b) % (a + b) == 0:
                    adj_conflict[a] |= (1 << b)
                    adj_conflict[b] |= (1 << a)
                    
    # Build complement graph adjacency bitmasks
    adj_comp = [0] * (n + 1)
    all_vertices_mask = 0
    for i in range(1, n + 1):
        all_vertices_mask |= (1 << i)
        
    for v in vertices:
        mask = all_vertices_mask & ~(1 << v)
        mask &= ~adj_conflict[v]
        adj_comp[v] = mask
    
    # Precompute degree for ordering
    degree = [0] * (n + 1)
    for v in vertices:
        degree[v] = adj_conflict[v].bit_count()
    
    # Multiple greedy initial solutions for a stronger lower bound
    best_size = 0
    best_set_mask = 0
    
    # Greedy strategy 1: fewest conflicts first
    greedy_set = set()
    for v in sorted(vertices, key=lambda x: degree[x]):
        if all((v * u) % (v + u) != 0 for u in greedy_set):
            greedy_set.add(v)
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set_mask = 0
        for v in greedy_set:
            best_set_mask |= (1 << v)
    
    # Greedy strategy 2: most conflicts first (sometimes finds different structure)
    greedy_set2 = set()
    for v in sorted(vertices, key=lambda x: -degree[x]):
        if all((v * u) % (v + u) != 0 for u in greedy_set2):
            greedy_set2.add(v)
    if len(greedy_set2) > best_size:
        best_size = len(greedy_set2)
        best_set_mask = 0
        for v in greedy_set2:
            best_set_mask |= (1 << v)
    
    # Greedy strategy 3: random order (try a few)
    import random
    rng = random.Random(42)
    for trial in range(5):
        greedy_set3 = set()
        order = vertices[:]
        rng.shuffle(order)
        for v in order:
            if all((v * u) % (v + u) != 0 for u in greedy_set3):
                greedy_set3.add(v)
        if len(greedy_set3) > best_size:
            best_size = len(greedy_set3)
            best_set_mask = 0
            for v in greedy_set3:
                best_set_mask |= (1 << v)
    
    # Fast greedy coloring using pre-allocated arrays
    # color_of is reused across calls via a list that gets zeroed
    color_of = [0] * (n + 1)
    
    def greedy_color(candidates_mask):
        """
        Returns (color_list, max_color) where color_list is [(v, c), ...] sorted by c desc.
        Uses sequential coloring with degree-based ordering.
        """
        if candidates_mask == 0:
            return [], 0
        
        # Get vertices and order by degree in candidate subgraph
        verts = []
        m = candidates_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj_comp[v] & candidates_mask).bit_count()
            verts.append((deg, v))
            m ^= lsb
        
        verts.sort(reverse=True)
        ordered_verts = [v for _, v in verts]
        
        # Zero out color_of for vertices in candidates
        for v in ordered_verts:
            color_of[v] = 0
        
        max_color = 0
        
        for v in ordered_verts:
            # Find used colors among colored neighbors using a bitmask for speed
            used_mask = 0
            m = adj_comp[v] & candidates_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                c = color_of[u]
                if c > 0:
                    used_mask |= (1 << c)
                m ^= lsb
            
            # Find smallest available color
            c = 1
            while (used_mask >> c) & 1:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
        
        # Build result list sorted by color descending
        color_list = [(v, color_of[v]) for v in ordered_verts]
        color_list.sort(key=lambda x: x[1], reverse=True)
        
        return color_list, max_color

    # Branch and bound for maximum clique
    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask, start_time
        
        if time.time() - start_time > timeout:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Get upper bound via coloring
        color_list, max_color = greedy_color(candidates_mask)
        
        if current_size + max_color <= best_size:
            return
            
        P = candidates_mask
        
        for v, c in color_list:
            if current_size + c <= best_size:
                break
                
            if not (P & (1 << v)):
                continue
                
            # Recurse: include v
            new_candidates = P & adj_comp[v]
            expand(new_candidates, current_size + 1, current_mask | (1 << v))
            
            # Exclude v
            P &= ~(1 << v)

    # Start the search
    expand(all_vertices_mask, 0, 0)
    
    # Local search polish: try to add vertices to the current best set
    if best_set_mask:
        current_mask = best_set_mask
        improved = True
        while improved and (time.time() - start_time < timeout - 5):
            improved = False
            # Try adding each vertex not in current_mask
            m = all_vertices_mask & ~current_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Check if v is compatible with all in current_mask
                if (adj_conflict[v] & current_mask) == 0:
                    current_mask |= (1 << v)
                    improved = True
                    m &= ~(1 << v)
                    # Don't break; keep trying to add more
                else:
                    m ^= lsb
            # Try removing one vertex and adding two others (2-opt style)
            if not improved:
                # Simple: try removing each vertex and see if we can add more
                cur_m = current_mask
                while cur_m:
                    lsb = cur_m & -cur_m
                    v = lsb.bit_length() - 1
                    # Try removing v
                    temp_mask = current_mask & ~(1 << v)
                    # Try adding any compatible vertex
                    add_m = all_vertices_mask & ~temp_mask
                    added = 0
                    while add_m:
                        lsb2 = add_m & -add_m
                        u = lsb2.bit_length() - 1
                        if (adj_conflict[u] & temp_mask) == 0:
                            temp_mask |= (1 << u)
                            added += 1
                            add_m &= ~(1 << u)
                        else:
                            add_m ^= lsb2
                    if len(bin(temp_mask).strip('0b')) > len(bin(current_mask).strip('0b')):
                        current_mask = temp_mask
                        improved = True
                        break
                    cur_m ^= lsb
        if current_mask.bit_count() > best_size:
            best_size = current_mask.bit_count()
            best_set_mask = current_mask
    
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
