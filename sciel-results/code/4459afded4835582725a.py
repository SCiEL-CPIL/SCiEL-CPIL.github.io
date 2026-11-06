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
        remaining = set(range(1, n + 1))
        result = []
        
        # Precompute degrees? Or compute on the fly.
        # Since n=256, on-the-fly is fine.
        
        while remaining:
            if time.time() - start_time > timeout - 5:
                break
                
            # Find vertex with minimum degree in the induced subgraph of 'remaining'
            min_v = None
            min_deg = float('inf')
            
            # Optimization: Instead of checking all remaining vertices against all others,
            # we can just pick one. But min-degree is a standard heuristic.
            # To speed up, we can approximate or just pick random if too slow.
            # Let's stick to min-degree for quality.
            
            for v in remaining:
                deg = 0
                m = adj_conflict[v]
                # Count bits in m that are in 'remaining'
                # 'remaining' is a set, so checking membership is O(1) but iterating is O(n^2) total.
                # 256^2 = 65536 ops. This is very fast.
                for u in remaining:
                    if (m >> u) & 1:
                        deg += 1
                
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                    
            result.append(min_v)
            
            # Remove min_v and its neighbors from 'remaining'
            to_remove = {min_v}
            m = adj_conflict[min_v]
            for u in remaining:
                if (m >> u) & 1:
                    to_remove.add(u)
            
            remaining -= to_remove
            
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
            
        # Simple local search: try to add a vertex if possible, or swap
        # For MIS on G: A vertex v can be added if it has no neighbors in current_set (in G).
        # i.e., v is not in G-neighborhood of any u in current_set.
        
        improved = True
        while improved and time.time() - start_time < timeout - 1:
            improved = False
            # Try adding any vertex not in set
            for v in range(1, n + 1):
                if v in current_set:
                    continue
                # Check if v conflicts with any in current_set
                conflicts = False
                m = adj_conflict[v]
                for u in current_set:
                    if (m >> u) & 1:
                        conflicts = True
                        break
                
                if not conflicts:
                    current_set.add(v)
                    improved = True
                    # Restart loop? Or continue?
                    # Since adding one might open up no new possibilities for other additions 
                    # (it only adds constraints), we can just continue.
                    # But we need to update best_size and mask
                    best_size = len(current_set)
                    new_mask = 0
                    for x in current_set:
                        new_mask |= (1 << x)
                    best_set_mask = new_mask
                    # break to restart the for loop? No, just continue checking others.
                    # Actually, adding v removes potential candidates. 
                    # So we should probably restart the scan or just continue.
                    # Continuing is fine, we just won't add conflicting ones.
        
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
