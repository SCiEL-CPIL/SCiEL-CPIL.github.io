import time
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        # Generate graph
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict_masks(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        m = len(edges)
        # Map edge index to vertices
        edge_verts = [(u, v) for u, v in edges]
        
        conflict_masks = [0] * m
        
        for i in range(m):
            u, v = edge_verts[i]
            # Vertices in the closed neighborhood of edge (u,v)
            # S = {u, v} U N(u) U N(v)
            S = {u, v}
            S.update(adj[u])
            S.update(adj[v])
            
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                x, y = edge_verts[j]
                if x in S or y in S:
                    mask |= (1 << j)
            conflict_masks[i] = mask
            
        return conflict_masks

    def dsatur_coloring(conflict_masks, rng=None):
        m = len(conflict_masks)
        if m == 0:
            return []
            
        color = [-1] * m
        # Saturation degree: number of distinct colors in neighbors
        sat_deg = [0] * m
        # Set of colors used by neighbors
        neighbor_colors = [set() for _ in range(m)]
        
        uncoloured = list(range(m))
        degrees = [conflict_masks[i].bit_count() for i in range(m)]
        
        if rng is None:
            rng = Random(0)
            
        while uncoloured:
            # Find vertex with max saturation, then max degree
            best = -1
            best_sat = -1
            best_deg = -1
            for v in uncoloured:
                s = len(neighbor_colors[v])
                if s > best_sat or (s == best_sat and degrees[v] > best_deg):
                    best = v
                    best_sat = s
                    best_deg = degrees[v]
            
            uncoloured.remove(best)
            
            # Find smallest available color
            used = neighbor_colors[best]
            c = 0
            while c in used:
                c += 1
                
            color[best] = c
            
            # Update neighbors
            mask = conflict_masks[best]
            # Iterate set bits in mask
            temp_mask = mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                nb = lsb.bit_length() - 1
                temp_mask ^= lsb
                
                if color[nb] == -1:
                    neighbor_colors[nb].add(c)
                    
        return color

    def compress_colors(color):
        """Remap colors to 0..k-1"""
        if not color:
            return color, 0
        mapping = {}
        new_color = [0] * len(color)
        next_c = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = next_c
                next_c += 1
            new_color[i] = mapping[c]
        return new_color, next_c

    def can_merge(color_groups, conflict_masks):
        """
        Check if any two color groups can be merged.
        Returns (i, j) if mergeable, else None.
        color_groups: dict {color_idx: [list of vertices]}
        """
        colors = list(color_groups.keys())
        for idx_a in range(len(colors)):
            ca = colors[idx_a]
            group_a = color_groups[ca]
            mask_a = 0
            for v in group_a:
                mask_a |= (1 << v)
                
            for idx_b in range(idx_a + 1, len(colors)):
                cb = colors[idx_b]
                group_b = color_groups[cb]
                
                # Check if any vertex in group_a conflicts with any vertex in group_b
                # We can check: for each u in group_a, does (conflict_masks[u] & mask_b) != 0?
                # To speed up, we can precompute mask_b?
                mask_b = 0
                for v in group_b:
                    mask_b |= (1 << v)
                    
                mergeable = True
                for u in group_a:
                    if conflict_masks[u] & mask_b:
                        mergeable = False
                        break
                        
                if mergeable:
                    return ca, cb
        return None

    def try_merge_pass(color, conflict_masks):
        """
        Repeatedly merge color classes until no more merges are possible.
        Returns new color list and number of colors.
        """
        color, num_colors = compress_colors(color)
        while True:
            groups = defaultdict(list)
            for i, c in enumerate(color):
                groups[c].append(i)
                
            merge_pair = can_merge(groups, conflict_masks)
            if merge_pair is None:
                break
                
            ca, cb = merge_pair
            # Merge cb into ca
            for i, c in enumerate(color):
                if c == cb:
                    color[i] = ca
            
            # Compress again to keep indices clean
            color, num_colors = compress_colors(color)
            
        return color, num_colors

    def local_search_improve(color, conflict_masks, rng, time_limit):
        """
        Local search using moves and swaps to enable merges.
        """
        color, num_colors = compress_colors(color)
        best_color = color[:]
        best_nc = num_colors
        m = len(color)
        
        # Precompute conflict masks for fast checking
        # We need to check if a move is valid:
        # Move v from c_old to c_new is valid if no neighbor of v has color c_new.
        
        while time.time() < time_limit:
            improved = False
            # Randomly select a vertex to move
            v = rng.randrange(m)
            c_old = color[v]
            
            # Try moving to any other color
            # Shuffle colors to try them in random order
            other_colors = [c for c in range(num_colors) if c != c_old]
            rng.shuffle(other_colors)
            
            moved = False
            for c_new in other_colors:
                # Check validity: no neighbor of v has color c_new
                mask = conflict_masks[v]
                # Get all vertices with color c_new
                # This is slow if we scan all m.
                # Optimization: Maintain color_to_vertices?
                # For simplicity in this rewrite, we scan. m is small (<300).
                valid = True
                temp_mask = mask
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    nb = lsb.bit_length() - 1
                    temp_mask ^= lsb
                    if color[nb] == c_new:
                        valid = False
                        break
                if valid:
                    color[v] = c_new
                    moved = True
                    break
            
            if not moved:
                # Try swapping with a random vertex
                u = rng.randrange(m)
                if u == v or color[u] == c_old:
                    continue
                c_u = color[u]
                
                # Check if swap is valid
                # v can take c_u? (No neighbor of v has c_u)
                # u can take c_old? (No neighbor of u has c_old)
                # Note: v and u might be neighbors.
                
                # Check v -> c_u
                valid_v = True
                mask_v = conflict_masks[v]
                temp = mask_v
                while temp:
                    lsb = temp & -temp
                    nb = lsb.bit_length() - 1
                    temp ^= lsb
                    if nb == u:
                        # u has color c_u, so if v is neighbor of u, v cannot take c_u?
                        # Wait, if v and u are neighbors, and u has c_u, then v cannot take c_u.
                        # So if nb == u, valid_v = False.
                        valid_v = False
                        break
                    if color[nb] == c_u:
                        valid_v = False
                        break
                        
                if valid_v:
                    # Check u -> c_old
                    valid_u = True
                    mask_u = conflict_masks[u]
                    temp = mask_u
                    while temp:
                        lsb = temp & -temp
                        nb = lsb.bit_length() - 1
                        temp ^= lsb
                        if nb == v:
                            valid_u = False
                            break
                        if color[nb] == c_old:
                            valid_u = False
                            break
                    
                    if valid_u:
                        color[v], color[u] = color[u], color[v]
                        moved = True
                        improved = True # Swaps are more disruptive, mark as improved to trigger merge check
                        break
            
            if moved:
                # Check if we can merge now
                new_color, new_nc = try_merge_pass(color, conflict_masks)
                if new_nc < best_nc:
                    best_nc = new_nc
                    best_color = new_color[:]
                    color = new_color
                    # Continue search from this better state
                else:
                    # If no improvement in count, we might have changed the structure
                    # Keep the change for now, it might enable future moves
                    pass
        
        return best_color, best_nc

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() > deadline:
            break
        edges = instance(*spec)
        n = spec[0]
        conflict_masks = build_conflict_masks(edges, n)
        m = len(edges)
        
        if m == 0:
            out.append([])
            continue
            
        best_color = None
        best_nc = float('inf')
        
        # Run multiple restarts
        # Allocate time for this instance
        start_time = time.time()
        instance_time_limit = 40 # seconds per instance roughly
        
        for restart in range(100):
            if time.time() > deadline - 5:
                break
                
            rng = Random(1000 + restart)
            color = dsatur_coloring(conflict_masks, rng)
            color, nc = try_merge_pass(color, conflict_masks)
            
            if nc < best_nc:
                best_nc = nc
                best_color = color[:]
                
            # Local Search
            ls_limit = time.time() + (instance_time_limit / 100) # small budget per restart
            ls_color, ls_nc = local_search_improve(color, conflict_masks, rng, ls_limit)
            
            if ls_nc < best_nc:
                best_nc = ls_nc
                best_color = ls_color[:]
                
            # If we hit the theoretical lower bound (e.g. max degree + 1 for strong? No, strong is higher)
            # The lower bound for strong edge coloring is at least max degree + 1?
            # Actually, $\chi_s(G) \ge \Delta(G) + 1$.
            # For d=4, LB=5. For d=5, LB=6. For d=6, LB=7.
            # The conjecture is 5d^2/4.
            # If we get close to LB, we can stop.
            if best_nc <= spec[1] + 1:
                break

        out.append(best_color)
        
    # Fill remaining if timeout
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        n = spec[0]
        conflict_masks = build_conflict_masks(edges, n)
        m = len(edges)
        if m == 0:
            out.append([])
            continue
        rng = Random(9999)
        color = dsatur_coloring(conflict_masks, rng)
        color, _ = try_merge_pass(color, conflict_masks)
        out.append(color)
        
    return out
