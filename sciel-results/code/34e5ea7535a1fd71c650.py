import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def time_ok():
        return time.time() < deadline
    
    best_edges = []
    best_count = 0
    
    # Precompute all 4-cycles as tuples of 4 edges (u,v) where u<v
    # A 4-cycle is (a,b,c,d) with a<b, b<c or b>d, etc.
    # Better: enumerate all sets of 4 vertices that form a C4
    # For vertices a,b,c,d, the 4-cycle is a-b-c-d-a if edges exist
    # We'll use bitmask representation for colors
    
    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns the set of colors that can be assigned to edge (u, v)
        without violating the rainbow 4-cycle property.
        Assumes (u, v) is NOT currently in the graph.
        """
        forbidden = 0  # bitmask
        for w in neighbors[v]:
            if w == u:
                continue
            c_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v or x == w:
                    continue
                if not adj[w][x]:
                    continue
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                # The 3 edges of the path are c_vw, c_wx, c_xu
                # If any two are equal, the cycle cannot be rainbow regardless of c_uv
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return 0
                # Otherwise, c_uv must be different from these 3
                forbidden |= (1 << c_vw) | (1 << c_wx) | (1 << c_xu)
        
        return ((1 << num_colors) - 1) & ~forbidden

    def is_valid_recolor(u, v, new_c, adj, color_mat, neighbors):
        """
        Checks if recoloring edge (u, v) to new_c is valid.
        (u, v) is currently in the graph.
        """
        old_c = color_mat[u][v]
        if new_c == old_c:
            return True
        
        for w in neighbors[v]:
            if w == u:
                continue
            c_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v or x == w:
                    continue
                if not adj[w][x]:
                    continue
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                # The 4 edges are (u,v), (v,w), (w,x), (x,u)
                # Colors: new_c, c_vw, c_wx, c_xu
                # They must be all distinct.
                if new_c == c_vw or new_c == c_wx or new_c == c_xu:
                    return False
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return False
        
        return True

    def run_greedy_init(seed):
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            valid = get_valid_colors(u, v, adj, color_mat, neighbors)
            if valid:
                # Pick a random valid color from the bitmask
                valid_list = [c for c in range(num_colors) if (valid >> c) & 1]
                c = random.choice(valid_list)
                adj[u][v] = True
                adj[v][u] = True
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
        
        return adj, color_mat, neighbors, edges

    def local_search(adj, color_mat, neighbors, edges, seed):
        random.seed(seed)
        cur_edges = [e[:] for e in edges]
        cur_count = len(cur_edges)
        
        # Build edge lookup for fast removal
        edge_set = set()
        for e in cur_edges:
            u, v = e[0], e[1]
            if u > v:
                u, v = v, u
            edge_set.add((u, v))
        
        def add_edge(u, v, c):
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            cur_edges.append([u, v, c])
            if u > v:
                u, v = v, u
            edge_set.add((u, v))
            nonlocal cur_count
            cur_count += 1

        def remove_edge(u, v):
            adj[u][v] = False
            adj[v][u] = False
            color_mat[u][v] = -1
            color_mat[v][u] = -1
            neighbors[u].remove(v)
            neighbors[v].remove(u)
            # Remove from cur_edges
            for i, e in enumerate(cur_edges):
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    cur_edges.pop(i)
                    break
            if u > v:
                u, v = v, u
            edge_set.discard((u, v))
            nonlocal cur_count
            cur_count -= 1

        def recolor_edge(u, v, new_c):
            color_mat[u][v] = new_c
            color_mat[v][u] = new_c
            for e in cur_edges:
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    e[2] = new_c
                    break

        def try_add_any_edge():
            """Try to add any missing edge. Returns True if successful."""
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            random.shuffle(missing)
            for u, v in missing:
                valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                if valid:
                    valid_list = [c for c in range(num_colors) if (valid >> c) & 1]
                    c = random.choice(valid_list)
                    add_edge(u, v, c)
                    return True
            return False

        def try_recolor_any_edge():
            """Try to recolor any existing edge. Returns True if successful."""
            if not cur_edges:
                return False
            # Try a few random edges
            attempts = min(len(cur_edges), 15)
            indices = random.sample(range(len(cur_edges)), attempts)
            for idx in indices:
                eu, ev, ec = cur_edges[idx]
                other_colors = [c for c in range(num_colors) if c != ec]
                random.shuffle(other_colors)
                for new_c in other_colors:
                    if is_valid_recolor(eu, ev, new_c, adj, color_mat, neighbors):
                        recolor_edge(eu, ev, new_c)
                        return True
            return False

        def try_edge_swap():
            """Try to swap: remove one edge, then add another. Returns True if net gain."""
            if not cur_edges:
                return False
            # Pick a random edge to temporarily remove
            idx = random.randint(0, len(cur_edges) - 1)
            eu, ev, ec = cur_edges[idx]
            
            # Temporarily remove
            remove_edge(eu, ev)
            
            # Try to add a different edge
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            random.shuffle(missing)
            added = False
            for u, v in missing:
                valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                if valid:
                    valid_list = [c for c in range(num_colors) if (valid >> c) & 1]
                    c = random.choice(valid_list)
                    add_edge(u, v, c)
                    added = True
                    break
            
            if not added:
                # Re-add the original edge
                add_edge(eu, ev, ec)
                return False
            
            # Check if we can now add MORE edges
            # Try to add up to 2 more edges
            extra_adds = 0
            for _ in range(2):
                if try_add_any_edge():
                    extra_adds += 1
            
            # Net change: -1 (removed) + 1 (added) + extra_adds
            # This is always >= 0, so it's never worse
            return True

        max_no_progress = 30
        no_progress = 0
        
        while time_ok():
            # Phase 1: Try to add edges
            if try_add_any_edge():
                no_progress = 0
                continue
            
            # Phase 2: Try to recolor
            if try_recolor_any_edge():
                no_progress += 1
                if no_progress >= max_no_progress:
                    break
                continue
            
            # Phase 3: Try edge swap
            if try_edge_swap():
                no_progress = 0
                continue
            
            # Stuck
            break

        return cur_edges

    # Main loop
    iterations = 0
    while time_ok():
        iterations += 1
        seed = iterations
        adj, color_mat, neighbors, edges = run_greedy_init(seed)
        
        # Local search
        improved_edges = local_search(adj, color_mat, neighbors, edges, seed)
        
        if len(improved_edges) > best_count:
            best_count = len(improved_edges)
            best_edges = [e[:] for e in improved_edges]
        
    return best_edges
