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

    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns the set of colors that can be assigned to edge (u, v)
        without violating the rainbow 4-cycle property.
        Assumes (u, v) is NOT currently in the graph.
        """
        forbidden = set()
        # Check all 3-paths v-w-x-u
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
                    return set()
                # Otherwise, c_uv must be different from these 3
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
        
        return set(range(num_colors)) - forbidden

    def is_valid_recolor(u, v, new_c, adj, color_mat, neighbors):
        """
        Checks if recoloring edge (u, v) to new_c is valid.
        (u, v) is currently in the graph.
        """
        old_c = color_mat[u][v]
        if new_c == old_c:
            return True
        
        # The edge (u, v) is part of 4-cycles u-v-w-x-u
        # For each such cycle, the colors of the other 3 edges must be distinct
        # AND distinct from new_c.
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
                    # This would mean the existing graph was invalid, 
                    # which shouldn't happen if we maintain validity.
                    # But if it does, recoloring won't fix the other 3.
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
                c = random.choice(list(valid))
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
        
        # Helper to update structures
        def add_edge(u, v, c):
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            cur_edges.append([u, v, c])
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
            nonlocal cur_count
            cur_count -= 1

        def recolor_edge(u, v, new_c):
            color_mat[u][v] = new_c
            color_mat[v][u] = new_c
            for e in cur_edges:
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    e[2] = new_c
                    break

        max_recolors_no_add = 50
        recolors_no_add = 0
        
        while time_ok():
            improved = False
            
            # Phase 1: Try to add edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            random.shuffle(missing)
            
            for u, v in missing:
                valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                if valid:
                    c = random.choice(list(valid))
                    add_edge(u, v, c)
                    improved = True
                    recolors_no_add = 0
                    break
            
            if improved:
                continue
                
            # Phase 2: Try to recolor edges to enable additions
            # Pick a random existing edge
            if cur_edges:
                idx = random.randint(0, len(cur_edges) - 1)
                eu, ev, ec = cur_edges[idx]
                
                # Try all other colors
                other_colors = [c for c in range(num_colors) if c != ec]
                random.shuffle(other_colors)
                
                recolored = False
                for new_c in other_colors:
                    if is_valid_recolor(eu, ev, new_c, adj, color_mat, neighbors):
                        recolor_edge(eu, ev, new_c)
                        recolored = True
                        recolors_no_add += 1
                        break
                
                if recolored:
                    # Check if we can add an edge now? 
                    # We just recolor, the loop will restart and try to add.
                    # But to avoid infinite recolor loops without progress, 
                    # we limit consecutive recolors.
                    if recolors_no_add > max_recolors_no_add:
                        break
                else:
                    # No valid recolor for this edge
                    # Try a different edge? 
                    # For simplicity, break and restart the whole search if stuck
                    # or try a few more edges.
                    # Let's try a few more edges before giving up on this state.
                    tried = 1
                    while tried < 10:
                        idx2 = random.randint(0, len(cur_edges) - 1)
                        eu2, ev2, ec2 = cur_edges[idx2]
                        other_colors2 = [c for c in range(num_colors) if c != ec2]
                        random.shuffle(other_colors2)
                        for new_c2 in other_colors2:
                            if is_valid_recolor(eu2, ev2, new_c2, adj, color_mat, neighbors):
                                recolor_edge(eu2, ev2, new_c2)
                                recolored = True
                                recolors_no_add += 1
                                break
                        if recolored:
                            break
                        tried += 1
                    if not recolored:
                        break # Stuck
            else:
                break # No edges to recolor

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
            
        # If we reached a very high density, we might want to try perturbation
        # But for now, simple restarts are robust.
        
    return best_edges
