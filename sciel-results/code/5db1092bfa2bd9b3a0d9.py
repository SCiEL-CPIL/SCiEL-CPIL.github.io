import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_edges = []
    best_count = 0
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def analyze_edge(u, v, adj, color_mat, neighbors):
        """
        Returns (impossible, forbidden_colors)
        impossible: True if no color can make (u,v) valid.
        forbidden_colors: Set of colors that cannot be used for (u,v).
        """
        forbidden = set()
        impossible = False
        
        # Iterate over all potential 4-cycles u-x-y-v-u
        # x is neighbor of u, y is neighbor of v
        for x in neighbors[u]:
            if x == v:
                continue
            for y in neighbors[v]:
                if y == u or y == x:
                    continue
                
                # Check if x and y are connected
                if not adj[x][y]:
                    continue
                
                c_ux = color_mat[u][x]
                c_xy = color_mat[x][y]
                c_yv = color_mat[y][v]
                
                # The cycle colors will be {c_ux, c_xy, c_yv, c_uv}
                # For the cycle to be rainbow, these 4 must be distinct.
                
                # If any of the existing 3 are equal, the cycle can never be rainbow.
                if c_ux == c_xy or c_xy == c_yv or c_ux == c_yv:
                    impossible = True
                    return impossible, forbidden
                
                # c_uv must be different from the other 3
                forbidden.add(c_ux)
                forbidden.add(c_xy)
                forbidden.add(c_yv)
                
        return impossible, forbidden

    def run_greedy(seed):
        random.seed(seed)
        adj = [[0] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        # Shuffle edges to break symmetry
        shuffled_edges = all_possible_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            impossible, forbidden = analyze_edge(u, v, adj, color_mat, neighbors)
            
            if impossible:
                continue
                
            valid_colors = [c for c in range(num_colors) if c not in forbidden]
            
            if valid_colors:
                c = random.choice(valid_colors)
                adj[u][v] = 1
                adj[v][u] = 1
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
                
        return edges, adj, color_mat, neighbors

    def local_search(edges, adj, color_mat, neighbors, seed):
        """
        Try to add more edges by recoloring.
        """
        random.seed(seed)
        edges = [e[:] for e in edges]
        # Rebuild structures from edges just in case, though we pass them in
        adj = [[0] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            
        current_count = len(edges)
        missing_edges = [(u, v) for u, v in all_possible_edges if not adj[u][v]]
        
        # Time limit for local search per restart
        ls_deadline = time.time() + 5.0 
        
        while time.time() < ls_deadline:
            if not missing_edges:
                break
            
            random.shuffle(missing_edges)
            added_any = False
            
            for u, v in missing_edges:
                impossible, forbidden = analyze_edge(u, v, adj, color_mat, neighbors)
                
                if not impossible:
                    valid_colors = [c for c in range(num_colors) if c not in forbidden]
                    if valid_colors:
                        c = random.choice(valid_colors)
                        adj[u][v] = 1
                        adj[v][u] = 1
                        color_mat[u][v] = c
                        color_mat[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        missing_edges.remove((u, v))
                        current_count += 1
                        added_any = True
                        break # Restart loop to re-evaluate others
            
            if not added_any:
                # Try to recolor an existing edge to help a missing one
                # Pick a random missing edge
                u, v = random.choice(missing_edges)
                impossible, forbidden = analyze_edge(u, v, adj, color_mat, neighbors)
                
                if impossible:
                    # Need to fix a path where colors collide
                    # Find a conflicting path
                    fixed = False
                    for x in neighbors[u]:
                        if x == v: continue
                        for y in neighbors[v]:
                            if y == u or y == x: continue
                            if not adj[x][y]: continue
                            c_ux = color_mat[u][x]
                            c_xy = color_mat[x][y]
                            c_yv = color_mat[y][v]
                            
                            # Check if this path causes impossibility
                            if c_ux == c_xy or c_xy == c_yv or c_ux == c_yv:
                                # Try to recolor one of these 3 edges
                                # Candidates: (u,x), (x,y), (y,v)
                                candidates = [(u, x), (x, y), (y, v)]
                                random.shuffle(candidates)
                                
                                for cu, cv in candidates:
                                    old_c = color_mat[cu][cv]
                                    # Try to find a new color for (cu, cv) that doesn't break its own cycles
                                    # This is complex. Simple heuristic: try random colors.
                                    new_c = random.choice([c for c in range(num_colors) if c != old_c])
                                    
                                    # Temporarily change
                                    color_mat[cu][cv] = new_c
                                    color_mat[cv][cu] = new_c
                                    
                                    # Check if this recoloring breaks any existing 4-cycle
                                    valid = True
                                    # Check cycles involving (cu, cv)
                                    for w in neighbors[cu]:
                                        if w == cv: continue
                                        for z in neighbors[cv]:
                                            if z == cu or z == w: continue
                                            if not adj[w][z]: continue
                                            
                                            c_cu_w = color_mat[cu][w]
                                            c_wz = color_mat[w][z]
                                            c_zv = color_mat[z][cv]
                                            
                                            # Colors: new_c, c_cu_w, c_wz, c_zv
                                            # Must be distinct
                                            if len({new_c, c_cu_w, c_wz, c_zv}) < 4:
                                                valid = False
                                                break
                                        if not valid: break
                                    if not valid:
                                        # Revert
                                        color_mat[cu][cv] = old_c
                                        color_mat[cv][cu] = old_c
                                        continue
                                    
                                    # If valid, accept and break
                                    # Update edges list
                                    for e in edges:
                                        if (e[0] == cu and e[1] == cv) or (e[0] == cv and e[1] == cu):
                                            e[2] = new_c
                                            break
                                    fixed = True
                                    break
                                if fixed:
                                    break
                        if fixed:
                            break
                    # If we fixed a path, the edge (u,v) might be addable now
                    # Loop will retry adding (u,v) in next iteration
                else:
                    # Not impossible, but no valid color (forbidden set is full)
                    # This means all 20 colors are forbidden.
                    # We need to remove a color from the forbidden set by recoloring.
                    # Pick a random color in forbidden, find an edge causing it, recolor.
                    if len(forbidden) == num_colors:
                        # Find an edge that contributes a forbidden color
                        # This is harder. Just try random recolors of random edges and see if it helps.
                        # This is a weak fallback.
                        pass
                            
                # If we didn't add or fix, break to avoid infinite loop if stuck
                if not added_any and not fixed if 'fixed' in locals() else not added_any:
                     # If we tried to fix and failed, or didn't try to fix, we might be stuck.
                     # To prevent infinite loop, we can just break if no progress in a few iterations.
                     # For now, break if we didn't add. The 'fixed' logic above sets a flag.
                     pass
                     
        return edges

    restarts = 0
    while time.time() < deadline:
        edges, adj, color_mat, neighbors = run_greedy(restarts)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [e[:] for e in edges]
            
        # Only do local search if we have time and haven't found perfect
        if best_count < 190 and time.time() < deadline - 10:
            improved_edges = local_search(edges, adj, color_mat, neighbors, restarts)
            if len(improved_edges) > best_count:
                best_count = len(improved_edges)
                best_edges = [e[:] for e in improved_edges]
                
        if best_count == 190:
            break
            
        restarts += 1
        
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
