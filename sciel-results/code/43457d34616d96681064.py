import random
import time
import math

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # State: adj (n x n list of 0/1), color_mat (n x n list of int, -1 if no edge)
    
    def init_greedy():
        adj = [[0] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        edges = []
        
        # Shuffle all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Find a valid color
            valid_colors = []
            for c in range(num_colors):
                if is_valid_add(u, v, c, adj, color_mat):
                    valid_colors.append(c)
            
            if valid_colors:
                c = random.choice(valid_colors)
                adj[u][v] = 1
                adj[v][u] = 1
                color_mat[u][v] = c
                color_mat[v][u] = c
                edges.append([u, v, c])
        
        return adj, color_mat, edges

    def is_valid_add(u, v, c, adj, color_mat):
        """Check if adding edge (u,v) with color c keeps graph valid."""
        if adj[u][v]:
            return color_mat[u][v] == c
        
        # Check all 4-cycles u-v-w-x-u
        # Optimize by iterating neighbors
        for w in range(n):
            if w == u or w == v: continue
            if not adj[v][w]: continue
            
            for x in range(n):
                if x == u or x == v or x == w: continue
                if not adj[u][x]: continue
                if not adj[w][x]: continue
                
                # Cycle u-v-w-x-u
                c_vw = color_mat[v][w]
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                if len({c, c_vw, c_wx, c_xu}) != 4:
                    return False
        return True

    def is_valid_recolor(u, v, new_c, adj, color_mat):
        """Check if recoloring edge (u,v) to new_c keeps graph valid."""
        if not adj[u][v]:
            return False
        if color_mat[u][v] == new_c:
            return True
        
        # Check all 4-cycles u-v-w-x-u
        for w in range(n):
            if w == u or w == v: continue
            if not adj[v][w]: continue
            
            for x in range(n):
                if x == u or x == v or x == w: continue
                if not adj[u][x]: continue
                if not adj[w][x]: continue
                
                # Cycle u-v-w-x-u
                c_vw = color_mat[v][w]
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                if len({new_c, c_vw, c_wx, c_xu}) != 4:
                    return False
        return True

    # Run SA
    adj, color_mat, edges = init_greedy()
    best_edges = [e[:] for e in edges]
    best_count = len(edges)
    
    T = 1.0
    cool_rate = 0.99
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an edge
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u][v]]
            if not missing:
                continue
            u, v = random.choice(missing)
            valid_colors = [c for c in range(num_colors) if is_valid_add(u, v, c, adj, color_mat)]
            if valid_colors:
                c = random.choice(valid_colors)
                # Apply
                adj[u][v] = 1
                adj[v][u] = 1
                color_mat[u][v] = c
                color_mat[v][u] = c
                edges.append([u, v, c])
                
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = [e[:] for e in edges]
                T = max(T * cool_rate, 0.01)
            else:
                T = max(T * cool_rate, 0.01)
                
        elif move_type < 0.8:
            # Try to recolor
            if not edges:
                continue
            u, v, old_c = random.choice(edges)
            if not adj[u][v]:
                continue
            new_c = random.randint(0, num_colors - 1)
            if new_c == old_c:
                continue
            
            if is_valid_recolor(u, v, new_c, adj, color_mat):
                # Accept recoloring
                color_mat[u][v] = new_c
                color_mat[v][u] = new_c
                # Update edge in list
                for e in edges:
                    if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                        e[2] = new_c
                        break
                T = max(T * cool_rate, 0.01)
            else:
                T = max(T * cool_rate, 0.01)
                
        else:
            # Try to remove an edge
            if not edges:
                continue
            idx = random.randint(0, len(edges) - 1)
            u, v, c = edges[idx]
            
            if adj[u][v]:
                # Removal is always valid.
                # Accept with probability exp(1/T) to escape local optima.
                # If T is high, we accept more.
                # Standard SA: accept worse move with prob exp(-delta/T).
                # Delta = -1 (edges decrease).
                # Prob = exp(1/T).
                
                if random.random() < math.exp(1.0 / T):
                    # Apply removal
                    adj[u][v] = 0
                    adj[v][u] = 0
                    color_mat[u][v] = -1
                    color_mat[v][u] = -1
                    edges.pop(idx)
                
                T = max(T * cool_rate, 0.01)

    return best_edges
