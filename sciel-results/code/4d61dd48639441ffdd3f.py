"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    
    def get_initial_dist():
        dist = [[10**9 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            for u, v in P['edges']:
                if u == i:
                    dist[i][v] = 1
                if v == i:
                    dist[i][u] = 1
        # Floyd Warshall
        for k in range(n):
            for i in range(n):
                if dist[i][k] == 10**9: continue
                for j in range(n):
                    if dist[k][j] == 10**9: continue
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def get_diameter(dist):
        max_d = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d

    def update_dist_after_edge(dist, u, v):
        new_dist = [row[:] for row in dist]
        for i in range(n):
            for j in range(n):
                d_i_u = dist[i][u]
                d_v_j = dist[v][j]
                if d_i_u + 1 + d_v_j < new_dist[i][j]:
                    new_dist[i][j] = d_i_u + 1 + d_v_j
                d_i_v = dist[i][v]
                d_u_j = dist[u][j]
                if d_i_v + 1 + d_u_j < new_dist[i][j]:
                    new_dist[i][j] = d_i_v + 1 + d_u_j
        return new_dist

    def is_valid_edge(u, v, dist):
        return dist[u][v] != 2

    current_dist = get_initial_dist()
    current_diam = get_diameter(current_dist)
    
    if current_diam <= 4:
        return []

    def search(max_depth):
        # DFS
        def dfs(depth, adj_edges, dist, added_edges):
            if time.time() > deadline:
                return None
            
            diam = get_diameter(dist)
            if diam <= 4:
                return list(added_edges)
            
            if depth >= max_depth:
                return None
            
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if dist[u][v] > 4 and is_valid_edge(u, v, dist):
                        new_dist = update_dist_after_edge(dist, u, v)
                        new_diam = get_diameter(new_dist)
                        gain = diam - new_diam
                        candidates.append((gain, u, v, new_dist))
            
            candidates.sort(key=lambda x: -x[0])
            
            # Branching factor 15
            for gain, u, v, new_dist in candidates[:15]:
                res = dfs(depth + 1, adj_edges, new_dist, added_edges + [(u, v)])
                if res is not None:
                    return res
            return None
        
        return dfs(0, set(), current_dist, [])

    for k in range(1, 11):
        if time.time() > deadline:
            break
        result = search(k)
        if result is not None:
            return [[int(u), int(v)] for u, v in result]
            
    # Fallback Greedy
    dist = current_dist
    added = []
    while get_diameter(dist) > 4:
        if time.time() > deadline:
            break
        best_gain = -1
        best_edge = None
        best_new_dist = None
        diam = get_diameter(dist)
        for u in range(n):
            for v in range(u+1, n):
                if dist[u][v] > 4 and is_valid_edge(u, v, dist):
                    new_dist = update_dist_after_edge(dist, u, v)
                    new_diam = get_diameter(new_dist)
                    gain = diam - new_diam
                    if gain > best_gain:
                        best_gain = gain
                        best_edge = (u, v)
                        best_new_dist = new_dist
        if best_edge is None:
            break
        added.append(best_edge)
        dist = best_new_dist
        
    return [[int(u), int(v)] for u, v in added]

# EVOLVE-BLOCK-END
