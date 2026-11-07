# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges_set = set()
    for e in P['edges']:
        base_edges_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Initial adjacency
    initial_adj = [set() for _ in range(n)]
    for u, v in base_edges_set:
        initial_adj[u].add(v)
        initial_adj[v].add(u)
        
    # We will search for minimum k
    # Upper bound: path 24 needs roughly 24/4 = 6 edges? 
    # Let's try k from 1 to 15
    
    solution_edges = []
    
    def get_diameter_and_pair(adj):
        # BFS from each node
        max_d = 0
        u_star = 0
        v_star = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            # Find max dist
            for i in range(n):
                if dist[i] > max_d:
                    max_d = dist[i]
                    u_star = start
                    v_star = i
        return max_d, u_star, v_star

    def bfs_from(adj, start):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def is_triangle_free(adj, u, v):
        return len(adj[u] & adj[v]) == 0

    def dfs(adj, current_edges, depth, max_depth, solution):
        # Check if solved
        diam, u, v = get_diameter_and_pair(adj)
        if diam <= 4:
            solution.extend(current_edges)
            return True
            
        if depth == max_depth:
            return False
            
        # Get distances from u and v to all nodes to evaluate candidates
        dist_u = bfs_from(adj, u)
        dist_v = bfs_from(adj, v)
        current_uv_dist = dist_u[v]
        
        candidates = []
        # Iterate all possible edges
        for a in range(n):
            for b in range(a + 1, n):
                if (a, b) in base_edges_set:
                    continue
                if (a, b) in current_edges:
                    continue
                # Check triangle free
                if not is_triangle_free(adj, a, b):
                    continue
                
                # Check if it reduces dist(u,v)
                # New dist via a-b: min(dist_u[a] + 1 + dist_v[b], dist_u[b] + 1 + dist_v[a])
                new_dist_1 = dist_u[a] + 1 + dist_v[b]
                new_dist_2 = dist_u[b] + 1 + dist_v[a]
                new_uv = min(new_dist_1, new_dist_2)
                
                if new_uv < current_uv_dist:
                    # It is a candidate
                    # Score: how much it reduces
                    reduction = current_uv_dist - new_uv
                    candidates.append((reduction, a, b))
                    
        # Sort by reduction descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for reduction, a, b in candidates:
            # Add edge
            adj[a].add(b)
            adj[b].add(a)
            current_edges.append((a, b))
            
            if dfs(adj, current_edges, depth + 1, max_depth, solution):
                return True
                
            # Remove edge
            adj[a].remove(b)
            adj[b].remove(a)
            current_edges.pop()
            
        return False

    for k in range(1, 16):
        adj_copy = [set(s) for s in initial_adj]
        sol = []
        if dfs(adj_copy, [], 0, k, sol):
            # Convert to list of lists
            return [[u, v] for u, v in sol]
            
    return [] # Should not happen

# EVOLVE-BLOCK-END
