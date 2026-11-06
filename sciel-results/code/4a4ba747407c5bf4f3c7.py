import sys
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edges = P['edge_order']
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin
    
    # Map edge to its 1-based index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_to_idx[(u, v)] = i + 1
    
    def build_adjacency(K):
        """Build adjacency list and degree array for first K edges."""
        adj = [set() for _ in range(n)]
        for i in range(K):
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def check_feasible(adj):
        """Quick feasibility checks: min degree >= 2, connectivity."""
        for i in range(n):
            if len(adj[i]) < 2:
                return False
        # Connectivity via BFS
        visited = [False] * n
        visited[0] = True
        stack = [0]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        return count == n
    
    def find_hamilton_cycle(K, adj=None):
        """
        Finds a Hamilton cycle in the graph formed by the first K edges.
        Uses enhanced backtracking with multiple pruning strategies.
        Returns the cycle as a list of vertices, or None if not found.
        """
        if adj is None:
            adj = build_adjacency(K)
        
        if not check_feasible(adj):
            return None
        
        # Convert adjacency to lists for faster iteration
        adj_list = [list(s) for s in adj]
        
        # Warnsdorff's heuristic with additional pruning
        path = [0]
        used = [False] * n
        used[0] = True
        
        # Count unvisited neighbors for each vertex (dynamic)
        unvisited_deg = [len(adj_list[i]) for i in range(n)]
        unvisited_deg[0] = 0  # start vertex is used
        
        def backtrack(current, depth):
            if depth == n:
                # Check if we can close the cycle
                return 0 in adj[current]
            
            # Get unvisited neighbors of current
            neighbors = adj_list[current]
            unvisited_neighbors = []
            for nb in neighbors:
                if not used[nb]:
                    unvisited_neighbors.append(nb)
            
            if not unvisited_neighbors:
                return False
            
            # Warnsdorff: sort by number of unvisited neighbors (ascending)
            # Tie-break: prefer vertices with fewer total degree
            unvisited_neighbors.sort(key=lambda x: (unvisited_deg[x], len(adj_list[x])))
            
            # Pruning: if only one unvisited vertex remains and it's not the last to place,
            # check it can connect to 0
            if len(unvisited_neighbors) == 1 and depth < n - 1:
                last_v = unvisited_neighbors[0]
                # This vertex must eventually connect to 0, but it's not the last step
                # Actually this pruning is tricky; skip for now
            
            for nb in unvisited_neighbors:
                # Additional pruning: if this is the second-to-last vertex,
                # the remaining unvisited vertex must connect to both nb and 0
                if depth == n - 2:
                    # Find the one remaining unvisited vertex
                    remaining = None
                    for v in range(n):
                        if not used[v] and v != nb:
                            remaining = v
                            break
                    if remaining is not None:
                        if nb not in adj[remaining] or 0 not in adj[remaining]:
                            continue
                
                path.append(nb)
                used[nb] = True
                unvisited_deg[nb] = 0
                # Update degrees of neighbors
                for neighbor in adj_list[nb]:
                    if not used[neighbor]:
                        unvisited_deg[neighbor] -= 1
                
                if backtrack(nb, depth + 1):
                    return True
                
                path.pop()
                used[nb] = False
                unvisited_deg[nb] = len(adj_list[nb])
                # Restore degrees
                for neighbor in adj_list[nb]:
                    if not used[neighbor]:
                        unvisited_deg[neighbor] += 1
            
            return False
        
        if backtrack(0, 1):
            return path
        return None
    
    # Binary search for minimal K
    low = n
    high = num_edges
    best_cycle = None
    best_K = num_edges + 1
    
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        cycle = find_hamilton_cycle(mid)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            high = mid - 1
        else:
            low = mid + 1
    
    if best_cycle is None:
        # Try progressively larger K values if binary search failed
        for K in range(n, num_edges + 1):
            if time.time() >= deadline:
                break
            cycle = find_hamilton_cycle(K)
            if cycle is not None:
                best_cycle = cycle
                best_K = K
                break
    
    if best_cycle is None:
        # Fallback: try the full graph
        cycle = find_hamilton_cycle(num_edges)
        if cycle is not None:
            best_cycle = cycle
        else:
            # Last resort: identity permutation (may not be valid but avoids crash)
            best_cycle = list(range(n))
    
    return best_cycle

# EVOLVE-BLOCK-END
