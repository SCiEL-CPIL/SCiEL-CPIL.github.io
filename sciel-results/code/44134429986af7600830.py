"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    n = P['n']
    
    # Build adjacency from path
    adj = [set() for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        base_edges.add(frozenset((u, v)))
    
    # All possible non-adjacent edges (u < v, not already in base)
    all_candidate_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if frozenset((u, v)) not in base_edges:
                all_candidate_edges.append((u, v))
    
    def is_triangle_free(adj, u, v):
        # Check if adding edge (u,v) creates a triangle
        common = adj[u] & adj[v]
        return len(common) == 0
    
    def bfs_diameter(adj, n):
        # Compute diameter via BFS from each node
        max_diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            # Check connectivity and update max diameter
            for i in range(n):
                if dist[i] == -1:
                    return -1  # Not connected
                max_diam = max(max_diam, dist[i])
        return max_diam
    
    def check_valid(adj, n):
        # Check triangle-free and diameter <= 4 and connected
        # Triangle-free check
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    if len(adj[u] & adj[v]) > 0:
                        return False
        # Diameter check
        diam = bfs_diameter(adj, n)
        if diam == -1:
            return False
        return diam <= 4
    
    # Greedy approach: start with path, add edges one by one
    # Priority: edges that reduce the maximum distance the most
    
    def get_max_dist_pair(adj, n):
        """Find the pair with maximum BFS distance, and the distance."""
        max_dist = 0
        max_pair = None
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            for i in range(n):
                if dist[i] > max_dist:
                    max_dist = dist[i]
                    max_pair = (start, i)
        return max_dist, max_pair
    
    # Try to find a good solution with greedy + local search
    best_solution = None
    best_count = float('inf')
    
    # First, try a structured approach: add edges that connect far-apart vertices
    # For a path of 24 nodes, to get diameter 4, we need shortcuts
    
    # Let's try a more systematic greedy:
    # At each step, find the edge that when added, most reduces the diameter
    # while keeping the graph triangle-free
    
    import copy
    
    def greedy_construct():
        adj_copy = [set(s) for s in adj]
        added = []
        
        for iteration in range(50):  # Max 50 edges
            if time.time() > deadline:
                break
            
            current_diam, _ = get_max_dist_pair(adj_copy, n)
            if current_diam <= 4:
                break
            
            best_edge = None
            best_new_diam = current_diam + 1  # We want to minimize new diameter
            
            # Try all candidate edges not yet added
            for (u, v) in all_candidate_edges:
                if frozenset((u, v)) in base_edges:
                    continue
                if (u, v) in added or (v, u) in added:
                    continue
                # Check triangle-free
                if not is_triangle_free(adj_copy, u, v):
                    continue
                
                # Try adding this edge
                adj_copy[u].add(v)
                adj_copy[v].add(u)
                
                new_diam, _ = get_max_dist_pair(adj_copy, n)
                
                # Undo
                adj_copy[u].remove(v)
                adj_copy[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj_copy[u].add(v)
            adj_copy[v].add(u)
            added.append((u, v))
        
        return added
    
    # Run greedy
    solution = greedy_construct()
    
    # Verify
    adj_check = [set(s) for s in adj]
    for u, v in solution:
        adj_check[u].add(v)
        adj_check[v].add(u)
    
    if check_valid(adj_check, n):
        best_solution = solution
        best_count = len(solution)
    else:
        # Fallback: use the original approach but try to prune
        # Add all odd-parity edges, then try to remove redundant ones
        adj_check = [set(s) for s in adj]
        full_solution = []
        for u in range(n):
            for v in range(u + 1, n):
                if (v - u) % 2 == 1 and frozenset((u, v)) not in base_edges:
                    full_solution.append((u, v))
                    adj_check[u].add(v)
                    adj_check[v].add(u)
        
        if check_valid(adj_check, n):
            # Try to prune
            pruned = list(full_solution)
            for i in range(len(pruned)):
                if time.time() > deadline:
                    break
                u, v = pruned[i]
                # Try removing this edge
                adj_check[u].remove(v)
                adj_check[v].remove(u)
                if check_valid(adj_check, n):
                    pruned.pop(i)
                    i -= 1
                else:
                    adj_check[u].add(v)
                    adj_check[v].add(u)
            best_solution = pruned
            best_count = len(pruned)
    
    # Convert to required format
    if best_solution is None:
        # Last resort: return empty (invalid, but shouldn't happen)
        return []
    
    result = [[int(u), int(v)] for u, v in best_solution]
    return result
# EVOLVE-BLOCK-END
