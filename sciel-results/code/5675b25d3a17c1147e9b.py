import itertools
import random

def construct():
    # Construct Andrásfai graph And(9)
    n = 26
    k = 9
    # Vertices 0..25
    # Edge if (j-i) mod 26 is in {1, 4, 7, 10, 13, 16, 19, 22, 25}
    # i.e., (j-i) mod 3 == 1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    
    # Helper to check triangle-free
    def is_triangle_free(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    # Helper to calculate independence number
    def alpha(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Branch and bound for max independent set
        # We can use a simple greedy to get a lower bound, and DFS for upper bound
        # For n<=40, simple DFS with pruning is sufficient.
        
        best = 0
        
        def dfs(current_set, candidates):
            nonlocal best
            if not candidates:
                if len(current_set) > best:
                    best = len(current_set)
                return
            
            # Pruning: if len(current_set) + len(candidates) <= best, return
            if len(current_set) + len(candidates) <= best:
                return
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with max degree in candidates?
            # Or just first one.
            v = candidates[0]
            rest = candidates[1:]
            
            # Option 1: Include v
            # New candidates are those in rest not adjacent to v
            new_candidates = [u for u in rest if not adj[v][u]]
            dfs(current_set + [v], new_candidates)
            
            # Option 2: Exclude v
            dfs(current_set, rest)

        dfs([], list(range(n)))
        return best

    # Current graph state
    current_n = n
    current_edges = edges
    
    # Greedy expansion
    for _ in range(20): # Limit iterations
        # Find a max independent set
        # For efficiency, we can find one using the same DFS logic but returning the set
        # Or just use a greedy heuristic to find a large independent set, then verify alpha of new graph
        # Since n is small, let's find a max independent set exactly.
        
        adj = [[False]*current_n for _ in range(current_n)]
        for u, v in current_edges:
            adj[u][v] = True
            adj[v][u] = True
            
        # Find max independent set
        max_set = []
        # Using a simple recursive search for max independent set
        def find_max_is(candidates, current_set):
            nonlocal max_set
            if not candidates:
                if len(current_set) > len(max_set):
                    max_set = current_set.copy()
                return
            
            if len(current_set) + len(candidates) <= len(max_set):
                return
                
            v = candidates[0]
            rest = candidates[1:]
            
            # Include v
            new_cands = [u for u in rest if not adj[v][u]]
            find_max_is(new_cands, current_set + [v])
            
            # Exclude v
            find_max_is(rest, current_set)

        find_max_is(list(range(current_n)), [])
        
        I = max_set
        
        # Create new vertex
        new_n = current_n + 1
        new_vertex = current_n
        
        # New edges: connect new_vertex to all in I
        new_edges = current_edges.copy()
        for v in I:
            new_edges.append([new_vertex, v])
            
        # Check if new graph is valid
        # 1. Triangle-free? Yes, because I is independent.
        # 2. Alpha < 10?
        new_alpha = alpha(new_n, new_edges)
        
        if new_alpha < 10:
            current_n = new_n
            current_edges = new_edges
        else:
            break
            
    return {'n': current_n, 'edges': current_edges}
