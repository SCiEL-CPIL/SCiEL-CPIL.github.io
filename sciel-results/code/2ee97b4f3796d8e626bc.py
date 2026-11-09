import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # Seed solution
    def get_seed():
        n = 9
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}
    
    def edge_index(n, i, j):
        # Lexicographic order
        # i < j
        # index = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
        return i*(n-1) - i*(i-1)//2 + (j-i-1)
        
    def check_c6(n, colors):
        # Build adjacency lists for each color
        adj = [[set() for _ in range(n)] for _ in range(4)]
        for idx, c in enumerate(colors):
            # Decode idx to (i, j)
            # This is slow. Better to build from i,j loops.
            pass
        
        # Rebuild adj from i,j loops is better for clarity, 
        # but we need to check the specific coloring.
        
        # Let's just iterate i, j
        adj = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = colors[edge_index(n, i, j)]
                adj[c][i].add(j)
                adj[c][j].add(i)
                
        # Check for C6 in each color
        for c in range(4):
            # For each edge (u, v), check if u, v are in a C6
            # A C6 containing (u,v) requires a path of length 5 between u and v
            # Path: u - a - b - d - e - v
            # a in N(u), e in N(v)
            # b in N(a), d in N(e)
            # b == d ? No, distinct vertices.
            # Actually, we just need to find ANY C6.
            # We can check for cycles of length 6 directly.
            
            # Optimization: If degree < 2, no C6.
            # If graph is small, just check.
            
            # Check for C6 by iterating pairs of non-adjacent? No.
            # Iterate all 6-tuples? Too slow.
            
            # Use the property: C6 exists iff there is a path of length 5 between adjacent vertices u, v
            # such that the path is simple and does not use edge (u,v).
            
            for u in range(n):
                for v in adj[c][u]:
                    if u >= v: continue
                    # Check if there is a path of length 5 from u to v in G - (u,v)
                    # DFS from u to v with depth 5
                    # We can use a simple recursive search or BFS.
                    # Since depth is small (5), recursion is fine.
                    
                    def has_path(u, v, depth, visited, adj_c):
                        if depth == 0:
                            return u == v
                        if u == v:
                            return False # Should not happen if we manage visited correctly
                        for w in adj_c[u]:
                            if w in visited:
                                continue
                            if w == v and depth == 1:
                                return True
                            if w != v: # Don't step on v early
                                visited.add(w)
                                if has_path(w, v, depth - 1, visited, adj_c):
                                    return True
                                visited.remove(w)
                        return False

                    # We need path of length 5: u-a-b-d-e-v
                    # So depth 5 from u to v.
                    # Visited must contain u.
                    if has_path(u, v, 5, {u}, adj[c]):
                        return True
                        
        return False

    def try_solve(n):
        # Random search
        num_edges = n * (n - 1) // 2
        best_colors = None
        best_penalty = float('inf')
        
        # We need a penalty function that is faster than check_c6
        # Or just use check_c6 and accept if 0.
        
        # For speed, we can use a local search.
        # Start with random coloring.
        
        for restart in range(100):
            if time.time() > deadline:
                break
                
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Shuffle edges to avoid order bias
                edge_indices = list(range(num_edges))
                random.shuffle(edge_indices)
                
                for idx in edge_indices:
                    if time.time() > deadline:
                        return None
                    
                    old_c = colors[idx]
                    # Try other colors
                    for new_c in [0, 1, 2, 3]:
                        if new_c == old_c:
                            continue
                        colors[idx] = new_c
                        if not check_c6(n, colors):
                            return colors
                        # If we want to minimize penalty, we'd need a penalty counter.
                        # But check_c6 is boolean.
                        # We can just keep the random one if it doesn't improve?
                        # This is a hill climber with boolean fitness.
                        # It will only move if it finds a solution.
                        # This is very slow.
                        
                    colors[idx] = old_c
                    
            # If we didn't find a solution, restart
            
        return None

    # Try n=18
    res = try_solve(18)
    if res:
        return {'n': 18, 'colors': res}
        
    # Try n=17
    res = try_solve(17)
    if res:
        return {'n': 17, 'colors': res}
        
    # Try n=16
    res = try_solve(16)
    if res:
        return {'n': 16, 'colors': res}
        
    # Fallback to seed
    return get_seed()
