import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    # We want to maximize n.
    # Let's try to find a valid coloring for n=17.
    # If we fail, we try 16, 15, etc.
    # But backtracking is exponential. We should start with a smaller n and try to extend.
    
    # Actually, let's just run a backtracking search for n=17.
    # If it finds one, return it.
    # If it times out, we need a fallback.
    
    # Fallback: The seed program returns n=9.
    # We can try to improve the seed slightly with a greedy approach first to get a better baseline, 
    # then use the remaining time for backtracking on a larger n.
    
    # Let's implement a function to check if adding edge (u,v) with color c creates a C6 in graph G_c.
    # G_c is represented by a list of bitmasks adj[c][u].
    
    def has_path_len_5(adj, c, u, v):
        """
        Check if there is a simple path of length 5 from u to v in the graph defined by adj[c].
        adj[c] is a list of bitmasks, where adj[c][x] is the set of neighbors of x.
        """
        # We do a DFS from u, keeping track of visited nodes.
        # We want to reach v at exactly depth 5.
        # Since n is small, we can use recursion.
        
        # Optimization: If u and v are not in the same component, return False.
        # Also, if the shortest path is > 5, return False.
        # If the shortest path is < 5, there might still be a longer simple path.
        
        # Let's just DFS with depth limit 5.
        # visited is a bitmask.
        
        def dfs(current, depth, visited):
            if depth == 5:
                return current == v
            if current == v:
                # If we reached v before depth 5, this path is too short.
                # We need a path of EXACTLY length 5?
                # A cycle of length 6 is formed by an edge (u,v) and a path of length 5.
                # If the path is shorter, it forms a shorter cycle, which is allowed.
                # So we must ensure the path is exactly length 5 and simple.
                return False
            
            # Get neighbors of current
            neighbors = adj[c][current]
            # Remove visited nodes
            neighbors &= ~visited
            
            while neighbors:
                # Extract lowest set bit
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                
                if dfs(next_node, depth + 1, visited | lsb):
                    return True
            return False
        
        return dfs(u, 0, 1 << u)

    # Backtracking search
    # We build the coloring for n vertices.
    # colors[i][j] for i < j.
    
    def solve(n):
        # adj[c][u] is bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        # edges list to track assigned colors
        # We assign colors to edges (i, j) in lexicographic order.
        # Let's iterate i from 0 to n-1, and for each i, assign colors for (i, j) for j > i.
        
        # To speed up, we can fix the colors of the first few edges if symmetry allows?
        # For now, plain backtracking.
        
        def backtrack(i, j):
            if time.time() > deadline:
                return None
            
            if i == n - 1:
                if j == n:
                    return True
                j += 1
                if j == n:
                    return backtrack(i + 1, i + 2)
            
            if j == n:
                return backtrack(i + 1, i + 2)
            
            # Try all 4 colors for edge (i, j)
            for c in range(4):
                # Check if adding (i, j) with color c creates a C6
                # We need to check if there is a path of length 5 between i and j in G_c
                if has_path_len_5(adj, c, i, j):
                    continue
                
                # Assign color c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                # Recurse
                result = backtrack(i, j + 1)
                if result:
                    return True
                
                # Unassign
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
            
            return False
        
        return backtrack(0, 1)

    # We want to maximize n.
    # Let's try n=17 first.
    # If it fails, try n=16, etc.
    # But if n=17 takes too long, we might not have time for smaller n.
    # So we should run the search for n=17 with a time limit.
    # If it doesn't finish, we return a valid smaller n.
    
    # To be safe, let's start with a known valid small n and try to extend.
    # Or just try to solve for n=17 directly.
    
    # The seed n=9 is valid.
    # Let's try to find a valid coloring for n=17.
    
    # Note: The search space is huge.
    # We might need to use a heuristic to order vertices or colors.
    # For now, let's just try.
    
    best_n = 9
    best_colors = [min(j - i, 17 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)] # Seed colors for n=9
    
    # Try to solve for n=17
    # We need to store the colors if we succeed.
    # The backtrack function above only returns True/False.
    # Let's modify it to return the colors.
    
    def solve_with_colors(n):
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)] # colors[i][j]
        
        def backtrack(i, j):
            if time.time() > deadline:
                return None
            
            if i == n - 1:
                if j == n:
                    # Success
                    return colors
            
            if j == n:
                res = backtrack(i + 1, i + 2)
                if res: return res
                return None
            
            for c in range(4):
                if has_path_len_5(adj, c, i, j):
                    continue
                
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                colors[i][j] = c
                colors[j][i] = c
                
                res = backtrack(i, j + 1)
                if res:
                    return res
                
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[i][j] = -1
                colors[j][i] = -1
            
            return None
        
        return backtrack(0, 1)

    # Try n=17
    # This might be too slow.
    # Let's try n=10 first to ensure the logic works and is fast.
    # Then try 11, 12...
    
    # Actually, let's just try n=17. If it's too slow, it will time out and return None.
    # Then we try 16.
    
    for target_n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5:
            break
        
        # Reset deadline for this specific solve? No, global deadline is better.
        # But if solve(17) takes 100s and fails, we have 15s left for solve(16).
        
        res = solve_with_colors(target_n)
        if res:
            # Convert 2D list to flat list in lexicographic order
            flat_colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    flat_colors.append(res[i][j])
            return {'n': target_n, 'colors': flat_colors}
        
        # If failed, continue to next smaller n

    # If all fail, return the seed
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
