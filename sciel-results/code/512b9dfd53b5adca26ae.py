import time

class TimeoutError(Exception):
    pass

def construct():
    n_target = 17
    start_time = time.time()
    
    def solve(n):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * len(edges)
        
        def has_path_length_5(u, v, c):
            # Check for simple path of length 5 from u to v in color c
            # Using iterative DFS to avoid recursion overhead/limits?
            # Recursion is fine for depth 5.
            
            # Optimization: If u and v are not connected in a way that allows path length 5, return False.
            # Simple BFS/DFS
            # State: (current, remaining_len, visited_mask)
            # visited_mask is an integer bitmask
            
            def dfs_iterative(start, target, length):
                # Stack items: (node, remaining, visited_mask)
                stack = [(start, length, 1 << start)]
                while stack:
                    node, rem, visited = stack.pop()
                    if rem == 0:
                        if node == target:
                            return True
                        continue
                    if rem < 0:
                        continue
                    
                    # Get neighbors
                    neighbors = adj[c][node]
                    m = neighbors
                    while m:
                        lsb = m & (-m)
                        neighbor = lsb.bit_length() - 1
                        m ^= lsb
                        
                        if neighbor & visited:
                            continue
                        new_visited = visited | (1 << neighbor)
                        if rem == 1 and neighbor == target:
                            return True
                        stack.append((neighbor, rem - 1, new_visited))
                return False
            
            return dfs_iterative(u, v, 5)

        def backtrack(idx):
            if time.time() - start_time > 100:
                raise TimeoutError()
                
            if idx == len(edges):
                return True
                
            u, v = edges[idx]
            
            # Symmetry breaking
            if idx == 0:
                possible = [0]
            elif idx == 1:
                possible = [0, 1]
            elif idx == 2:
                possible = [0, 1, 2]
            else:
                possible = [0, 1, 2, 3]
                
            for c in possible:
                if not has_path_length_5(u, v, c):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if backtrack(idx + 1):
                        return True
                    else:
                        colors[idx] = -1
                        adj[c][u] &= ~(1 << v)
                        adj[c][v] &= ~(1 << u)
                        
            return False
            
        try:
            if backtrack(0):
                return {'n': n, 'colors': colors}
        except TimeoutError:
            return None
        return None

    # Try n=17
    res = solve(17)
    if res:
        return res
        
    # Try n=16
    res = solve(16)
    if res:
        return res
        
    # Fallback to n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
