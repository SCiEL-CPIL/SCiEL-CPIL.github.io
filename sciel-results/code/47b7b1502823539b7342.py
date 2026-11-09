import sys
import time
import random

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    def find_one_c6(n, adj):
        """
        Finds one monochromatic C6.
        Returns (color, vertices_tuple) or None.
        """
        # Iterate over all 6-subsets
        # To speed up, we can randomize the order?
        # Standard order is fine if we stop early.
        
        # Optimization: Check edge count first.
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        for m in range(l+1, n):
                            for p in range(m+1, n):
                                verts = (i, j, k, l, m, p)
                                
                                # Quick edge count check for each color
                                # We can compute edge counts for all 4 colors at once
                                # edges_c = [0, 0, 0, 0]
                                
                                # But we only need to know if *any* color has a C6.
                                # A C6 requires at least 6 edges of that color.
                                
                                # Let's just check colors 0..3
                                for c in range(4):
                                    # Count edges of color c in this set
                                    # If < 6, skip
                                    # We can short-circuit the count
                                    count = 0
                                    # Pairs: (0,1),(0,2)...
                                    # 15 pairs
                                    if (i in adj[c][j]): count += 1
                                    if (i in adj[c][k]): count += 1
                                    if (i in adj[c][l]): count += 1
                                    if (i in adj[c][m]): count += 1
                                    if (i in adj[c][p]): count += 1
                                    if (j in adj[c][k]): count += 1
                                    if (j in adj[c][l]): count += 1
                                    if (j in adj[c][m]): count += 1
                                    if (j in adj[c][p]): count += 1
                                    if (k in adj[c][l]): count += 1
                                    if (k in adj[c][m]): count += 1
                                    if (k in adj[c][p]): count += 1
                                    if (l in adj[c][m]): count += 1
                                    if (l in adj[c][p]): count += 1
                                    if (m in adj[c][p]): count += 1
                                    
                                    if count < 6:
                                        continue
                                        
                                    # Check for C6
                                    # Use DFS
                                    def ham_cycle(node, depth, visited, current_path):
                                        if depth == 6:
                                            return (current_path[0] in adj[c][node])
                                        
                                        for neighbor in adj[c][node]:
                                            if neighbor in visited:
                                                continue
                                            if depth == 5:
                                                if neighbor == current_path[0]:
                                                    return True
                                            else:
                                                visited.add(neighbor)
                                                current_path.append(neighbor)
                                                if ham_cycle(neighbor, depth + 1, visited, current_path):
                                                    return True
                                                current_path.pop()
                                                visited.remove(neighbor)
                                        return False
                                        
                                    visited = {verts[0]}
                                    if ham_cycle(verts[0], 1, visited, [verts[0]]):
                                        return (c, verts)
                                        
        return None

    def solve_local_search(n):
        adj = [ [set() for _ in range(n)] for _ in range(4) ]
        colors = [[-1]*n for _ in range(n)]
        
        # Initialize random
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, 3)
                colors[i][j] = c
                colors[j][i] = c
                adj[c][i].add(j)
                adj[c][j].add(i)
                
        def recolor(i, j, new_c):
            old_c = colors[i][j]
            if old_c == new_c:
                return
            adj[old_c][i].remove(j)
            adj[old_c][j].remove(i)
            adj[new_c][i].add(j)
            adj[new_c][j].add(i)
            colors[i][j] = new_c
            colors[j][i] = new_c
            
        while time.time() - start_time < time_limit - 1.0:
            result = find_one_c6(n, adj)
            if not result:
                # Found a solution
                flat_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        flat_colors.append(colors[i][j])
                return {'n': n, 'colors': flat_colors}
            
            c, verts = result
            # Find an edge of color c in this set
            candidates = []
            for u_idx in range(6):
                for v_idx in range(u_idx+1, 6):
                    u = verts[u_idx]
                    v = verts[v_idx]
                    if colors[u][v] == c:
                        candidates.append((u, v))
                        
            if not candidates:
                # Should not happen
                continue
                
            u, v = random.choice(candidates)
            new_c = random.randint(0, 3)
            while new_c == c:
                new_c = random.randint(0, 3)
                
            recolor(u, v, new_c)

        return None

    # Try n=17
    res = solve_local_search(17)
    if res:
        return res
        
    # Try n=16
    res = solve_local_search(16)
    if res:
        return res
        
    # Try n=15
    res = solve_local_search(15)
    if res:
        return res
        
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
