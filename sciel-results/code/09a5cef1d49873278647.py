import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # We will try to find a solution for n from 17 down to 9
    # If we find one, we return it.
    # We use a greedy + local search heuristic.
    
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this n
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize random coloring
            colors = [[-1]*n for _ in range(n)]
            adj = [ [set() for _ in range(n)] for _ in range(4) ]
            
            def add_edge(c, i, j):
                adj[c][i].add(j)
                adj[c][j].add(i)
                colors[i][j] = c
                colors[j][i] = c

            def remove_edge(c, i, j):
                adj[c][i].remove(j)
                adj[c][j].remove(i)
                colors[i][j] = -1
                colors[j][i] = -1

            # Greedy Construction
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            # Helper to count paths of length 5 between u and v in color c
            def count_paths_5(c, u, v):
                count = 0
                # DFS from u, depth 0, visited {u}
                # We want to reach v at depth 5
                # We can use a recursive helper
                def dfs_count(node, depth, visited):
                    if depth == 5:
                        return 1 if v in adj[c][node] and v not in visited else 0
                    total = 0
                    for nb in adj[c][node]:
                        if nb not in visited and nb != v: # Don't go to v before depth 5
                            visited.add(nb)
                            total += dfs_count(nb, depth+1, visited)
                            visited.remove(nb)
                    return total
                
                return dfs_count(u, 0, {u})

            for u, v in edges:
                best_c = -1
                best_count = float('inf')
                for c in range(4):
                    count = count_paths_5(c, u, v)
                    if count < best_count:
                        best_count = count
                        best_c = c
                add_edge(best_c, u, v)
                
            # Local Search
            # We try to eliminate C6s by moving edges.
            # We iterate until no C6s are found or time limit.
            for _ in range(1000):
                if time.time() > deadline:
                    break
                    
                # Find a C6 in any color
                c6_found = None
                c_found = -1
                for c in range(4):
                    # Find a C6 in color c
                    # We iterate over edges and check for paths of length 5
                    found_c6 = False
                    for u in range(n):
                        if found_c6: break
                        for v in adj[c][u]:
                            if u < v:
                                # Check if there is a path of length 5
                                # We can do a DFS to find one
                                path = [u]
                                def dfs_find(node, depth, visited):
                                    nonlocal found_c6
                                    if found_c6: return
                                    if depth == 5:
                                        if v in adj[c][node] and v not in visited:
                                            path.append(v)
                                            found_c6 = True
                                            return
                                        return
                                    for nb in adj[c][node]:
                                        if nb not in visited and nb != v:
                                            visited.add(nb)
                                            path.append(nb)
                                            dfs_find(nb, depth+1, visited)
                                            path.pop()
                                            visited.remove(nb)
                                            if found_c6: return
                                        
                                dfs_find(u, 0, {u})
                                if found_c6:
                                    c6_found = path
                                    c_found = c
                                    break
                    if found_c6:
                        break
                        
                if not c6_found:
                    # No C6s found!
                    # Verify
                    flat_colors = []
                    for i in range(n):
                        for j in range(i+1, n):
                            flat_colors.append(colors[i][j])
                    return {'n': n, 'colors': flat_colors}
                    
                # We found a C6 in color c_found
                # Pick a random edge from the C6
                idx = random.randint(0, 5)
                u = c6_found[idx]
                v = c6_found[(idx+1)%6]
                
                # Try to move (u,v) to another color
                # Remove (u,v) from c_found
                remove_edge(c_found, u, v)
                
                # Find the best new color
                best_new_c = -1
                best_count = float('inf')
                for c in range(4):
                    if c == c_found: continue
                    count = count_paths_5(c, u, v)
                    if count < best_count:
                        best_count = count
                        best_new_c = c
                        
                # Add to best_new_c
                add_edge(best_new_c, u, v)
                
                # If best_count was 0, we reduced the number of C6s (or kept it 0 for this edge)
                # But we might have created C6s elsewhere? No, only involving (u,v).
                # And we destroyed C6s in c_found involving (u,v).
                # So this move is likely good.
                
    # If we didn't find a solution, return n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
