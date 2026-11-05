import time
import sys
import random
import heapq
from collections import deque

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    # Pinned graph edges from G(100, 1.8/100) with random.Random(748)
    P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)

    # Sort components by size, descending
    components.sort(key=len, reverse=True)

    best_path = []

    for comp in components:
        if time.time() - start_time > time_limit:
            break

        if len(comp) <= len(best_path):
            continue

        comp_set = set(comp)
        comp_nodes = list(comp)

        # Check if it's a tree - use diameter
        edge_count = sum(len(adj[u]) for u in comp) // 2
        if edge_count == len(comp) - 1:
            def bfs_farthest(start, comp_set, adj):
                dist = {start: 0}
                parent = {start: None}
                q = [start]
                farthest = start
                max_dist = 0
                while q:
                    u = q.pop(0)
                    if dist[u] > max_dist:
                        max_dist = dist[u]
                        farthest = u
                    for v in adj[u]:
                        if v in comp_set and v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                return farthest, parent, max_dist

            u1, _, _ = bfs_farthest(comp[0], comp_set, adj)
            u2, parent2, diameter_len = bfs_farthest(u1, comp_set, adj)

            path = []
            cur = u2
            while cur is not None:
                path.append(cur)
                cur = parent2[cur]
            path.reverse()

            if len(path) > len(best_path):
                best_path = path
            continue

        # General case: use multiple strategies
        local_best = []

        # Strategy 1: Multi-start greedy with Warnsdorff heuristic
        # Try starting from each node, but with a time budget per start
        start_budget = 0.5  # seconds per starting node
        
        for start_node in comp_nodes:
            if time.time() - start_time > time_limit:
                break
            
            path = [start_node]
            seen = {start_node}
            current = start_node
            
            while True:
                # Find neighbors not yet visited
                neighbors = [nb for nb in adj[current] if nb in comp_set and nb not in seen]
                if not neighbors:
                    break
                
                # Warnsdorff: pick neighbor with fewest unvisited neighbors
                def count_options(nb):
                    return sum(1 for x in adj[nb] if x in comp_set and x not in seen)
                
                neighbors.sort(key=lambda nb: (count_options(nb), nb))
                current = neighbors[0]
                path.append(current)
                seen.add(current)
            
            if len(path) > len(local_best):
                local_best = path[:]
        
        # Strategy 2: Local search / improvement on the best path found
        # Try to extend or rearrange the path
        if local_best:
            # Try 2-opt style swaps to extend the path
            improved = True
            while improved and time.time() - start_time < time_limit:
                improved = False
                path = local_best[:]
                seen = set(path)
                
                # Try inserting unvisited nodes at various positions
                unvisited = [u for u in comp_nodes if u not in seen]
                
                for pos in range(len(path) + 1):
                    for u in unvisited:
                        # Check if u can be inserted at position pos
                        prev_node = path[pos - 1] if pos > 0 else None
                        next_node = path[pos] if pos < len(path) else None
                        
                        if prev_node is not None and u not in adj[prev_node]:
                            continue
                        if next_node is not None and u not in adj[next_node]:
                            continue
                        
                        # Try inserting
                        new_path = path[:pos] + [u] + path[pos:]
                        # Verify it's a valid path
                        valid = True
                        for i in range(len(new_path) - 1):
                            if new_path[i+1] not in adj[new_path[i]]:
                                valid = False
                                break
                        
                        if valid and len(new_path) > len(local_best):
                            local_best = new_path
                            improved = True
                            break
                    if improved:
                        break
            
            if len(local_best) > len(best_path):
                best_path = local_best[:]

        # Strategy 3: DFS with branch and bound for smaller components or if time permits
        if len(comp) <= 50 and time.time() - start_time < time_limit - 10:
            # Use DFS with pruning
            def dfs_bb(start, comp_set, adj, time_limit):
                best_len = 0
                best_p = []
                
                # Recursive DFS with memoization of visited sets (bitmask for small n)
                # For larger n, use set-based approach
                sys.setrecursionlimit(10000)
                
                def dfs(node, path, seen):
                    nonlocal best_len, best_p
                    
                    if time.time() - start_time > time_limit:
                        return
                    
                    if len(path) > best_len:
                        best_len = len(path)
                        best_p = path[:]
                    
                    # Pruning: can't beat best if remaining + current <= best
                    remaining = len(comp_set) - len(seen)
                    if len(path) + remaining <= best_len:
                        return
                    
                    # Find unvisited neighbors
                    neighbors = [nb for nb in adj[node] if nb in comp_set and nb not in seen]
                    
                    # Sort by heuristic: fewest options first
                    def count_opts(nb):
                        return sum(1 for x in adj[nb] if x in comp_set and x not in seen)
                    
                    neighbors.sort(key=lambda nb: (count_opts(nb), nb))
                    
                    for nb in neighbors:
                        if time.time() - start_time > time_limit:
                            return
                        dfs(nb, path + [nb], seen | {nb})
                
                # Try from each start node with a time slice
                per_start_time = 2.0
                for start in comp:
                    if time.time() - start_time > time_limit:
                        break
                    slice_start = time.time()
                    dfs(start, [start], {start})
                    if time.time() - slice_start > per_start_time:
                        break
                
                return best_len, best_p
            
            bfs_len, bfs_path = dfs_bb(None, comp_set, adj, time_limit)
            if bfs_len > len(best_path):
                best_path = bfs_path

    return best_path if best_path else [0]
