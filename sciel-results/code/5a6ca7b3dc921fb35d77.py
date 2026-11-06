"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    deadline = time.time() + 55  # 55s safety margin
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to list for faster iteration
    adj_list = [list(s) for s in adj]
    
    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj_list[node]:
                    if not visited_comp[nb]:
                        visited_comp[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    # Sort components by size descending (search largest first)
    components.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    
    # Multiple greedy strategies
    def greedy_dead_end(start, adj_list, visited=None):
        """Prefer neighbors with fewest unvisited options."""
        if visited is None:
            visited = set()
        path = [start]
        visited = visited | {start}
        while True:
            last = path[-1]
            candidates = [nb for nb in adj_list[last] if nb not in visited]
            if not candidates:
                break
            candidates.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited]))
            path.append(candidates[0])
            visited.add(candidates[0])
        return path
    
    def greedy_max_degree(start, adj_list, visited=None):
        """Prefer neighbors with most unvisited options."""
        if visited is None:
            visited = set()
        path = [start]
        visited = visited | {start}
        while True:
            last = path[-1]
            candidates = [nb for nb in adj_list[last] if nb not in visited]
            if not candidates:
                break
            candidates.sort(key=lambda x: -len([y for y in adj_list[x] if y not in visited]))
            path.append(candidates[0])
            visited.add(candidates[0])
        return path
    
    def greedy_random(start, adj_list, visited=None):
        """Random neighbor selection."""
        if visited is None:
            visited = set()
        path = [start]
        visited = visited | {start}
        import random
        rng = random.Random(42)
        while True:
            last = path[-1]
            candidates = [nb for nb in adj_list[last] if nb not in visited]
            if not candidates:
                break
            path.append(rng.choice(candidates))
            visited.add(path[-1])
        return path
    
    # Get initial greedy solutions using multiple strategies
    for start in range(n):
        for gfunc in [greedy_dead_end, greedy_max_degree, greedy_random]:
            gpath = gfunc(start, adj_list)
            if len(gpath) > best_len:
                best_path = gpath
                best_len = len(gpath)
    
    # Also try extending best path from both ends
    def try_extend_both_ends(path, adj_list):
        """Try to extend the path from both endpoints."""
        path = list(path)
        visited = set(path)
        changed = True
        while changed:
            changed = False
            # Try extending from start
            if path:
                first = path[0]
                candidates = [nb for nb in adj_list[first] if nb not in visited]
                if candidates:
                    candidates.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited]))
                    path.insert(0, candidates[0])
                    visited.add(candidates[0])
                    changed = True
            # Try extending from end
            if path:
                last = path[-1]
                candidates = [nb for nb in adj_list[last] if nb not in visited]
                if candidates:
                    candidates.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited]))
                    path.append(candidates[0])
                    visited.add(candidates[0])
                    changed = True
        return path
    
    # Extend the best path found
    extended = try_extend_both_ends(best_path, adj_list)
    if len(extended) > best_len:
        best_path = extended
        best_len = len(best_path)
    
    # DFS with better heuristics
    def dfs(path, visited, comp_size):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        
        cur_len = len(path)
        if cur_len > best_len:
            best_path = path[:]
            best_len = cur_len
        
        # Pruning: even if we visit all remaining in component, can we beat best?
        remaining = comp_size - cur_len
        if cur_len + remaining <= best_len:
            return
        
        last = path[-1]
        # Get unvisited neighbors
        unvisited_neighbors = [nb for nb in adj_list[last] if nb not in visited]
        if not unvisited_neighbors:
            return
        
        # Sort by number of unvisited neighbors (ascending) - prefer dead ends
        unvisited_neighbors.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited]))
        
        for neighbor in unvisited_neighbors:
            path.append(neighbor)
            visited.add(neighbor)
            dfs(path, visited, comp_size)
            path.pop()
            visited.remove(neighbor)
            if time.time() > deadline:
                return
    
    # Try each component, starting with largest
    for comp in components:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        if comp_size <= best_len:
            break
        
        # Try vertices in this component as start, ordered by degree ascending (likely endpoints)
        comp_vertices = sorted(comp, key=lambda x: len(adj_list[x]))
        
        for start in comp_vertices:
            if time.time() > deadline:
                break
            path = [start]
            visited = {start}
            dfs(path, visited, comp_size)
    
    # Final attempt: extend best path again
    if best_path:
        extended = try_extend_both_ends(best_path, adj_list)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
    
    return best_path if best_path else [0]
# EVOLVE-BLOCK-END
