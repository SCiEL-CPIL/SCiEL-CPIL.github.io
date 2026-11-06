"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    degree = [len(adj[i]) for i in range(n)]
    
    # Find connected components
    visited_comp = set()
    components = []
    for i in range(n):
        if i not in visited_comp and degree[i] > 0:
            comp = []
            queue = deque([i])
            visited_comp.add(i)
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nb in adj[node]:
                    if nb not in visited_comp:
                        visited_comp.add(nb)
                        queue.append(nb)
            components.append(comp)
    
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    deadline = time.time() + 85  # 5s safety margin
    
    if not components:
        return [0]
    
    largest_comp = components[0]
    comp_set = set(largest_comp)
    comp_adj = {v: adj[v] & comp_set for v in largest_comp}
    comp_degree = {v: len(comp_adj[v]) for v in largest_comp}
    
    def unvisited_neighbor_count(v, visited):
        """Count unvisited neighbors of v."""
        count = 0
        for u in comp_adj[v]:
            if u not in visited:
                count += 1
        return count
    
    def is_safe_choice(v, visited, time_limit):
        """
        Check if choosing v as next vertex is safe (doesn't strand vertices).
        A vertex u is stranded if after visiting v, u has no unvisited neighbors
        other than potentially v itself (which is now visited).
        """
        if time.time() > time_limit:
            return True
        
        # After adding v to visited, check each unvisited neighbor of v
        # If any unvisited vertex w (not adjacent to v) becomes disconnected
        # from all other unvisited vertices, it's stranded.
        # More practical check: for each unvisited neighbor u of v,
        # count its unvisited neighbors excluding v. If 0, u must be visited next.
        
        new_visited = visited | {v}
        
        # Check if any unvisited vertex becomes isolated in the unvisited subgraph
        # after removing v
        stranded_count = 0
        for u in comp_adj[v]:
            if u in visited:
                continue
            # u is unvisited and adjacent to v
            # Count u's unvisited neighbors excluding v
            cnt = 0
            for w in comp_adj[u]:
                if w not in new_visited:
                    cnt += 1
            if cnt == 0:
                # u would be stranded unless visited immediately after v
                stranded_count += 1
        
        # If more than 1 vertex would be stranded, we can't visit all of them
        if stranded_count > 1:
            return False
        
        # Check connectivity of unvisited subgraph after removing v
        # (only needed if v is an articulation point)
        unvisited_after = comp_set - new_visited
        if not unvisited_after:
            return True
        
        # Quick connectivity check
        start = next(iter(unvisited_after))
        seen = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in comp_adj[node]:
                if nb in unvisited_after and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        
        # If not all unvisited vertices are reachable, check if the disconnected
        # parts contain only the stranded vertices
        if seen != unvisited_after:
            disconnected = unvisited_after - seen
            # The disconnected vertices must all be neighbors of v that are stranded
            for d in disconnected:
                if d not in comp_adj[v]:
                    return False
            # Check that the stranded vertices are exactly those in disconnected
            # that have no other unvisited neighbors
            for d in disconnected:
                has_other_unvisited = False
                for w in comp_adj[d]:
                    if w not in new_visited and w != v:
                        has_other_unvisited = True
                        break
                if has_other_unvisited:
                    return False
        
        return True
    
    def dfs_search(start, time_limit):
        """DFS with Warnsdorff heuristic and safety pruning."""
        nonlocal best_path, best_len
        
        path = [start]
        visited = {start}
        
        def _dfs():
            nonlocal best_path, best_len
            if time.time() > time_limit:
                return False
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            current = path[-1]
            unvisited_neighbors = [v for v in comp_adj[current] if v not in visited]
            
            if not unvisited_neighbors:
                return True
            
            # Warnsdorff: sort by number of unvisited neighbors (fewer first)
            # Add safety check as secondary criterion
            def sort_key(v):
                unc = unvisited_neighbor_count(v, visited)
                # Prefer safe moves; unsafe moves get higher key
                safe = 0 if is_safe_choice(v, visited, time_limit) else 1
                return (safe, unc, v)
            
            unvisited_neighbors.sort(key=sort_key)
            
            for v in unvisited_neighbors:
                if time.time() > time_limit:
                    return True
                
                path.append(v)
                visited.add(v)
                
                if not _dfs():
                    return True
                
                path.pop()
                visited.remove(v)
            
            return True
        
        _dfs()
        return best_path
    
    def two_end_extend(path, visited, time_limit):
        """Try to extend path from both ends greedily with safety checks."""
        nonlocal best_path, best_len
        path = list(path)
        visited = set(visited)
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            old_len = len(path)
            
            # Extend left
            current = path[0]
            candidates = [v for v in comp_adj[current] if v not in visited]
            if candidates:
                candidates.sort(key=lambda v: (is_safe_choice(v, visited, time_limit), unvisited_neighbor_count(v, visited), v))
                path.insert(0, candidates[0])
                visited.add(candidates[0])
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
                changed = True
            
            # Extend right
            if time.time() >= time_limit:
                break
            current = path[-1]
            candidates = [v for v in comp_adj[current] if v not in visited]
            if candidates:
                candidates.sort(key=lambda v: (is_safe_choice(v, visited, time_limit), unvisited_neighbor_count(v, visited), v))
                path.append(candidates[0])
                visited.add(candidates[0])
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
                changed = True
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    # Phase 1: Quick greedy to get a decent initial path
    if time.time() < deadline - 50:
        sorted_vertices = sorted(largest_comp, key=lambda v: (comp_degree[v], v))
        start = sorted_vertices[0]
        path = [start]
        visited = {start}
        current = start
        while time.time() < deadline - 50:
            candidates = [v for v in comp_adj[current] if v not in visited]
            if not candidates:
                break
            candidates.sort(key=lambda v: (is_safe_choice(v, visited, time.time() + 1), unvisited_neighbor_count(v, visited), v))
            current = candidates[0]
            path.append(current)
            visited.add(current)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Extend from both ends
        if best_path and time.time() < deadline - 40:
            two_end_extend(best_path, set(best_path), time.time() + 5)
    
    # Phase 2: DFS from multiple starts, prioritizing low-degree vertices
    sorted_vertices = sorted(largest_comp, key=lambda v: (comp_degree[v], v))
    
    num_starts = min(len(sorted_vertices), 30)
    time_per_start = (deadline - time.time() - 10) / max(num_starts, 1)
    time_per_start = min(time_per_start, 2.5)
    
    for i, start in enumerate(sorted_vertices[:num_starts]):
        if time.time() > deadline - 10:
            break
        if comp_degree[start] == 0:
            continue
        
        remaining_time = min(time_per_start, deadline - time.time() - 5)
        if remaining_time < 0.1:
            break
        
        dfs_search(start, time.time() + remaining_time)
        
        if best_path and time.time() < deadline - 5:
            two_end_extend(best_path, set(best_path), time.time() + 1.5)
    
    # Phase 3: Randomized restarts
    if time.time() < deadline - 5:
        rng = random.Random(123)
        attempts = 0
        while time.time() < deadline - 5 and attempts < 40:
            start = rng.choice(largest_comp)
            if comp_degree[start] == 0:
                attempts += 1
                continue
            
            remaining_time = min(1.0, deadline - time.time() - 3)
            if remaining_time < 0.1:
                break
            
            dfs_search(start, time.time() + remaining_time)
            
            if best_path and time.time() < deadline - 3:
                two_end_extend(best_path, set(best_path), time.time() + 0.5)
            
            attempts += 1
    
    # Phase 4: Final extension
    if best_path and time.time() < deadline - 1:
        two_end_extend(best_path, set(best_path), deadline - 1)
    
    if not best_path:
        best_path = [largest_comp[0]]
    
    return best_path
# EVOLVE-BLOCK-END
