"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deg = [len(adj[v]) for v in range(n)]
    
    # Deadline guard: 90 seconds max, use 85s for safety
    deadline = time.time() + 85.0
    
    def bfs_path(u, v, forbidden):
        """Find shortest path from u to v avoiding forbidden set."""
        if u == v:
            return [u]
        visited = {u}
        parent = {u: None}
        queue = deque([u])
        while queue:
            curr = queue.popleft()
            if curr == v:
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for neighbor in adj[curr]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)
        return None
    
    def try_branch_set(branches):
        """Try to find valid paths for the given branch set. Returns result dict or None."""
        if len(branches) < 2:
            return None
        
        pairs = list(combinations(branches, 2))
        if not pairs:
            return None
        
        import random
        random.seed(42)
        
        # Try natural order first, then many random orders
        orderings = [list(pairs)]
        for _ in range(100):
            shuffled = list(pairs)
            random.shuffle(shuffled)
            orderings.append(shuffled)
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            used_interiors = set()
            paths = {}
            success = True
            
            for (u, v) in ordering:
                forbidden_set = used_interiors | (set(branches) - {u, v})
                path = bfs_path(u, v, forbidden_set)
                
                if path is None:
                    success = False
                    break
                
                # Check interior vertices
                conflict = False
                for vertex in path[1:-1]:
                    if vertex in used_interiors:
                        conflict = True
                        break
                
                if conflict:
                    success = False
                    break
                
                used_interiors.update(path[1:-1])
                paths[(u, v)] = path
            
            if success:
                branch_list = sorted(branches)
                path_list = []
                for (u, v) in combinations(branch_list, 2):
                    path_list.append(paths[(u, v)])
                return {'branches': branch_list, 'paths': path_list}
        
        return None
    
    all_vertices = list(range(n))
    best_result = None
    best_size = 0
    
    import random
    random.seed(123)
    
    # Strategy 1: Greedy extension, high-degree vertices first
    vertices_by_degree = sorted(all_vertices, key=lambda x: -deg[x])
    
    for start_vertex in vertices_by_degree:
        if time.time() > deadline:
            break
        
        current_branches = [start_vertex]
        candidates = [v for v in vertices_by_degree if v != start_vertex]
        
        for candidate in candidates:
            if time.time() > deadline:
                break
            new_branches = current_branches + [candidate]
            result = try_branch_set(new_branches)
            if result is not None:
                current_branches = new_branches
                if len(current_branches) > best_size:
                    best_size = len(current_branches)
                    best_result = result
    
    # Strategy 2: Randomized greedy with many restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(all_vertices)
        current_branches = []
        for candidate in all_vertices:
            if time.time() > deadline:
                break
            new_branches = current_branches + [candidate]
            if len(new_branches) < 2:
                current_branches = new_branches
                continue
            result = try_branch_set(new_branches)
            if result is not None:
                current_branches = new_branches
                if len(current_branches) > best_size:
                    best_size = len(current_branches)
                    best_result = result
    
    # Strategy 3: Local search - try to add vertices to best solution
    if best_result is not None and time.time() < deadline:
        current_branches = list(best_result['branches'])
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for candidate in vertices_by_degree:
                if candidate in current_branches:
                    continue
                if time.time() > deadline:
                    break
                new_branches = current_branches + [candidate]
                result = try_branch_set(new_branches)
                if result is not None:
                    current_branches = new_branches
                    if len(current_branches) > best_size:
                        best_size = len(current_branches)
                        best_result = result
                    improved = True
                    break
    
    # Fallback
    if best_result is None:
        for u, v in edges:
            best_result = {'branches': [u, v], 'paths': [[u, v]]}
            break
    
    # Ensure types are correct
    if best_result is not None:
        best_result['branches'] = [int(v) for v in best_result['branches']]
        best_result['paths'] = [[int(v) for v in p] for p in best_result['paths']]
    
    return best_result
# EVOLVE-BLOCK-END
